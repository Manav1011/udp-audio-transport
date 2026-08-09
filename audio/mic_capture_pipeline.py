"""Microphone capture pipeline — converts native PCM16 mono at the
source's native sample rate into the transport's required 48 kHz / stereo
/ Float32 LE interleaved format.

This module is the Python equivalent of the Android AudioRecord MIC
capture path:

    AudioRecord(MIC, PCM16, mono, native_rate)
        → int16 short[] pcm16 mono
        → float32 normalized [-1.0, 1.0]
        → resample to 48000 Hz if native_rate != 48000
        → duplicate mono into [L, R] stereo
        → float32 LE interleaved bytes
        → 8-byte-frame aligned chunks
        → dispatch (caller's callback)

Conversion rules (verbatim per the spec):

    1. PCM16 sample normalization: signed-16-bit value / 32768.0
       (exactly the standard full-scale / 2^15 normalization).
    2. No gain multiplier is applied.
    3. No DSP: no noise suppression, no AGC, no filtering.
    4. Resampling: linear interpolation, only when native_rate != 48000.
    5. Stereo duplication: L[i] = R[i] = mono[i].
    6. Output alignment: every dispatched chunk is a multiple of 8 bytes
       (one float32 stereo frame = 2 channels * 4 bytes).

The pipeline is intentionally pull-based and stateful: callers feed
raw PCM16 mono chunks via `feed_pcm16_mono(bytes)` and pull emitted
8-byte-aligned chunks via `drain()`. Internal buffering absorbs any
sample-rate mismatch and preserves frame alignment across calls.

The pipeline does NOT depend on pipewire, pw-cat, or any audio device.
It is pure-Python and fully unit-testable.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

import numpy as np

TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 2  # stereo
FLOAT32_BYTES = 4
STEREO_FRAME_BYTES = TARGET_CHANNELS * FLOAT32_BYTES  # 8

# Standard signed-16-bit full-scale normalization. 32768 is the
# positive full-scale denominator (a positive sample of 32767 maps to
# 32767/32768 ≈ 0.99997; the asymmetric range matches the standard
# PCM16 format).
PCM16_FULL_SCALE = 32768.0


@dataclass
class PipelineConfig:
    """Configuration of a MicCapturePipeline.

    native_rate: the sample rate the upstream capture device (Android
        AudioRecord MIC, or pw-cat s16 mono) is producing. Resampling
        is applied only when native_rate != TARGET_SAMPLE_RATE.
    native_channels: must be 1 (mono). The pipeline is intentionally
        mono-only at the front door; stereo duplication happens at the
        back door.
    """
    native_rate: int
    native_channels: int = 1

    def __post_init__(self) -> None:
        if self.native_channels != 1:
            raise ValueError(
                f"native_channels must be 1 (mono); got {self.native_channels}"
            )
        if self.native_rate <= 0:
            raise ValueError(
                f"native_rate must be positive; got {self.native_rate}"
            )


class MicCapturePipeline:
    """Pull-based PCM16-mono-native-rate → Float32-stereo-48k pipeline.

    Every `feed_pcm16_mono()` call appends raw PCM16 mono data to an
    internal buffer. The buffer is converted on demand when `drain()`
    is called. Frame alignment is preserved across calls: any partial
    stereo frame at the end of one drain is held back for the next.

    Lifecycle:
        pipe = MicCapturePipeline(PipelineConfig(native_rate=44100))
        for raw_pcm16_chunk in capture_device:
            pipe.feed_pcm16_mono(raw_pcm16_chunk)
            for float32_stereo_chunk in pipe.drain():
                dispatch(float32_stereo_chunk)
        # End of stream: flush any remaining samples.
        for trailing_chunk in pipe.flush():
            dispatch(trailing_chunk)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        # All-but-the-last stereo-frame slice of the most recent
        # converted output. Held to enforce frame alignment.
        self._stereo_resample_pos: float = 0.0
        # Source-rate mono samples awaiting conversion. We accumulate
        # between feed() calls so partial reads are handled correctly.
        self._pending_mono: list[np.ndarray] = []
        # Number of valid mono samples currently in _pending_mono.
        self._pending_mono_total: int = 0
        # Output-side sampling bookkeeping: where in the SOURCE stream
        # the next output sample corresponds to. Used to drive the
        # linear interpolator across feed() boundaries.
        self._next_source_index: float = 0.0
        # True once feed() has been called for the first time.
        self._started = False

    # -- properties --------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def needs_resample(self) -> bool:
        return self._config.native_rate != TARGET_SAMPLE_RATE

    # -- ingestion ---------------------------------------------------------

    def feed_pcm16_mono(self, pcm16_bytes: bytes) -> None:
        """Append raw PCM16 mono bytes to the pipeline.

        Partial reads are handled: the bytes may be any length, including
        odd values (the underlying AudioRecord.read() may return any
        aligned/unaligned chunk size, though in practice it is always
        even because PCM16 is 2 bytes). We do not require the chunk to
        be a multiple of 2 because the original Android spec says
        "handle partial PCM16 reads correctly" — we tolerate the
        absolute edge case and raise on misaligned input.
        """
        if len(pcm16_bytes) % 2 != 0:
            raise ValueError(
                f"PCM16 input must be a multiple of 2 bytes; "
                f"got {len(pcm16_bytes)}"
            )
        if len(pcm16_bytes) == 0:
            return
        # Decode PCM16 LE signed → float32 in [-1.0, 1.0].
        # np.frombuffer with dtype='<i2' is the canonical, zero-copy
        # int16 LE reader.
        samples_int16 = np.frombuffer(pcm16_bytes, dtype="<i2")
        # Standard signed-16-bit normalization. No gain multiplier.
        samples_f32 = samples_int16.astype(np.float32) / PCM16_FULL_SCALE
        self._pending_mono.append(samples_f32)
        self._pending_mono_total += len(samples_f32)
        self._started = True

    # -- conversion --------------------------------------------------------

    def _stitch_pending_mono(self) -> np.ndarray:
        if not self._pending_mono:
            return np.empty(0, dtype=np.float32)
        # Concatenate lazily. Most feeds are small so this is cheap.
        stitched = np.concatenate(self._pending_mono) if len(self._pending_mono) > 1 \
            else self._pending_mono[0]
        self._pending_mono.clear()
        self._pending_mono_total = 0
        return stitched

    def _consume_resampled(self, mono: np.ndarray, n_samples: int) -> np.ndarray:
        """Pull `n_samples` linear-interpolated values from `mono`,
        advancing the resampler cursor. Returns a float32 array of
        length exactly n_samples.

        When native_rate == TARGET_SAMPLE_RATE no resampling is
        performed: the cursor advances by 1.0 per output sample and we
        return up to (n_samples) straight samples.
        """
        if mono.size == 0 or n_samples <= 0:
            return np.empty(0, dtype=np.float32)
        if not self.needs_resample:
            # No resampling — straight copy.
            out = mono[:n_samples]
            # Drop the consumed samples from the buffer.
            self._pending_mono.append(mono[n_samples:])
            self._pending_mono_total = len(mono) - n_samples
            return out.astype(np.float32, copy=False)
        # Resampling via linear interpolation.
        # The mapping is: out[i] = mono[floor(src_pos)] + frac * (mono[ceil(src_pos)] - mono[floor(src_pos)])
        # where src_pos = _next_source_index + i * (native_rate / target_rate).
        ratio = self._config.native_rate / TARGET_SAMPLE_RATE  # < 1 if upsampling
        src_pos = self._next_source_index
        out = np.empty(n_samples, dtype=np.float32)
        last_src_index = mono.size - 1
        for i in range(n_samples):
            sp = src_pos
            idx = int(sp)  # floor for non-negative sp
            if idx >= last_src_index:
                # Out of source range — clamp to the last sample.
                out[i] = float(mono[last_src_index])
                src_pos += ratio
                continue
            frac = sp - idx
            out[i] = float(mono[idx]) * (1.0 - frac) + float(mono[idx + 1]) * frac
            src_pos += ratio
        # Advance the cursor and split the buffer.
        new_cursor = self._next_source_index + n_samples * ratio
        consumed = int(np.floor(new_cursor))
        # Keep the residual [consumed, new_cursor) source samples.
        residual_count = mono.size - consumed
        if residual_count > 0:
            residual = mono[consumed:].copy()
            self._pending_mono.append(residual)
            self._pending_mono_total = residual.size
        # Store the fractional carry for the next drain.
        self._next_source_index = new_cursor - consumed
        return out

    # -- output ------------------------------------------------------------

    def _mono_to_stereo_bytes(self, mono: np.ndarray) -> bytes:
        """Return the mono float32 sample interleaved into [L,R] stereo
        as float32 LE bytes. Output length is exactly
        len(mono) * STEREO_FRAME_BYTES."""
        if mono.size == 0:
            return b""
        # Build interleaved [L,R] by stacking and transposing.
        stereo = np.empty((mono.size, TARGET_CHANNELS), dtype=np.float32)
        stereo[:, 0] = mono
        stereo[:, 1] = mono
        # numpy's .tobytes() emits float32 LE on x86 and ARM-LE.
        return stereo.tobytes()

    def drain(self) -> bytes:
        """Convert as many pending mono samples as possible into a
        frame-aligned transport-format chunk.

        Rules:
        - If no input has been fed yet, returns b"".
        - The first drain() after a feed() may produce a partial chunk
          if the input was small; subsequent drains amortize the
          partial frame across calls.
        - The returned chunk is always a multiple of STEREO_FRAME_BYTES
          (= 8). Any partial trailing frame is held for the next
          drain/flush.
        """
        if not self._started:
            return b""
        mono = self._stitch_pending_mono()
        if mono.size == 0:
            return b""
        # When native_rate == TARGET_SAMPLE_RATE, we can advance the
        # cursor freely. For resampling, the cursor is held in
        # _next_source_index and we update it via _consume_resampled.
        if self.needs_resample:
            # Determine how many output samples we can produce safely.
            # We need at least ceil(_next_source_index + n * ratio) source
            # samples. We produce as many full frames as possible.
            ratio = self._config.native_rate / TARGET_SAMPLE_RATE
            # Conservative: we can produce up to floor((mono.size - _next_source_index) / ratio)
            # full samples without re-reading.
            available_source = mono.size - self._next_source_index
            if available_source <= 0:
                return b""
            max_out = int(np.floor(available_source / ratio))
            if max_out <= 0:
                return b""
            out = self._consume_resampled(mono, max_out)
        else:
            # No resampling. Produce all of mono, except we still
            # need to handle alignment via the stitcher bookkeeping.
            # The stitcher already moved the unconsumed tail into
            # _pending_mono, so emitting all of mono is correct.
            out = mono
        if out.size == 0:
            return b""
        stereo_bytes = self._mono_to_stereo_bytes(out)
        # Frame-align the output: truncate to the largest multiple of
        # STEREO_FRAME_BYTES and hold back any trailing partial frame.
        n_full_frames = len(stereo_bytes) // STEREO_FRAME_BYTES
        if n_full_frames == 0:
            return b""
        aligned_len = n_full_frames * STEREO_FRAME_BYTES
        # The held-back tail of `out` is the last
        # (len(out) - n_full_frames) mono samples. Put them back as
        # pending so the next drain picks them up.
        held = out[n_full_frames:]
        if held.size > 0:
            self._pending_mono.append(held.astype(np.float32, copy=False))
            self._pending_mono_total += held.size
        return stereo_bytes[:aligned_len]

    def flush(self) -> bytes:
        """Final drain at end-of-stream. Returns any remaining frame-
        aligned stereo bytes. May return b"" if there is no in-flight
        output. Any partial trailing frame is dropped (it would have
        been too small to dispatch anyway)."""
        if not self._started:
            return b""
        out = self.drain()
        # Also pull any leftover stitched-but-unconverted tail.
        mono = self._stitch_pending_mono()
        if mono.size > 0:
            # Apply resampling to whatever is left.
            if self.needs_resample:
                # For the final flush, we want to use every remaining
                # source sample. Compute the floor output count.
                ratio = self._config.native_rate / TARGET_SAMPLE_RATE
                remaining_source = mono.size - self._next_source_index
                if remaining_source > 0:
                    max_out = int(np.floor(remaining_source / ratio))
                    if max_out > 0:
                        tail = self._consume_resampled(mono, max_out)
                        if tail.size > 0:
                            tail_bytes = self._mono_to_stereo_bytes(tail)
                            n_full_frames = len(tail_bytes) // STEREO_FRAME_BYTES
                            out += tail_bytes[:n_full_frames * STEREO_FRAME_BYTES]
            else:
                tail_bytes = self._mono_to_stereo_bytes(mono)
                n_full_frames = len(tail_bytes) // STEREO_FRAME_BYTES
                out += tail_bytes[:n_full_frames * STEREO_FRAME_BYTES]
        # Reset state for any subsequent reuse.
        self._next_source_index = 0.0
        self._pending_mono.clear()
        self._pending_mono_total = 0
        return out


def announce_native_rate(native_rate: int) -> None:
    """Debug-only helper that records the native capture rate and the
    transport format. Kept for backwards compatibility with callers that
    probe-and-announce; emits at DEBUG."""
    log = logging.getLogger("audio-bridge")
    log.debug(
        "mic capture: source=MIC, native=%d Hz / mono / PCM16, "
        "transport=%d Hz / stereo / Float32 LE",
        native_rate, TARGET_SAMPLE_RATE,
    )
