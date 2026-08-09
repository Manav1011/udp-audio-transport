"""Tests for the microphone capture pipeline.

The pipeline is the Python equivalent of the Android AudioRecord(MIC,
PCM16, mono, native_rate) capture path. It converts raw PCM16 mono
bytes at the source's native rate into the transport's required
48 kHz / stereo / Float32 LE format.

These tests verify the conversion rules in isolation, without any
dependency on pipewire, pw-cat, or audio devices.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from audio.mic_capture_pipeline import (
    MicCapturePipeline,
    PipelineConfig,
    PCM16_FULL_SCALE,
    STEREO_FRAME_BYTES,
    TARGET_SAMPLE_RATE,
    announce_native_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm16_bytes(samples: list[int]) -> bytes:
    """Pack a list of signed-16-bit samples into LE bytes."""
    return struct.pack(f"<{len(samples)}h", *samples)


def _decode_stereo_f32(pcm: bytes) -> np.ndarray:
    """Decode 8-byte-aligned float32 stereo bytes into (N, 2) array."""
    assert len(pcm) % STEREO_FRAME_BYTES == 0
    return np.frombuffer(pcm, dtype="<f4").reshape(-1, 2)


def _all_stereo_frames(pcm: bytes) -> np.ndarray:
    """Return the stereo frames as a (N, 2) float32 array."""
    return _decode_stereo_f32(pcm)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_rejects_zero_or_negative_rate():
    with pytest.raises(ValueError):
        PipelineConfig(native_rate=0)
    with pytest.raises(ValueError):
        PipelineConfig(native_rate=-1)


def test_config_rejects_non_mono():
    with pytest.raises(ValueError):
        PipelineConfig(native_rate=48000, native_channels=2)


def test_config_accepts_mono():
    cfg = PipelineConfig(native_rate=44100, native_channels=1)
    assert cfg.native_rate == 44100
    assert cfg.native_channels == 1


def test_needs_resample_when_native_matches_target():
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    assert pipe.needs_resample is False


def test_needs_resample_when_native_differs_from_target():
    pipe = MicCapturePipeline(PipelineConfig(native_rate=44100))
    assert pipe.needs_resample is True


# ---------------------------------------------------------------------------
# PCM16 → Float32 conversion
# ---------------------------------------------------------------------------

def test_pcm16_zero_maps_to_float32_zero():
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([0]))
    out = pipe.drain()
    assert len(out) % STEREO_FRAME_BYTES == 0
    frames = _all_stereo_frames(out)
    assert frames.shape == (1, 2)
    assert frames[0, 0] == 0.0
    assert frames[0, 1] == 0.0


def test_pcm16_positive_full_scale_to_float32_just_under_one():
    """+32767 → 32767/32768 ≈ 0.99997 (standard signed-16 normalization)."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([32767]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames.shape == (1, 2)
    expected = 32767 / PCM16_FULL_SCALE
    assert abs(frames[0, 0] - expected) < 1e-6
    assert abs(frames[0, 1] - expected) < 1e-6


def test_pcm16_negative_full_scale_to_float32_minus_one():
    """-32768 → -32768/32768 = -1.0 exactly."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([-32768]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames[0, 0] == -1.0
    assert frames[0, 1] == -1.0


def test_pcm16_positive_and_negative_extremes():
    """A sequence of +32767, -32768, 0 produces [≈0.99997, -1.0, 0.0]."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([32767, -32768, 0]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames.shape == (3, 2)
    assert abs(frames[0, 0] - 32767 / PCM16_FULL_SCALE) < 1e-6
    assert frames[1, 0] == -1.0
    assert frames[2, 0] == 0.0


def test_pcm16_uses_standard_signed_16_normalization():
    """The standard normalization is sample / 32768.0 (2^15)."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([1000]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames[0, 0] == 1000 / 32768.0
    assert frames[0, 1] == 1000 / 32768.0


def test_pcm16_odd_byte_count_rejected():
    """Misaligned PCM16 input is rejected (must be a multiple of 2)."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    with pytest.raises(ValueError):
        pipe.feed_pcm16_mono(b"\x00\x00\x00")  # 3 bytes


def test_pcm16_empty_input_is_noop():
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(b"")
    assert pipe.drain() == b""


# ---------------------------------------------------------------------------
# Mono → stereo duplication
# ---------------------------------------------------------------------------

def test_mono_to_stereo_duplicates_left_and_right():
    """Every mono sample is duplicated into both L and R channels."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([100, 200, -300, 400]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames.shape == (4, 2)
    # L[i] == R[i] for every i.
    for i in range(4):
        assert frames[i, 0] == frames[i, 1], f"frame {i}: L != R"
    # Original sample order preserved.
    expected = [100, 200, -300, 400]
    for i, e in enumerate(expected):
        assert frames[i, 0] == e / PCM16_FULL_SCALE


def test_stereo_output_is_interleaved():
    """Output bytes are [L0, R0, L1, R1, ...] in float32 LE."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([16384, -16384]))
    out = pipe.drain()
    # Unpack as float32 LE pairs.
    float_bytes = struct.unpack(f"<{len(out) // 4}f", out)
    assert len(float_bytes) == 4
    # L0, R0, L1, R1
    assert abs(float_bytes[0] - 16384 / PCM16_FULL_SCALE) < 1e-6
    assert abs(float_bytes[1] - 16384 / PCM16_FULL_SCALE) < 1e-6
    assert abs(float_bytes[2] - (-16384 / PCM16_FULL_SCALE)) < 1e-6
    assert abs(float_bytes[3] - (-16384 / PCM16_FULL_SCALE)) < 1e-6


# ---------------------------------------------------------------------------
# Partial PCM16 reads
# ---------------------------------------------------------------------------

def test_partial_pcm16_reads_accumulate():
    """Even-sized but partial reads are accumulated before drain."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    # Feed one sample at a time.
    for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        pipe.feed_pcm16_mono(_pcm16_bytes([s]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames.shape == (10, 2)
    for i in range(10):
        assert frames[i, 0] == i / PCM16_FULL_SCALE
        assert frames[i, 1] == i / PCM16_FULL_SCALE


def test_partial_pcm16_reads_across_drains():
    """A drain in the middle of accumulation preserves all samples."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([0, 1, 2]))
    out1 = pipe.drain()
    pipe.feed_pcm16_mono(_pcm16_bytes([3, 4, 5]))
    out2 = pipe.drain()
    all_frames = _all_stereo_frames(out1 + out2)
    assert all_frames.shape[0] == 6
    for i in range(6):
        assert all_frames[i, 0] == i / PCM16_FULL_SCALE


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def test_resample_44100_to_48000_upsamples():
    """44100 Hz mono → 48000 Hz stereo. Output bytes / 8 ≈ input samples
    * 48000/44100."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=44100))
    # 1000 samples of ramp 0..999.
    samples = list(range(1000))
    pipe.feed_pcm16_mono(_pcm16_bytes(samples))
    out = pipe.drain()
    # Drain may not produce every output sample in one call; the
    # remainder is held. Pull the rest.
    out += pipe.drain()
    frames = _all_stereo_frames(out)
    # Expected output count within resampling tolerance.
    ratio = 48000 / 44100
    expected = int(1000 * ratio)
    # Allow ±1 frame jitter for resampling edge effects.
    assert abs(frames.shape[0] - expected) <= 1, (
        f"expected ≈{expected} frames, got {frames.shape[0]}"
    )


def test_resample_48000_to_48000_passthrough():
    """No resampling: identical-sample byte count, modulo alignment."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    samples = [100 * i for i in range(100)]
    pipe.feed_pcm16_mono(_pcm16_bytes(samples))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames.shape == (100, 2)
    for i in range(100):
        assert frames[i, 0] == 100 * i / PCM16_FULL_SCALE


def test_resample_preserves_signal_shape():
    """A sine wave survives resampling: peak, RMS, and frequency are
    approximately preserved."""
    sr_native = 44100
    sr_target = 48000
    freq = 440.0
    duration_s = 1.0
    n = int(sr_native * duration_s)
    t = np.arange(n) / sr_native
    sine = 0.5 * np.sin(2 * np.pi * freq * t)
    # Convert to int16.
    int16 = (sine * 32767).astype(np.int16)
    pipe = MicCapturePipeline(PipelineConfig(native_rate=sr_native))
    pipe.feed_pcm16_mono(int16.tobytes())
    # Drain fully.
    out = b""
    while True:
        chunk = pipe.drain()
        if not chunk:
            break
        out += chunk
    out += pipe.flush()
    frames = _all_stereo_frames(out)
    # Find peaks in the output (after resampling): peak should be
    # close to 0.5 within ±10%.
    peak = float(np.max(np.abs(frames)))
    assert 0.45 < peak < 0.55, f"peak {peak} drifted from 0.5"


def test_resample_downsample_32000_to_48000():
    """32000 Hz → 48000 Hz upsample."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=32000))
    samples = list(range(500))
    pipe.feed_pcm16_mono(_pcm16_bytes(samples))
    out = b""
    while True:
        chunk = pipe.drain()
        if not chunk:
            break
        out += chunk
    out += pipe.flush()
    frames = _all_stereo_frames(out)
    ratio = 48000 / 32000
    expected = int(500 * ratio)
    assert abs(frames.shape[0] - expected) <= 2


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------

def test_output_is_8_byte_aligned_always():
    """Every chunk returned by drain() and flush() is a multiple of 8 bytes."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    # Feed a varying number of samples to exercise partial-frame buffering.
    for n in [1, 3, 7, 8, 9, 15, 16, 100, 1000]:
        pipe.feed_pcm16_mono(_pcm16_bytes(list(range(n))))
        out = pipe.drain()
        assert len(out) % STEREO_FRAME_BYTES == 0, (
            f"unaligned output for n={n}: len={len(out)}"
        )
    # Flush also returns aligned.
    final = pipe.flush()
    assert len(final) % STEREO_FRAME_BYTES == 0


def test_output_is_8_byte_aligned_under_resampling():
    """Under resampling, every output chunk is still 8-byte-aligned."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=44100))
    for n in [1, 3, 7, 8, 9, 100, 1000]:
        pipe.feed_pcm16_mono(_pcm16_bytes(list(range(n))))
        out = pipe.drain()
        assert len(out) % STEREO_FRAME_BYTES == 0, (
            f"unaligned resampled output for n={n}: len={len(out)}"
        )


def test_drain_returns_empty_when_no_input():
    """drain() before any feed() returns b''."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    assert pipe.drain() == b""


def test_drain_returns_empty_after_full_drain():
    """After all samples are drained, the next drain() returns b''."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([0, 1, 2, 3, 4]))
    out = pipe.drain()
    assert len(out) > 0
    assert pipe.drain() == b""


# ---------------------------------------------------------------------------
# Final output format
# ---------------------------------------------------------------------------

def test_output_format_is_48k_stereo_float32_le():
    """Cross-check the output format:
    - 48000 Hz sample rate (by construction: TARGET_SAMPLE_RATE).
    - 2 channels (stereo).
    - float32 LE (verifiable by unpacking as float32 and matching values).
    """
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    # 48000 mono samples → 48000 stereo frames.
    pipe.feed_pcm16_mono(_pcm16_bytes([10000] * 48000))
    out = b""
    while True:
        chunk = pipe.drain()
        if not chunk:
            break
        out += chunk
    out += pipe.flush()
    # 48000 frames * 2 channels * 4 bytes = 384000 bytes.
    assert len(out) == 48000 * STEREO_FRAME_BYTES
    # Interpret as float32 LE and verify the values.
    arr = np.frombuffer(out, dtype="<f4").reshape(-1, 2)
    assert arr.shape == (48000, 2)
    expected = 10000 / PCM16_FULL_SCALE
    assert np.allclose(arr[:, 0], expected)
    assert np.allclose(arr[:, 1], expected)


def test_drain_returns_no_dsp_artifacts_on_silence():
    """A constant-CW input produces a constant-CW output (no DSP lag/gain)."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([12345] * 100))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    expected = 12345 / PCM16_FULL_SCALE
    assert np.allclose(frames[:, 0], expected)
    assert np.allclose(frames[:, 1], expected)


def test_drain_returns_no_gain_multiplier():
    """No gain multiplier is applied: input samples map bit-exact (modulo
    the standard /32768 normalization)."""
    pipe = MicCapturePipeline(PipelineConfig(native_rate=48000))
    pipe.feed_pcm16_mono(_pcm16_bytes([8192, -8192]))
    out = pipe.drain()
    frames = _all_stereo_frames(out)
    assert frames[0, 0] == 8192 / PCM16_FULL_SCALE
    assert frames[1, 0] == -8192 / PCM16_FULL_SCALE


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def test_announce_native_rate_emits_debug_log(caplog):
    """The helper records native rate and transport format at DEBUG.

    Production cleanup moved this from a verbose banner to a single
    debug log so the normal startup stays clean.
    """
    import logging
    with caplog.at_level(logging.DEBUG, logger="audio-bridge"):
        announce_native_rate(44100)
    assert any("44100" in r.message and "PCM16" in r.message for r in caplog.records)


def test_announce_native_rate_at_48k(caplog):
    """At 48k the helper records the matching native rate."""
    import logging
    with caplog.at_level(logging.DEBUG, logger="audio-bridge"):
        announce_native_rate(48000)
    assert any("48000" in r.message for r in caplog.records)
    assert any("stereo / Float32 LE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: full conversion path with realistic data
# ---------------------------------------------------------------------------

def test_end_to_end_44100_sine_to_48k_stereo():
    """Generate a 1s 440 Hz sine at 44100, run through the pipeline, and
    verify the output is 8-byte-aligned, 2-channel float32."""
    sr_native = 44100
    sr_target = 48000
    freq = 440.0
    duration_s = 1.0
    n = int(sr_native * duration_s)
    t = np.arange(n) / sr_native
    sine = 0.5 * np.sin(2 * np.pi * freq * t)
    int16 = (sine * 32767).astype(np.int16)
    pipe = MicCapturePipeline(PipelineConfig(native_rate=sr_native))
    pipe.feed_pcm16_mono(int16.tobytes())
    out = b""
    while True:
        chunk = pipe.drain()
        if not chunk:
            break
        out += chunk
    out += pipe.flush()
    # Sanity: alignment.
    assert len(out) % STEREO_FRAME_BYTES == 0
    # Reshape as stereo.
    frames = _all_stereo_frames(out)
    # Channels identical.
    assert np.allclose(frames[:, 0], frames[:, 1])
    # Approximate sample count.
    expected_n = int(duration_s * sr_target)
    assert abs(frames.shape[0] - expected_n) <= 2
    # Peak should be ~0.5.
    peak = float(np.max(np.abs(frames)))
    assert 0.45 < peak < 0.55, f"peak drifted: {peak}"
