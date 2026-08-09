"""Diagnostic PCM recorder (TEMPORARY — observation only).

Tees the reconstructed PCM stream that is about to be handed to the
Injector.write_frames() into a WAV file on disk. Used for debugging
audio-quality issues without altering the production audio path.

Enabled only when AUDIO_DIAGNOSTIC_RECORD=1. Disabled completely when
the variable is unset or set to anything other than "1".

The WAV is written as 48 kHz stereo float32 little-endian (matching the
PCM format the Injector expects). Every byte handed to write() is
appended verbatim — no normalization, no resampling, no transformation.

This module is part of a temporary diagnostic patch. It does not
participate in any control flow; the production path proceeds identically
whether the diagnostic is enabled or not.
"""
from __future__ import annotations

import logging
import os
import struct

log = logging.getLogger("audio-bridge")


def is_enabled() -> bool:
    """True only when the env var is set to exactly "1"."""
    return os.environ.get("AUDIO_DIAGNOSTIC_RECORD") == "1"


def _write_wav_header(path: str, channels: int, sample_rate: int,
                      bits_per_sample: int) -> None:
    """Write a 44-byte IEEE-float WAV header (format code 3).

    Python's wave module produces format code 1 (PCM) for sampwidth=4,
    which is invalid for IEEE-float data. We write the header by hand so
    downstream tools (numpy/scipy/ffmpeg) recognize the file correctly.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,                # audio format = IEEE float
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    # data chunk header: "data" + size (4 bytes) — size will be patched on close
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 0))               # placeholder; RIFF size patched later
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", 0))               # placeholder; data size patched later


def _patch_wav_sizes(path: str) -> None:
    """Patch the RIFF and data chunk sizes in place.

    Python's wave module writes these after the data is closed; we open
    the file in append+read mode and write them ourselves.
    """
    size = os.path.getsize(path)
    data_size = size - 44
    with open(path, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<I", size - 8))  # RIFF chunk size
        f.seek(40)
        f.write(struct.pack("<I", data_size))


class DiagnosticWavWriter:
    """Append-only WAV writer that records raw PCM bytes.

    Opens /tmp/backend_received.wav on construction. Each write(pcm)
    appends the bytes verbatim and updates running counters. close()
    finalizes the WAV header and flushes.

    The writer is intentionally trivial — no resampling, no conversion.
    Whatever PCM bytes are handed to write() are what appear on disk,
    in the order they were received.
    """

    PATH = "/tmp/backend_received.wav"
    SAMPLE_RATE = 48000
    CHANNELS = 2
    SAMPLE_WIDTH = 4  # float32

    def __init__(self) -> None:
        # Truncate any previous file so each session starts fresh.
        if os.path.exists(self.PATH):
            os.remove(self.PATH)
        _write_wav_header(
            self.PATH, self.CHANNELS, self.SAMPLE_RATE, self.SAMPLE_WIDTH * 8
        )
        self._fp = open(self.PATH, "ab")

        self._bytes_written = 0
        self._frames_written = 0

        log.debug("Diagnostic recording started: %s (%d Hz, %d ch, %d-bit IEEE float LE)",
                  self.PATH, self.SAMPLE_RATE, self.CHANNELS, self.SAMPLE_WIDTH * 8)

    def write(self, pcm: bytes) -> None:
        """Append raw PCM bytes. Bytes are written unmodified."""
        if not pcm:
            return
        self._fp.write(pcm)
        # Flush per write so we capture bytes even on a hard crash.
        self._fp.flush()
        self._bytes_written += len(pcm)
        self._frames_written += len(pcm) // (self.CHANNELS * self.SAMPLE_WIDTH)
        log.debug("Diagnostic bytes written: %d (frames=%d)",
                  self._bytes_written, self._frames_written)

    def close(self) -> None:
        """Finalize the WAV header and close the file."""
        if self._fp is None:
            return
        self._fp.close()
        self._fp = None
        _patch_wav_sizes(self.PATH)
        log.debug("Diagnostic recording finalized: %s (%d bytes, %d frames)",
                  self.PATH, self._bytes_written, self._frames_written)


class NullDiagnosticWavWriter:
    """No-op stand-in used when AUDIO_DIAGNOSTIC_RECORD is not set.

    Lets the production code call diagnostic.write(pcm) unconditionally
    without an if-check at every call site.
    """

    def write(self, pcm: bytes) -> None:
        pass

    def close(self) -> None:
        pass
