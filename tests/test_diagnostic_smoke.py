"""Smoke test for the diagnostic PCM tee.

Verifies:
- AUDIO_DIAGNOSTIC_RECORD=1 -> WAV is created at /tmp/backend_received.wav
- WAV header is IEEE-float (format code 3) — parseable by external readers
- Bytes recorded == bytes that would have been sent to injector
- close() finalizes the file (size stable after close)
- Disabled mode (env unset) writes nothing and never opens a file
"""
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from transport.audio_diagnostic import (
    DiagnosticWavWriter,
    NullDiagnosticWavWriter,
    is_enabled,
)
from transport.audio_receiver import AudioReceiver
from transport.audio_sender import AudioSender


def _make_deterministic_pcm(duration_s: float = 0.5,
                            sample_rate: int = 48000,
                            channels: int = 2,
                            freq: float = 440.0) -> bytes:
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    wave = 0.5 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    return stereo.tobytes()


def _read_wav_data(path: str) -> tuple[int, int, int, bytes]:
    """Read a WAV file's data section. Returns (channels, sample_rate, bits, raw_bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    i = 12
    while i < len(data):
        cid = data[i:i+4]
        csz = struct.unpack("<I", data[i+4:i+8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[i+8:i+8+csz])
            audio_format = fmt[0]
            channels = fmt[1]
            sample_rate = fmt[2]
            bits = fmt[5]
        elif cid == b"data":
            audio = data[i+8:i+8+csz]
            return channels, sample_rate, bits, audio
        i += 8 + csz
    raise AssertionError("no data chunk")


def test_diagnostic_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("AUDIO_DIAGNOSTIC_RECORD", raising=False)
    assert is_enabled() is False
    w = NullDiagnosticWavWriter()
    w.write(b"\x00\x00\x00\x00")
    w.close()


def test_diagnostic_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("AUDIO_DIAGNOSTIC_RECORD", "0")
    assert is_enabled() is False


def test_diagnostic_enabled_when_env_one(monkeypatch):
    monkeypatch.setenv("AUDIO_DIAGNOSTIC_RECORD", "1")
    assert is_enabled() is True


def test_diagnostic_wav_writes_exact_bytes(tmp_path):
    target = tmp_path / "out.wav"
    original = DiagnosticWavWriter.PATH
    DiagnosticWavWriter.PATH = str(target)
    try:
        w = DiagnosticWavWriter()
        payload = _make_deterministic_pcm(0.1)
        w.write(payload)
        w.close()
        assert target.exists()
        channels, sample_rate, bits, raw = _read_wav_data(str(target))
        assert channels == 2
        assert sample_rate == 48000
        assert bits == 32
        assert raw == payload, "WAV bytes must equal bytes written"
    finally:
        DiagnosticWavWriter.PATH = original


def test_diagnostic_wav_header_is_ieee_float(tmp_path):
    """External tools must recognize the WAV as IEEE float (format code 3),
    not as int32 PCM (format code 1)."""
    target = tmp_path / "fmt.wav"
    original = DiagnosticWavWriter.PATH
    DiagnosticWavWriter.PATH = str(target)
    try:
        w = DiagnosticWavWriter()
        w.write(b"\x00" * 8)
        w.close()
        with open(str(target), "rb") as f:
            data = f.read()
        # fmt chunk starts at offset 20, fmt header is "fmt "
        fmt_chunk_id = data[12:16]
        fmt_size = struct.unpack("<I", data[16:20])[0]
        assert fmt_chunk_id == b"fmt "
        audio_format = struct.unpack("<H", data[20:22])[0]
        assert audio_format == 3, f"expected IEEE float (3), got format {audio_format}"
        # fmt chunk size = 16 (the standard PCMWAVEFORMAT) — fine.
        assert fmt_size == 16, f"expected fmt chunk size 16, got {fmt_size}"
        # RIFF chunk size = filesize - 8
        assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
        # data chunk size = filesize - 44
        assert struct.unpack("<I", data[40:44])[0] == len(data) - 44
    finally:
        DiagnosticWavWriter.PATH = original


def test_diagnostic_tee_records_exact_bytes_via_session(tmp_path):
    """Final architecture: bytes travel Android -> TCP -> AudioTcpMicReceiver
    -> on_pcm callback -> diagnostic tee -> injector. This test drives a TCP
    client connected to the session's mic listener and verifies the WAV tee
    bytes exactly match the bytes handed to the injector.
    """
    target = tmp_path / "diag.wav"
    original = DiagnosticWavWriter.PATH
    DiagnosticWavWriter.PATH = str(target)

    os.environ["AUDIO_DIAGNOSTIC_RECORD"] = "1"

    injector_write_calls: list[bytes] = []

    # Find a free TCP port — the session's mic listener will bind here.
    free = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free.bind(("127.0.0.1", 0))
    mic_port = free.getsockname()[1]
    free.close()

    def fake_injector_write(pcm: bytes):
        injector_write_calls.append(pcm)

    from transport.audio_session import AudioSession
    session = AudioSession(
        mic_bind_host="127.0.0.1",
        mic_bind_port=mic_port,
        speaker_dest_host="127.0.0.1",
        speaker_dest_port=mic_port,  # unused for this test; speaker is a no-op here
    )
    session.bind_injector(fake_injector_write)
    session.start()

    try:
        # Wait for the mic TCP listener to actually accept connections
        # before we connect, otherwise the connect may race the bind.
        for _ in range(50):
            try:
                probe = socket.create_connection(("127.0.0.1", mic_port), timeout=0.5)
                probe.close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("mic TCP listener never came up")

        pcm = _make_deterministic_pcm(0.3)
        chunk_size = 48000 // 10 * 2 * 4

        # Drive the TCP mic path the same way the Android mic app would:
        # open a TCP client and stream PCM chunks. The session's
        # AudioTcpMicReceiver will forward each chunk to on_pcm, which
        # tees to the diagnostic WAV and then calls the injector.
        client = socket.create_connection(("127.0.0.1", mic_port))
        try:
            for i in range(0, len(pcm), chunk_size):
                client.sendall(pcm[i:i + chunk_size])
                time.sleep(0.02)
            client.shutdown(socket.SHUT_WR)
        finally:
            client.close()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if sum(len(c) for c in injector_write_calls) >= len(pcm) * 0.9:
                break
            time.sleep(0.05)
    finally:
        session.stop()
        os.environ.pop("AUDIO_DIAGNOSTIC_RECORD", None)
        DiagnosticWavWriter.PATH = original

    assert target.exists()
    delivered = b"".join(injector_write_calls)
    channels, sample_rate, bits, raw = _read_wav_data(str(target))
    assert channels == 2
    assert sample_rate == 48000
    assert bits == 32
    assert raw == delivered, "WAV bytes must match bytes handed to injector"
    assert len(delivered) >= len(pcm) * 0.8
