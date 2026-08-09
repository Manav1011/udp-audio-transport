"""Tests for the raw Android TCP receiver.

The single most important property is byte fidelity:

    receiver writes ONLY bytes received from the socket.
    no transformation, no reordering, no header, no validation.

These tests verify that property using a local TCP loopback: a
synthetic sender pushes arbitrary bytes (including zero bytes,
binary bytes, and Float32-looking samples) into the receiver, and
the resulting file is compared byte-for-byte to the input.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

_DIAG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "diagnostics" / "raw_android_tcp_receiver.py"
)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Get a free TCP port by letting the OS pick one."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_receiver_subprocess(
    output_path: str, port: int
) -> subprocess.Popen:
    """Launch the diagnostic receiver as a subprocess so we can feed it
    real TCP traffic on a real loopback socket."""
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": project_root}
    return subprocess.Popen(
        [
            sys.executable,
            str(_DIAG_PATH),
            "--bind", "127.0.0.1",
            "--port", str(port),
            "--output", output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _send_all(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(payload)


def _run_end_to_end(
    payload_chunks: list[bytes],
) -> tuple[bytes, dict, subprocess.Popen]:
    """Run the receiver end-to-end against a list of send chunks and
    return the file contents, parsed stats from the stdout, and the
    subprocess handle."""
    output_path = (
        pathlib.Path(__file__).resolve().parent
        / "_tmp_tcp_out.pcm"
    )
    if output_path.exists():
        output_path.unlink()
    port = _free_port()
    proc = _start_receiver_subprocess(str(output_path), port)
    # Wait for the receiver to start listening.
    for _ in range(50):
        time.sleep(0.1)
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sender.connect(("127.0.0.1", port))
            break
        except (ConnectionRefusedError, OSError):
            sender.close()
    else:
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("receiver did not start listening")
    try:
        for chunk in payload_chunks:
            _send_all(sender, chunk)
        sender.shutdown(socket.SHUT_WR)
    finally:
        sender.close()
    # Wait for the receiver to finish.
    try:
        out, _err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    file_bytes = output_path.read_bytes()
    stats = _parse_stats(out)
    return file_bytes, stats, proc


def _parse_stats(stdout: bytes) -> dict:
    """Pull the printed stats from the receiver's stdout."""
    text = stdout.decode(errors="replace")
    out = {}
    for line in text.splitlines():
        if "bytes received:" in line:
            out["bytes_received"] = int(line.split(":")[-1].strip().replace(",", ""))
        elif "recv calls:" in line:
            out["recv_calls"] = int(line.split(":")[-1].strip().replace(",", ""))
        elif "SHA256:" in line:
            out["sha256"] = line.split("SHA256:")[-1].strip()
        elif "bytes % 8:" in line:
            out["bytes_mod_8"] = int(line.split(":")[-1].strip().split()[0])
    return out


# ---------------------------------------------------------------------------
# Byte-fidelity tests
# ---------------------------------------------------------------------------

def test_receiver_writes_exact_bytes_simple():
    """Three chunks concatenated = file contents, byte-for-byte."""
    chunks = [b"hello", b" ", b"world"]
    expected = b"".join(chunks)
    file_bytes, stats, _ = _run_end_to_end(chunks)
    assert file_bytes == expected
    assert stats["bytes_received"] == len(expected)
    assert stats["sha256"] == hashlib.sha256(expected).hexdigest()


def test_receiver_writes_exact_bytes_with_zero_bytes():
    """Zero bytes are preserved (binary mode, not text)."""
    chunks = [b"before\x00\x00\x00", b"\x00", b"after", b"\x00\x00"]
    expected = b"".join(chunks)
    file_bytes, stats, _ = _run_end_to_end(chunks)
    assert file_bytes == expected
    assert hashlib.sha256(file_bytes).hexdigest() == stats["sha256"]


def test_receiver_writes_exact_bytes_arbitrary_binary():
    """All 256 byte values appear in the output when sent."""
    payload = bytes(range(256)) * 4  # 1024 bytes
    file_bytes, stats, _ = _run_end_to_end([payload])
    assert file_bytes == payload
    assert hashlib.sha256(file_bytes).hexdigest() == stats["sha256"]


def test_receiver_writes_exact_bytes_float32le_samples():
    """Real Float32 LE samples (including negative) pass through verbatim."""
    sr = 48000
    samples = np.linspace(-1.0, 1.0, sr * 2, dtype=np.float32)
    payload = samples.tobytes()
    expected_hash = hashlib.sha256(payload).hexdigest()
    file_bytes, stats, _ = _run_end_to_end([payload])
    assert file_bytes == payload
    assert stats["sha256"] == expected_hash
    # Verify the bytes at the receiver are still a valid Float32 LE array.
    recovered = np.frombuffer(file_bytes, dtype="<f4")
    assert recovered.shape == samples.shape
    assert np.allclose(recovered, samples)


def test_receiver_writes_exact_bytes_split_into_many_small_chunks():
    """Many small TCP sends are concatenated correctly."""
    rng = np.random.default_rng(42)
    chunks = [rng.integers(0, 256, size=37, dtype=np.uint8).tobytes() for _ in range(50)]
    expected = b"".join(chunks)
    file_bytes, stats, _ = _run_end_to_end(chunks)
    assert file_bytes == expected
    assert stats["bytes_received"] == len(expected)


def test_receiver_writes_exact_bytes_audio_looking_payload():
    """A multi-second payload of realistic 48k stereo Float32 frames."""
    sr = 48000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    payload = stereo.tobytes()
    assert len(payload) % 8 == 0
    file_bytes, stats, _ = _run_end_to_end([payload])
    assert file_bytes == payload
    assert stats["bytes_received"] == len(payload)
    assert stats["bytes_mod_8"] == 0


def test_receiver_preserves_non_aligned_bytes():
    """A payload whose length is NOT a multiple of 8 is still passed
    through verbatim. The receiver never enforces alignment."""
    payload = b"".join(bytes([i % 256]) for i in range(7919))  # 7919 % 8 = 7
    file_bytes, stats, _ = _run_end_to_end([payload])
    assert file_bytes == payload
    assert stats["bytes_received"] == 7919
    assert stats["bytes_mod_8"] == 7  # 7919 % 8


def test_receiver_sha256_matches_overall_hash():
    """The reported SHA256 is the hash of the entire file content."""
    payload = bytes(range(256)) * 16  # 4096 bytes
    file_bytes, stats, _ = _run_end_to_end([payload])
    assert hashlib.sha256(file_bytes).hexdigest() == stats["sha256"]


def test_receiver_recv_calls_counter_is_at_least_one():
    """The recv-calls counter is non-zero after a successful transfer."""
    payload = b"abcdefghij" * 1000  # 10000 bytes
    _file_bytes, stats, _ = _run_end_to_end([payload])
    assert stats["recv_calls"] >= 1


def test_receiver_default_port_constant_is_5002():
    """The diagnostic's default TCP port is 5002 per the spec."""
    spec = importlib.util.spec_from_file_location(
        "_raw_tcp_receiver_under_test", _DIAG_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_PORT == 5002
    assert mod.DEFAULT_BIND == "0.0.0.0"
    assert mod.DEFAULT_OUTPUT == "/tmp/android_raw_tcp.pcm"


def test_receiver_help_text_default_port_is_5002():
    """argparse --help must show port 5002 as the default."""
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": project_root}
    proc = subprocess.run(
        [sys.executable, str(_DIAG_PATH), "--help"],
        capture_output=True, text=True, env=env, timeout=5,
    )
    assert "default 5002" in proc.stdout, proc.stdout


def test_receiver_prints_three_section_banner():
    """The shutdown banner contains the three required statistical lines."""
    payload = b"x" * 256
    _file_bytes, stats, _ = _run_end_to_end([payload])
    # Stats dict must contain the three required keys.
    for key in ("bytes_received", "recv_calls", "sha256"):
        assert key in stats, f"missing {key} in stats: {stats}"
    assert stats["bytes_received"] == 256
    assert stats["sha256"] == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Unit tests for the `_serve_one_connection` helper, in-process (no subprocess)
# ---------------------------------------------------------------------------

def _serve_one_connection_helper_smoke():
    """Drive _serve_one_connection directly via a local socket pair to
    verify the helper's return tuple shape."""
    spec = importlib.util.spec_from_file_location(
        "_raw_tcp_receiver_helper", _DIAG_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    output_path = (
        pathlib.Path(__file__).resolve().parent / "_tmp_helper_out.pcm"
    )
    if output_path.exists():
        output_path.unlink()

    accepted: list = []

    def accept_in_thread():
        accepted.append(mod._serve_one_connection(
            server, str(output_path), threading.Lock()
        ))

    t = threading.Thread(target=accept_in_thread, daemon=True)
    t.start()
    # Connection.
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    payload = b"hello world!"
    client.sendall(payload)
    client.shutdown(socket.SHUT_WR)
    client.close()
    t.join(timeout=5)
    server.close()
    bytes_received, recv_calls, sha256_hex, duration = accepted[0]
    assert bytes_received == len(payload)
    assert recv_calls >= 1
    assert sha256_hex == hashlib.sha256(payload).hexdigest()
    assert duration >= 0
    assert output_path.read_bytes() == payload
    output_path.unlink()


def test_serve_one_connection_helper():
    """Wraps the helper so pytest picks it up properly."""
    _serve_one_connection_helper_smoke()
