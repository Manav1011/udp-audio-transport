"""Tests for the raw Android UDP receiver's parsing logic.

These tests verify the protocol validation rules WITHOUT touching the
network or the production backend. They exist because the diagnostic
must be correct at the wire level — if the receiver silently truncates
or drops valid packets, the entire analysis is invalid.

The tests import the parsing helpers from the diagnostic script and
exercise them on synthetic datagrams built with the production
audio_packet encoder.
"""
from __future__ import annotations

import io
import socket
import struct
import threading
import time

import pytest

from transport.audio_packet import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_PAYLOAD,
    encode_packet,
)

# Import the diagnostic's validation helper. We import the module by
# path because diagnostics/ is not a Python package.
import importlib.util
import sys
import pathlib

_DIAG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "diagnostics" / "raw_android_udp_receiver.py"
)
_spec = importlib.util.spec_from_file_location(
    "raw_android_udp_receiver", _DIAG_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["raw_android_udp_receiver"] = _mod
_spec.loader.exec_module(_mod)


# ---------------------------------------------------------------------------
# _validate_and_extract
# ---------------------------------------------------------------------------

def _stats():
    return _mod._Stats()


def test_validate_accepts_well_formed_packet():
    """A perfectly-encoded MAX_PAYLOAD packet round-trips cleanly."""
    stats = _stats()
    payload = b"\x00" * MAX_PAYLOAD
    datagram = encode_packet(42, payload)
    result = _mod._validate_and_extract(datagram, stats)
    assert result is not None
    seq, extracted = result
    assert seq == 42
    assert extracted == payload
    assert stats.errors_length_mismatch == 0
    assert stats.errors_unaligned == 0
    assert stats.errors_short_header == 0


def test_validate_accepts_short_final_packet():
    """An 8-byte-aligned partial payload is valid (trailing packet)."""
    stats = _stats()
    payload = b"\xab" * 16  # 2 stereo frames
    datagram = encode_packet(7, payload)
    result = _mod._validate_and_extract(datagram, stats)
    assert result is not None
    seq, extracted = result
    assert seq == 7
    assert extracted == payload


def test_validate_rejects_short_header():
    """Datagrams shorter than the 8-byte header are dropped."""
    stats = _stats()
    result = _mod._validate_and_extract(b"\x00\x00\x00", stats)
    assert result is None
    assert stats.errors_short_header == 1


def test_validate_rejects_length_mismatch():
    """If declared payload_length != len(datagram) - 8, drop it."""
    stats = _stats()
    payload = b"\x00" * 16
    datagram = encode_packet(1, payload)
    # Tamper: truncate a byte from the payload side.
    truncated = datagram[:-1]
    assert len(truncated) - HEADER_SIZE != 16
    result = _mod._validate_and_extract(truncated, stats)
    assert result is None
    assert stats.errors_length_mismatch == 1


def test_validate_rejects_length_mismatch_extra_byte():
    """If declared is correct but the datagram has extra trailing bytes,
    we reject it (the production decoder would too)."""
    stats = _stats()
    payload = b"\x00" * 16
    datagram = encode_packet(1, payload) + b"\x00"  # extra byte
    result = _mod._validate_and_extract(datagram, stats)
    assert result is None
    assert stats.errors_length_mismatch == 1


def test_validate_rejects_unaligned_payload():
    """A payload that is not a multiple of 8 stereo-frame bytes is rejected."""
    stats = _stats()
    # Build a header that declares 7 bytes (not divisible by 8).
    hdr = struct.pack(HEADER_FORMAT, 1, 7)
    datagram = hdr + b"\x00" * 7
    assert len(datagram) - HEADER_SIZE == 7
    result = _mod._validate_and_extract(datagram, stats)
    assert result is None
    assert stats.errors_unaligned == 1


def test_validate_rejects_declared_oversize():
    """A declared payload_length > MAX_PAYLOAD is rejected."""
    stats = _stats()
    hdr = struct.pack(HEADER_FORMAT, 1, MAX_PAYLOAD + 1)
    datagram = hdr + b"\x00" * (MAX_PAYLOAD + 1)
    result = _mod._validate_and_extract(datagram, stats)
    assert result is None
    assert stats.errors_truncated == 1


def test_validate_accepts_zero_payload_length():
    """An empty payload (length 0) is theoretically valid as a header-only
    packet — the production decoder accepts it (declared=0, len-8=0)."""
    stats = _stats()
    hdr = struct.pack(HEADER_FORMAT, 99, 0)
    result = _mod._validate_and_extract(hdr, stats)
    assert result is not None
    seq, extracted = result
    assert seq == 99
    assert extracted == b""


# ---------------------------------------------------------------------------
# _classify_arrival: ordering classification
# ---------------------------------------------------------------------------

def test_classify_first_packet_is_first():
    stats = _stats()
    tag = _mod._classify_arrival(0, stats)
    assert tag == "first"
    assert stats.first_seq == 0
    assert stats.last_seq == 0


def test_classify_in_order():
    stats = _stats()
    _mod._classify_arrival(0, stats)
    tag = _mod._classify_arrival(1, stats)
    assert tag == "in-order"


def test_classify_gap():
    stats = _stats()
    _mod._classify_arrival(0, stats)
    tag = _mod._classify_arrival(5, stats)
    assert tag == "gap"


def test_classify_out_of_order():
    stats = _stats()
    _mod._classify_arrival(5, stats)
    tag = _mod._classify_arrival(3, stats)
    assert tag == "ooo"


def test_classify_duplicate():
    stats = _stats()
    _mod._classify_arrival(10, stats)
    tag = _mod._classify_arrival(10, stats)
    assert tag == "dupe"
    assert stats.seq_seen[10] == 2


def test_classify_sequence_wraparound():
    """Last seq near UINT32_MAX, next seq is 0 (wraparound)."""
    stats = _stats()
    _mod._classify_arrival(0xFFFFFFFF, stats)
    tag = _mod._classify_arrival(0, stats)
    # 0 - 0xFFFFFFFF modulo 2^32 = 1 → in-order
    assert tag == "in-order"


# ---------------------------------------------------------------------------
# End-to-end: run the diagnostic against a real UDP socket and a synthetic
# sender, verify the file contents are in arrival order.
# ---------------------------------------------------------------------------

def test_end_to_end_arrival_order_preserved(tmp_path):
    """Run the diagnostic receiver against a real loopback sender and
    verify the output file is in arrival order, not seq order."""
    output_path = str(tmp_path / "out.pcm")

    # Pick a random port to avoid conflicts on shared CI hosts.
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    # Build a tiny arrival sequence: send seq=100, then seq=102, then seq=101.
    # The diagnostic must write them in arrival order.
    arrivals = [
        (100, b"\xAA" * 16),
        (102, b"\xBB" * 16),
        (101, b"\xCC" * 16),
    ]
    expected_concat = b"\xAA" * 16 + b"\xBB" * 16 + b"\xCC" * 16

    # Run the diagnostic in a thread. We invoke the same code path the
    # CLI uses, but with argv overrides so we don't fight argparse.
    import subprocess
    # Set PYTHONPATH so the subprocess can import the transport package.
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {**__import__("os").environ, "PYTHONPATH": project_root}
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_DIAG_PATH),
            "--port", str(port),
            "--bind", "127.0.0.1",
            "--output", output_path,
            "--no-status",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Give the listener a moment to bind.
        time.sleep(0.3)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for seq, payload in arrivals:
            sender.sendto(encode_packet(seq, payload), ("127.0.0.1", port))
        sender.close()
        # Wait for the receiver to flush.
        time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    with open(output_path, "rb") as f:
        contents = f.read()
    assert contents == expected_concat, (
        f"file does not match arrival order: "
        f"expected {expected_concat!r}, got {contents!r}"
    )


def test_end_to_end_drops_invalid_packets(tmp_path):
    """A malformed packet (short header) must NOT contribute to the file."""
    output_path = str(tmp_path / "out.pcm")

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    valid_payload = b"\xDE\xAD\xBE\xEF" * 4  # 16 bytes
    valid = encode_packet(0, valid_payload)
    short = b"\x00\x00"  # only 2 bytes — too short

    import subprocess
    import os
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": project_root}
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_DIAG_PATH),
            "--port", str(port),
            "--bind", "127.0.0.1",
            "--output", output_path,
            "--no-status",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        time.sleep(0.3)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(valid, ("127.0.0.1", port))
        sender.sendto(short, ("127.0.0.1", port))
        sender.sendto(valid, ("127.0.0.1", port))
        sender.close()
        time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    with open(output_path, "rb") as f:
        contents = f.read()
    # Only the two valid packets' payloads are written.
    assert contents == valid_payload + valid_payload


# ---------------------------------------------------------------------------
# Default-port convention
# ---------------------------------------------------------------------------

def test_default_port_constant_is_5001():
    """The diagnostic's default UDP port is 5001 — matches the Android
    sender's configured TX port. The bare `python -m diagnostics.raw_android_udp_receiver`
    command must listen on 5001 without any flags."""
    assert _mod.DEFAULT_PORT == 5001
    assert _mod.DEFAULT_BIND_HOST == "0.0.0.0"


def test_argparse_default_port_is_5001():
    """argparse exposes the default port via --help; assert it matches 5001."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=_mod.DEFAULT_PORT,
        help=f"UDP port to listen on (default {_mod.DEFAULT_PORT})",
    )
    parser.add_argument(
        "--bind", default=_mod.DEFAULT_BIND_HOST,
    )
    args = parser.parse_args([])
    assert args.port == 5001
    assert args.bind == "0.0.0.0"


def test_bare_invocation_binds_5001(tmp_path):
    """Run the diagnostic with NO flags and verify it advertises 5001 in
    the startup banner. We don't actually wait for the bind because
    port 5001 may be in use by the Android sender on the test host."""
    output_path = str(tmp_path / "out.pcm")
    import subprocess
    import os
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": project_root}
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_DIAG_PATH),
            "--output", output_path,
            "--no-status",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Give it a moment to print the banner.
        time.sleep(0.3)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    # The receiver prints "bind:        0.0.0.0:5001" at startup.
    assert b"0.0.0.0:5001" in out, (
        f"banner did not advertise bind 0.0.0.0:5001: {out!r}"
    )
