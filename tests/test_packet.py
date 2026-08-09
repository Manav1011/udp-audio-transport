"""Unit tests for transport/audio_packet.py."""
import struct

import pytest

from transport.audio_packet import (
    HEADER_SIZE,
    MAX_DATAGRAM,
    MAX_PAYLOAD,
    decode_packet,
    encode_packet,
)


def test_header_size_constant():
    assert HEADER_SIZE == 8


def test_max_payload_multiple_of_stereo_frame():
    # 1152 bytes / 4 bytes-per-sample / 2 channels = 144 frames
    assert MAX_PAYLOAD % 8 == 0


def test_max_datagram_under_1200():
    assert MAX_DATAGRAM <= 1200


def test_roundtrip_minimal():
    pkt = encode_packet(0, b"")
    assert decode_packet(pkt) == (0, b"")


def test_roundtrip_typical():
    payload = b"\x01\x02\x03\x04" * 100
    pkt = encode_packet(42, payload)
    assert len(pkt) == HEADER_SIZE + len(payload)
    assert decode_packet(pkt) == (42, payload)


def test_roundtrip_max_payload():
    # Build exactly MAX_PAYLOAD bytes without truncation artifacts.
    payload = bytes(range(256)) * (MAX_PAYLOAD // 256 + 1)
    payload = payload[:MAX_PAYLOAD]
    assert len(payload) == MAX_PAYLOAD
    pkt = encode_packet(2**32 - 1, payload)
    assert len(pkt) == MAX_DATAGRAM
    assert decode_packet(pkt) == (2**32 - 1, payload)


def test_encode_rejects_oversized_payload():
    with pytest.raises(ValueError):
        encode_packet(0, b"x" * (MAX_PAYLOAD + 1))


def test_decode_rejects_short_header():
    assert decode_packet(b"\x00\x00\x00") is None


def test_decode_rejects_oversized_declared_length():
    # Header says 2000 bytes but datagram is only header + 0
    bad = struct.pack("!II", 0, MAX_PAYLOAD + 1)
    assert decode_packet(bad) is None


def test_decode_rejects_length_mismatch():
    # Header says 100 bytes but payload is only 50
    header = struct.pack("!II", 0, 100)
    payload = b"\x00" * 50
    assert decode_packet(header + payload) is None


def test_big_endian_header():
    # seq=1, payload_length=1 (single byte)
    pkt = encode_packet(1, b"\xff")
    assert pkt[:4] == b"\x00\x00\x00\x01"
    assert pkt[4:8] == b"\x00\x00\x00\x01"
    assert pkt[8:] == b"\xff"


def test_sequence_number_at_max_uint32():
    pkt = encode_packet(0xFFFFFFFF, b"x")
    assert pkt[:4] == b"\xff\xff\xff\xff"
    assert decode_packet(pkt) == (0xFFFFFFFF, b"x")
