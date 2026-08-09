"""Audio packet format — compact binary header + raw PCM payload.

Wire format (big-endian, network byte order):

    uint32 sequence_number
    uint32 payload_length
    byte[] payload (raw PCM bytes)

Header:       8 bytes
Max payload: 1152 bytes  (= 288 stereo float32 frames = 6 ms @ 48 kHz)
Max datagram: 1160 bytes (header + payload, comfortably under 1200)
"""
from __future__ import annotations

import struct

HEADER_FORMAT = "!II"  # sequence_number, payload_length
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 8

MAX_PAYLOAD = 1152
MAX_DATAGRAM = HEADER_SIZE + MAX_PAYLOAD  # 1160


def encode_packet(sequence_number: int, payload: bytes) -> bytes:
    """Pack a packet: 8-byte header + raw payload."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(
            f"payload too large: {len(payload)} > {MAX_PAYLOAD}"
        )
    return struct.pack(HEADER_FORMAT, sequence_number & 0xFFFFFFFF, len(payload)) + payload


def decode_packet(data: bytes) -> tuple[int, bytes] | None:
    """Decode a packet. Returns (sequence_number, payload) or None on invalid input.

    Rejects:
      - datagrams shorter than the header
      - payload_length field exceeding MAX_PAYLOAD
      - datagrams whose actual length doesn't match the declared payload
    """
    if len(data) < HEADER_SIZE:
        return None
    seq, declared = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    if declared > MAX_PAYLOAD:
        return None
    if len(data) - HEADER_SIZE != declared:
        return None
    return seq, data[HEADER_SIZE:]
