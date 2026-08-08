"""Packet definitions and serialization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum


class PacketType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    PING = 3
    PONG = 4
    MESSAGE = 5


@dataclass
class Packet:
    type: PacketType
    payload: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


def serialize(packet: Packet) -> bytes:
    """Serialize a packet to JSON bytes."""
    body = {"type": int(packet.type), "payload": packet.payload}
    return json.dumps(body).encode("utf-8")


def deserialize(data: bytes) -> Packet | None:
    """Deserialize bytes to a Packet. Returns None on invalid input."""
    try:
        body = json.loads(data.decode("utf-8"))
        ptype = PacketType(body["type"])
        payload = body.get("payload")
        if not isinstance(payload, dict):
            return None
        return Packet(type=ptype, payload=payload)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
