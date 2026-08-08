"""High-level protocol behaviour: how the server responds to packets."""
from __future__ import annotations

from protocol.packet import Packet, PacketType


def handle(packet: Packet) -> Packet | None:
    """Return a reply packet for the given input, or None (no reply)."""
    match packet.type:
        case PacketType.HELLO:
            return Packet(PacketType.HELLO_ACK, payload={"status": "ok"})
        case PacketType.PING:
            return Packet(PacketType.PONG, payload={"status": "ok"})
        case PacketType.MESSAGE:
            # LOG is handled by the server loop so the IP/port is visible.
            return None
