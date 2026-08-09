"""Integration test: deterministic PCM over UDP loopback.

Sends 2 seconds of a deterministic sine wave over UDP on localhost and
asserts the receiver reconstructs the PCM byte-for-byte.

Runs on ephemeral ports so it never collides with anything.
"""
import math
import socket
import struct
import threading
import time

import numpy as np

from transport.audio_packet import decode_packet
from transport.audio_sender import AudioSender
from transport.audio_receiver import AudioReceiver


def _bind_ephemeral() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    return s


def _make_deterministic_pcm(duration_s: float = 2.0,
                            sample_rate: int = 48000,
                            channels: int = 2,
                            freq: float = 440.0) -> bytes:
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    # Use deterministic sin — no rng, no averaging.
    wave = 0.5 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    return stereo.tobytes()


def test_pcm_round_trip_byte_identical():
    recv_sock = _bind_ephemeral()
    recv_sock.settimeout(0.5)

    # The receiver expects a port to bind to. We pass the socket we already
    # bound so the test controls the port.
    received_chunks: list[bytes] = []
    received_lock = threading.Lock()
    done_event = threading.Event()

    receiver = AudioReceiver(
        on_pcm=lambda b: (received_chunks.append(b), None)[1]
                        if not done_event.is_set() else None,
        sock=recv_sock,
    )

    # Sender uses a UDP socket pointing at the receiver's bound port.
    recv_port = recv_sock.getsockname()[1]
    sender = AudioSender(dest=("127.0.0.1", recv_port))

    receiver.start()
    sender.start()

    try:
        pcm = _make_deterministic_pcm(duration_s=2.0)
        assert len(pcm) == 2 * 48000 * 2 * 4  # 768000 bytes

        # Send in 100ms chunks (matches capture cadence).
        chunk_size = 48000 // 10 * 2 * 4  # 38400 bytes
        for i in range(0, len(pcm), chunk_size):
            sender.submit(pcm[i:i + chunk_size])
            time.sleep(0.05)

        # Wait for the receiver to gather all data.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with received_lock:
                # Allow ongoing arrival; stop when 95%+ received.
                if sum(len(c) for c in received_chunks) >= len(pcm) * 0.95:
                    break
            time.sleep(0.05)
    finally:
        done_event.set()
        sender.stop()
        receiver.stop()

    joined = b"".join(received_chunks)
    assert len(joined) == len(pcm), f"got {len(joined)} bytes, expected {len(pcm)}"
    assert joined == pcm, "PCM bytes did not match — corruption in transit"


def test_pcm_round_trip_short_payload():
    """Smaller payload (less than one MAX_PAYLOAD) to test the partial-packet path."""
    recv_sock = _bind_ephemeral()
    received: list[bytes] = []
    done = threading.Event()

    receiver = AudioReceiver(
        on_pcm=lambda b: received.append(b) if not done.is_set() else None,
        sock=recv_sock,
    )
    sender = AudioSender(dest=("127.0.0.1", recv_sock.getsockname()[1]))

    receiver.start(); sender.start()
    try:
        # 500 bytes — well under MAX_PAYLOAD (1152)
        pcm = bytes(range(256)) + bytes(range(244))
        assert len(pcm) == 500
        sender.submit(pcm)
        time.sleep(0.5)
    finally:
        done.set()
        sender.stop()
        receiver.stop()

    assert b"".join(received) == pcm
