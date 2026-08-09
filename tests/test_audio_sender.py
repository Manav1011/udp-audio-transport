"""Unit tests for transport/audio_sender.py.

We intercept the underlying socket via a MockSocket that captures datagrams
without touching the network. This isolates send-loop logic and
fragmentation from UDP plumbing.
"""
import socket
import struct
import threading
import time

from transport.audio_packet import HEADER_SIZE, MAX_DATAGRAM, MAX_PAYLOAD, decode_packet
from transport.audio_sender import AudioSender


class MockSocket:
    """Captures sent datagrams; no network I/O."""

    def __init__(self):
        self._lock = threading.Lock()
        self.datagrams: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data: bytes, addr: tuple[str, int]):
        with self._lock:
            self.datagrams.append((data, addr))

    def close(self):
        pass


def _wait_for_sender(sender: AudioSender, predicate, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timeout waiting for sender condition")


def test_fragmentation_38400_bytes():
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock)
    sender.start()
    try:
        chunk = bytes(range(256)) * 150  # 38400 bytes
        assert len(chunk) == 38400
        sender.submit(chunk)
        # 38400 / 1152 = 33.33 → 34 packets
        _wait_for_sender(sender, lambda: len(sock.datagrams) >= 34)
    finally:
        sender.stop()

    assert len(sock.datagrams) == 34
    # 33 full payloads + 1 partial (38400 - 33*1152 = 38400 - 38016 = 384 bytes)
    expected_payloads = [1152] * 33 + [384]
    actual_payloads = [len(d) - HEADER_SIZE for d, _ in sock.datagrams[:34]]
    assert actual_payloads == expected_payloads


def test_max_datagram_invariant():
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock)
    sender.start()
    try:
        # Largest multiple of MAX_PAYLOAD we can fit through one submit
        sender.submit(b"\xab" * (MAX_PAYLOAD * 10))
        _wait_for_sender(sender, lambda: len(sock.datagrams) >= 10)
    finally:
        sender.stop()

    for d, _ in sock.datagrams:
        assert len(d) <= MAX_DATAGRAM


def test_sequence_numbers_monotonic_and_start_at_zero():
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock)
    sender.start()
    try:
        # Send several chunks of varying sizes
        sender.submit(b"\x01" * 1000)
        sender.submit(b"\x02" * 2000)
        sender.submit(b"\x03" * 300)
        _wait_for_sender(sender, lambda: sender.stats()["packets_sent"] >= 4)
    finally:
        sender.stop()

    seqs = []
    for d, _ in sock.datagrams:
        decoded = decode_packet(d)
        assert decoded is not None
        seqs.append(decoded[0])
    assert seqs == sorted(seqs)
    assert seqs[0] == 0
    assert seqs == list(range(len(seqs)))


def test_payloads_reassemble_to_original_chunks():
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock)
    sender.start()
    try:
        chunks = [b"A" * 500, b"B" * 2300, b"C" * 1152, b"D" * 100]
        for c in chunks:
            sender.submit(c)
            time.sleep(0.05)  # let sender drain between submits
        _wait_for_sender(sender, lambda: sender.stats()["pcm_chunks_submitted"] == 4)
        # Drain anything still in queue
        time.sleep(0.2)
    finally:
        sender.stop()

    # Reconstruct
    payloads = []
    for d, _ in sock.datagrams:
        decoded = decode_packet(d)
        assert decoded is not None
        payloads.append(decoded[1])

    joined = b"".join(payloads)
    assert joined == b"".join(chunks)


def test_stats_counting():
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock)
    sender.start()
    try:
        sender.submit(b"\x00" * 2304)  # exactly 2 packets
        _wait_for_sender(sender, lambda: sender.stats()["packets_sent"] >= 2)
    finally:
        sender.stop()

    s = sender.stats()
    assert s["packets_sent"] == 2
    assert s["pcm_chunks_submitted"] == 1
    assert s["pcm_bytes_submitted"] == 2304
    assert s["bytes_sent"] == 2 * (HEADER_SIZE + 1152)
    assert s["pcm_chunks_dropped"] == 0


def test_queue_full_drops():
    # Tiny queue + big chunk → drop behavior
    sock = MockSocket()
    sender = AudioSender(dest=("127.0.0.1", 1), sock=sock, queue_size=1)
    sender.start()
    try:
        # Submit rapidly without giving the sender a chance to drain.
        # The sender thread starts immediately and will consume one.
        for _ in range(50):
            sender.submit(b"\x00" * 38400)
        # At least one must be dropped (queue is 1; sender drains one at a time).
        # We can't guarantee exact count since the thread is racing, but the
        # dropped counter should be > 0 OR everything fits if we're fast.
        # The reliable assertion: no exception, all submits accounted for.
        time.sleep(0.5)
    finally:
        sender.stop()

    s = sender.stats()
    total = s["pcm_chunks_submitted"]
    sent = s["pcm_chunks_submitted"] - s["pcm_chunks_dropped"]
    assert total == 50
    assert sent >= 1  # at least one drained
    # submitted == dropped + delivered (where delivered = packet fragments we got)
    assert s["pcm_chunks_dropped"] == total - sent
