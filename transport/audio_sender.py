"""Audio sender: queue PCM → fragment into packets → UDP send.

PCM bytes arriving from capture.py are typically ~38 KB chunks (100 ms @ 48 kHz
stereo float32). Each chunk is split into MAX_PAYLOAD-sized payloads, each
prepended with an 8-byte header (sequence_number, payload_length).

Sequence numbers are monotonically increasing across the lifetime of the
sender. The header carries them as big-endian uint32 — sends wrap at 2^32
(about 414 days at 167 pps, not a concern).
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
from typing import Tuple

from transport.audio_packet import MAX_PAYLOAD, encode_packet

log = logging.getLogger("audio-bridge")


class AudioSender:
    """Fragments PCM into UDP packets and sends to a configured destination."""

    def __init__(
        self,
        dest: Tuple[str, int],
        queue_size: int = 32,
        sock: socket.socket | None = None,
    ):
        self.dest = dest
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_size)
        self._external_sock = sock
        self._sock: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stats = {
            "packets_sent": 0,
            "bytes_sent": 0,
            "pcm_chunks_submitted": 0,
            "pcm_chunks_dropped": 0,
            "pcm_bytes_submitted": 0,
        }

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self._external_sock is not None:
            self._sock = self._external_sock
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        log.debug("AudioSender started → %s:%d", *self.dest)

    def submit(self, pcm_bytes: bytes) -> None:
        """Non-blocking enqueue. Drops if queue is full."""
        self._stats["pcm_chunks_submitted"] += 1
        self._stats["pcm_bytes_submitted"] += len(pcm_bytes)
        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            self._stats["pcm_chunks_dropped"] += 1
            log.warning("AudioSender queue full; dropped %d-byte chunk", len(pcm_bytes))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._external_sock is None and self._sock is not None:
            self._sock.close()
        self._sock = None
        log.debug("AudioSender stopped")

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # -- internal ---------------------------------------------------------

    def _send_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                pcm = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._send_fragmented(pcm)

    def _send_fragmented(self, pcm: bytes) -> None:
        assert self._sock is not None
        for offset in range(0, len(pcm), MAX_PAYLOAD):
            payload = pcm[offset:offset + MAX_PAYLOAD]
            with self._lock:
                seq = self._seq
                self._seq += 1
            datagram = encode_packet(seq, payload)
            self._sock.sendto(datagram, self.dest)
            self._stats["packets_sent"] += 1
            self._stats["bytes_sent"] += len(datagram)
