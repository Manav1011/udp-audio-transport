"""UDP transport: sends and receives raw bytes.

Knows nothing about packets, JSON, or protocol types.
"""
from __future__ import annotations

import socket
import threading
from typing import Callable

from utils.logger import log


class UDPTransport:
    """Low-level UDP socket wrapper.

    Provides:
      - start() / stop()
      - send_bytes(bytes)
      - receive_bytes(callback) where callback(bytes, addr)
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._client: tuple[str, int] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: Callable[[bytes, tuple[str, int]], None] | None = None

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        log.info("UDP transport started on %s:%d", self.host, self.port)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def send_bytes(self, data: bytes) -> None:
        if self._client is None:
            log.warning("No client yet — cannot send reply")
            return
        self._sock.sendto(data, self._client)
        log.info("Sent %d bytes to %s:%d", len(data), *self._client)

    def receive_bytes(self, callback: Callable[[bytes, tuple[str, int]], None]) -> None:
        """Register a callback invoked for each received datagram."""
        self._callback = callback

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
        log.info("UDP transport stopped")

    # -- internals ---------------------------------------------------------

    def _recv_loop(self) -> None:
        buf = 65535
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(buf)
            except OSError as exc:
                if self._stop_event.is_set():
                    return
                log.error("recv error: %s", exc)
                continue
            self._client = addr
            if self._callback is not None:
                try:
                    self._callback(data, addr)
                except Exception:
                    log.exception("callback raised")
