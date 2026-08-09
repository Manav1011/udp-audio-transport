"""Android microphone TCP receiver — production mic path.

This is the TCP SERVER side of the Android microphone transport. The
Android app is the TCP CLIENT and connects to <backend_ip>:5002.

Pipeline (FINAL ARCHITECTURE, microphone path):

    Android AudioRecord
        -> existing Android PCM conversion
        -> TCP stream
        -> AudioTcpMicReceiver (this module)
        -> on_pcm callback
        -> AudioSession -> injector.write_frames
        -> Phone_Microphone sink
        -> apps recording from Phone_Microphone_Input

The receiver is byte-faithful: every byte received from the socket is
passed to on_pcm exactly as it arrived. There is no application-level
framing, no reordering, no jitter buffer, no resampling, no DSP, no
format conversion on the backend.

The existing PCM contract is preserved:
  - 48000 Hz
  - stereo
  - Float32 LE
  - interleaved [L, R, L, R, ...]
  - 8 bytes per stereo frame

The receiver does NOT validate frames. The Android sender is trusted
to emit well-formed PCM. We pass everything through.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable

log = logging.getLogger("audio-bridge")

# 64 KiB recv buffer — balances throughput vs. reporting granularity.
_RECV_CHUNK = 65536


class AudioTcpMicReceiver:
    """TCP listener that accepts one Android mic connection and forwards
    PCM bytes to the on_pcm callback.

    Lifecycle:
        recv = AudioTcpMicReceiver(on_pcm=injector.write_frames,
                                   bind_host="0.0.0.0",
                                   bind_port=5002)
        recv.start()
        # ... Android connects, streams PCM ...
        recv.stop()

    If a previous connection drops, the receiver keeps listening and
    accepts a new one (Android may reconnect on network blips).
    """

    def __init__(
        self,
        on_pcm: Callable[[bytes], None],
        bind_host: str = "0.0.0.0",
        bind_port: int = 5002,
        sock: socket.socket | None = None,
    ):
        self._on_pcm = on_pcm
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._external_sock = sock
        self._sock: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "connections_accepted": 0,
            "connections_active": 0,
            "bytes_received": 0,
            "recv_calls": 0,
            "pcm_chunks_delivered": 0,
        }

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self._external_sock is not None:
            self._sock = self._external_sock
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._bind_host, self._bind_port))
            self._sock.listen(1)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._external_sock is None and self._sock is not None:
            self._sock.close()
        self._sock = None

    def local_port(self) -> int:
        if self._sock is None:
            return self._bind_port
        return self._sock.getsockname()[1]

    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    # -- internal ---------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._sock is not None
        self._sock.settimeout(0.5)
        while not self._stop_event.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                log.exception("TCP accept error")
                continue
            log.info("Microphone TCP client connected: %s:%s", *addr)
            with self._stats_lock:
                self._stats["connections_accepted"] += 1
                self._stats["connections_active"] += 1
            # Serve this connection. If it drops, the outer accept loop
            # keeps running so the Android side can reconnect.
            try:
                self._serve_connection(conn)
            except Exception:
                log.exception("TCP mic connection handler crashed")
            finally:
                with self._stats_lock:
                    self._stats["connections_active"] -= 1

    def _serve_connection(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        while not self._stop_event.is_set():
            try:
                data = conn.recv(_RECV_CHUNK)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                log.debug("TCP mic: connection closed by peer")
                return
            with self._stats_lock:
                self._stats["recv_calls"] += 1
            if not data:
                # EOF: client closed cleanly.
                log.debug("TCP mic: EOF received")
                return
            with self._stats_lock:
                self._stats["bytes_received"] += len(data)
                self._stats["pcm_chunks_delivered"] += 1
            # Forward to the on_pcm callback. The injector contract
            # accepts raw PCM bytes and writes them to the PipeWire
            # Phone_Microphone sink. No validation, no framing.
            self._on_pcm(data)
