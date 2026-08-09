"""Speaker TCP sender — byte-faithful PCM stream to the Android device.

Pipeline (FINAL ARCHITECTURE, speaker path):

    PC application
        -> Phone_Speaker sink (selected by user in GNOME Sound)
        -> Phone_Speaker.monitor (resolved via Pulse source index)
        -> Capture (pw-cat record)
        -> submit(pcm_bytes)
        -> AudioTcpSpeakerSender (this module)
        -> TCP socket -> Android TCP server :5000
        -> Android receive/playback buffer -> AudioTrack -> speaker

This sender is the production speaker transport. It is intentionally
minimal:

    * Bytes are passed through verbatim — no encoding, no compression,
      no resampling, no PCM-format conversion, no packetization, no
      UDP-style headers, no sequence numbers, no application framing.
    * TCP is only the transport — it gives us a reliable ordered
      byte stream on top of IP.
    * The Android speaker implementation MUST still use a proper
      receive/playback buffer and continuous AudioTrack writes;
      raw byte-stream transport does NOT magically guarantee
      glitch-free playback on its own.

Connection lifecycle (the production contract — see audio_main.py):

    1. The sender is started BEFORE capture so it can establish the
       TCP connection while the rest of the application boots.
    2. The caller MUST call ``wait_until_connected(timeout)`` to block
       until the connection is up. ``audio_main.py`` only starts the
       pw-cat speaker capture after this returns True.
    3. Once connected, ``submit(pcm_bytes)`` enqueues for the sender
       thread to write. The queue is bounded (3.2 s of audio). If it
       fills while connected, ``submit`` BLOCKS — it does not drop.
       This back-pressures the capture path so the sender stays at
       the same rate as the TCP write. The capture path is sized for
       this rate; filling the queue means the network cannot keep up
       and we must wait rather than silently drop audio.
    4. While disconnected, ``submit`` drops immediately. Stale audio
       that has been sitting in the queue from a previous connection
       is discarded — we never replay seconds-old PCM after a long
       disconnect.
    5. When the connection drops, the sender reconnects with
       exponential backoff (1 s, 2 s, 4 s, 8 s, capped at 30 s).
       ``add_state_callback`` notifies the caller so it can pause and
       restart the capture subprocess around the disconnect window.
    6. ``stop()`` is clean: closes the socket and joins the thread.

Threading model:

    * ``submit()`` is called from the audio capture callback thread.
      Behavior depends on connection state (see above).
    * A single sender daemon thread owns the TCP socket. It runs the
      connect / reconnect / send loop. All socket operations happen
      on this thread; ``submit`` never touches the socket directly.
"""
from __future__ import annotations

import logging
import queue
import socket
import threading

log = logging.getLogger("audio-bridge")


class AudioTcpSpeakerSender:
    """Byte-faithful TCP speaker transport.

    Lifecycle::

        sender = AudioTcpSpeakerSender(host="192.168.1.10", port=5000)
        sender.start()
        if not sender.wait_until_connected(timeout=10.0):
            raise RuntimeError("Android speaker unreachable")
        # ... capture callback calls sender.submit(pcm_bytes) ...
        sender.stop()

    Bytes arriving at ``submit()`` are forwarded to the Android device
    byte-for-byte, in order, over a single TCP connection. The sender
    auto-reconnects if the connection is lost.

    State callbacks: callers can register a callback via
    ``add_state_callback(callable)`` to be notified of every
    connection-state transition. This is what the production
    audio_main.py uses to pause pw-cat capture on disconnect and
    restart it on reconnect.
    """

    # Bounded PCM queue. Chunks are typically ~38 KB (100 ms @ 48 kHz
    # stereo float32). 32 chunks = ~3.2 s of buffering — enough to
    # ride out a short network blip. On a healthy connection the
    # queue never fills because the sender drains at the same rate
    # pw-cat produces. On disconnect, the queue is cleared (stale
    # audio is discarded — never replayed).
    _QUEUE_SIZE = 32

    # Reconnect backoff bounds.
    _RECONNECT_BACKOFF_INITIAL = 1.0   # seconds
    _RECONNECT_BACKOFF_MAX = 30.0     # seconds

    # send() / sendall() timeout — bound how long a single TCP write
    # can block the sender thread. Set high enough that a momentary
    # stall on the receiver does not spuriously disconnect; the
    # queue's bounded back-pressure is what actually rate-limits
    # the producer in the connected case.
    _SEND_TIMEOUT = 30.0

    def __init__(
        self,
        host: str,
        port: int,
        sock: socket.socket | None = None,
        queue_size: int = _QUEUE_SIZE,
    ):
        self._host = host
        self._port = port
        self._external_sock = sock
        self._sock: socket.socket | None = None
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._state_callbacks: list = []
        self._state_callbacks_lock = threading.Lock()
        self._sendall_override = None  # test seam
        self._stats_lock = threading.Lock()
        self._stats = {
            "bytes_sent": 0,
            "pcm_chunks_submitted": 0,
            "pcm_chunks_dropped_disconnected": 0,
            "pcm_chunks_blocked_full_queue": 0,
            "pcm_bytes_submitted": 0,
            "connection_attempts": 0,
            "connections_established": 0,
            "connections_lost": 0,
        }

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the sender thread. Idempotent.

        The thread begins attempting to connect to ``(host, port)``
        immediately. ``wait_until_connected()`` will block until the
        connection is up; ``submit()`` should not be called until that
        returns True (the production code wires this through the
        capture lifecycle).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._run, name="AudioTcpSpeakerSender", daemon=True,
        )
        self._thread.start()

    def wait_until_connected(self, timeout: float | None = None) -> bool:
        """Block until the TCP connection is established.

        Returns True if the connection came up within ``timeout``
        seconds, False otherwise (or if ``stop()`` was called). The
        caller MUST use this to gate any PCM production — without
        the connection, PCM will be discarded immediately.
        """
        return self._connected.wait(timeout=timeout)

    def add_state_callback(self, callback) -> None:
        """Register a function to be invoked on every connection-state
        transition.

        The callback receives a single boolean: True for "connected",
        False for "disconnected". Callbacks are invoked on the sender
        thread; they MUST return promptly and MUST NOT block on the
        sender's own state. Use a threading.Event or queue.Queue to
        hand work off to another thread if needed.
        """
        with self._state_callbacks_lock:
            self._state_callbacks.append(callback)

    def submit(self, pcm_bytes: bytes) -> None:
        """Enqueue PCM bytes for the sender thread to write.

        Called from the audio capture callback thread. Behavior depends
        on connection state:

        * **Connected**: enqueues into the bounded queue. If the queue
          is full, BLOCKS until the sender drains it. We deliberately
          do not drop during a healthy connection — the queue is sized
          for normal operation and dropping would create audible
          discontinuities. Blocking here back-pressures the capture
          path, which is rate-limited by PipeWire's monitor anyway.
        * **Disconnected**: drops immediately. We do NOT accumulate
          stale PCM during a disconnect, and we do NOT replay seconds
          of buffered audio on reconnect — we start fresh from the
          live monitor.
        """
        if not self._connected.is_set():
            with self._stats_lock:
                self._stats["pcm_chunks_dropped_disconnected"] += 1
                self._stats["pcm_chunks_submitted"] += 1
                self._stats["pcm_bytes_submitted"] += len(pcm_bytes)
            return

        with self._stats_lock:
            self._stats["pcm_chunks_submitted"] += 1
            self._stats["pcm_bytes_submitted"] += len(pcm_bytes)

        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Connected but the queue is full — back-pressure. Wait
            # for the sender thread to drain a slot. We respect
            # stop() so shutdown is prompt.
            with self._stats_lock:
                self._stats["pcm_chunks_blocked_full_queue"] += 1
            try:
                self._queue.put(pcm_bytes, timeout=1.0)
            except queue.Full:
                # Genuine stall — sender thread not draining. Drop
                # rather than deadlock the capture path. Logged once
                # per occurrence.
                with self._stats_lock:
                    self._stats["pcm_chunks_dropped_disconnected"] += 1
                log.warning(
                    "AudioTcpSpeakerSender queue stalled; dropped %d-byte "
                    "chunk while connected",
                    len(pcm_bytes),
                )

    def stop(self) -> None:
        """Stop the sender thread and close any open socket. Idempotent."""
        self._stop_event.set()
        # Wake any thread blocked in submit() on queue.put().
        try:
            self._queue.put_nowait(b"")
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._connected.clear()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def stats(self) -> dict:
        """Return a snapshot of the sender's counters."""
        with self._stats_lock:
            stats = dict(self._stats)
        stats["connected"] = self._connected.is_set()
        stats["queue_depth"] = self._queue.qsize()
        return stats

    def is_connected(self) -> bool:
        """True iff the TCP connection is currently established."""
        return self._connected.is_set()

    # -- internal ---------------------------------------------------------

    def _notify_state(self, connected: bool) -> None:
        """Invoke registered state callbacks under a snapshot of the list.

        The callbacks may be slow; holding the lock during the call
        would serialize them. We snapshot, then release, then call —
        so a slow callback does not block registration of new
        callbacks.
        """
        with self._state_callbacks_lock:
            callbacks = list(self._state_callbacks)
        for cb in callbacks:
            try:
                cb(connected)
            except Exception:
                log.exception("speaker state callback raised")

    def _clear_queue(self) -> None:
        """Drain the queue, discarding any stale PCM.

        Called on disconnect. We never replay buffered audio after a
        long disconnect — the user wants live audio from the moment
        the new connection is up.
        """
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            log.info(
                "Speaker TCP: discarded %d buffered PCM chunks after "
                "disconnect (will resume from live audio)",
                cleared,
            )

    def _run(self) -> None:
        """Sender thread main loop: connect → send loop → reconnect."""
        backoff = self._RECONNECT_BACKOFF_INITIAL
        while not self._stop_event.is_set():
            # 1. Try to (re)connect.
            sock = self._connect_with_backoff(backoff)
            if sock is None:
                # Stop event tripped during connect attempts.
                return
            backoff = self._RECONNECT_BACKOFF_INITIAL

            self._sock = sock
            self._connected.set()
            log.info(
                "Speaker TCP connected: %s:%d",
                self._host, self._port,
            )
            self._notify_state(True)

            # 2. Drain the queue until disconnect or stop.
            try:
                self._send_loop(sock)
            except (ConnectionError, OSError) as e:
                log.info(
                    "Speaker TCP connection lost (%s); will reconnect",
                    e,
                )
            finally:
                self._connected.clear()
                self._sock = None
                with self._stats_lock:
                    self._stats["connections_lost"] += 1
                try:
                    sock.close()
                except OSError:
                    pass
                # Discard buffered audio — do not replay stale PCM
                # when the next connection comes up.
                self._clear_queue()

            # 3. Notify listeners that we are now disconnected, then
            #    compute the next backoff.
            self._notify_state(False)
            backoff = min(backoff * 2, self._RECONNECT_BACKOFF_MAX)

    def _connect_with_backoff(
        self, initial_backoff: float,
    ) -> socket.socket | None:
        """Connect with exponential backoff. Returns None if stop was set."""
        backoff = initial_backoff
        while not self._stop_event.is_set():
            with self._stats_lock:
                self._stats["connection_attempts"] += 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._SEND_TIMEOUT)
                sock.connect((self._host, self._port))
                sock.settimeout(self._SEND_TIMEOUT)
            except (ConnectionError, OSError) as e:
                log.debug(
                    "Speaker TCP connect to %s:%d failed (%s); "
                    "retrying in %.1fs",
                    self._host, self._port, e, backoff,
                )
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                # Wait, but exit promptly on stop.
                if self._stop_event.wait(timeout=backoff):
                    return None
                backoff = min(backoff * 2, self._RECONNECT_BACKOFF_MAX)
                continue

            with self._stats_lock:
                self._stats["connections_established"] += 1
            return sock
        return None

    def _send_loop(self, sock: socket.socket) -> None:
        """Drain the queue and write to ``sock`` until error or stop.

        Raises whatever the underlying socket raises on disconnect;
        the outer ``_run`` loop handles reconnect.

        TCP only notices that the peer has closed the connection when
        we either send or receive on the socket. ``submit()`` drops
        PCM while disconnected, so on a quiet link the queue may be
        empty for long stretches — without a probe we would not
        notice a peer-side close for many seconds. We therefore call
        ``sock.recv(1, MSG_PEEK)`` whenever the queue is empty: this
        returns ``b''`` (zero-length) immediately when the peer has
        closed the connection, raising ``ConnectionResetError`` if the
        peer reset. This catches both FIN (clean close) and RST
        (hard close) promptly without consuming any data.
        """
        while not self._stop_event.is_set():
            try:
                pcm = self._queue.get(timeout=0.2)
            except queue.Empty:
                # Probe to detect peer-side close while the queue is
                # idle. ``MSG_PEEK`` keeps any bytes in the receive
                # buffer; an empty peek result means the peer closed.
                try:
                    peek = sock.recv(1, socket.MSG_PEEK)
                except (ConnectionError, OSError):
                    raise
                if peek == b"":
                    # Peer closed. Raise to drop into the reconnect
                    # path. ``ConnectionError`` is what the outer
                    # loop catches.
                    raise ConnectionError("peer closed connection")
                continue
            if not pcm:
                # Sentinel from stop(). Wake, then exit.
                continue
            # sendall() handles partial writes until the buffer is
            # fully flushed or an error occurs. The
            # ``_sendall_override`` test seam lets unit tests inject a
            # blocking sendall to exercise back-pressure without
            # touching the underlying socket.
            if self._sendall_override is not None:
                self._sendall_override(pcm)
            else:
                sock.sendall(pcm)
            with self._stats_lock:
                self._stats["bytes_sent"] += len(pcm)

    # -- test seam --------------------------------------------------------

    def _replace_sock_for_test(self, sock: socket.socket) -> None:
        """Test-only: install a socket that bypasses connect().

        Used by unit tests to inject a fake connected socket.
        """
        self._sock = sock
        self._connected.set()

    def _install_sendall_override_for_test(self, sendall) -> None:
        """Test-only: install a stub ``sendall`` used in place of the
        underlying socket's ``sendall``. Lets tests exercise back-
        pressure without touching the (read-only) ``socket.sendall``
        attribute on the C type.
        """
        self._sendall_override = sendall
