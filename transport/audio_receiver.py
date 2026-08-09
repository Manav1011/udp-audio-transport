"""Audio receiver: UDP socket → decode → jitter buffer → ordered PCM callbacks.

The JitterBuffer detects:
  - in-order packets (released immediately)
  - duplicates of already-released packets (dropped)
  - out-of-order packets (held briefly, released when gap fills)
  - gaps (advance past missing seqs one at a time after a time-based
    reorder window; never wipes the entire buffer)
  - late arrivals after a gap (dropped, since we've already moved on)

The AudioReceiver wraps a UDP socket and runs a recv loop on a daemon thread.
Received datagrams are decoded with audio_packet.decode_packet and pushed
into the JitterBuffer. Ordered payloads are delivered to the on_pcm callback.
The recv loop's existing socket timeout (0.5 s) doubles as the heartbeat
that drives jitter timeout evaluation when no packets are arriving.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable

from transport.audio_packet import decode_packet
from transport.audio_sequence_recorder import NullSequenceRecorder, SequenceIndexedPcmRecorder
from transport.audio_silence_inserter import NullSilenceInserter, SilenceInserter
from transport.udp_inspector import NullUdpInspector, UdpInspector

log = logging.getLogger("audio-bridge")

# uint32 sequence-number range used on the wire.
_SEQ_MOD = 1 << 32


class JitterBuffer:
    """Reorders packets, with a time-based reorder window.

    When a future seq arrives before the expected seq, we hold all later
    packets in a buffer until either (a) the missing seq arrives (closing
    the gap) or (b) reorder_window_ms elapses without it arriving, in which
    case we declare exactly one packet lost, advance next_expected by 1,
    and release whatever is now contiguous. The buffer is NEVER wiped in
    bulk — packets that arrived in time are always preserved.
    """

    def __init__(self, reorder_window_ms: int = 200):
        # Ordered payloads awaiting their predecessors.
        self._buffer: dict[int, bytes] = {}
        # Next seq to deliver to the injector. None until the first packet
        # syncs us.
        self._next_expected: int | None = None
        # Wall-clock time at which the current gap opened, or None if no gap.
        self._gap_started_at: float | None = None
        # Configurable reorder tolerance.
        self._reorder_window_s: float = reorder_window_ms / 1000.0
        # Instrumentation (no behavior impact): seqs released by the most
        # recent drain from push() or _skip_one_lost(), in order. Read by
        # the receiver loop to report per-chunk metadata to the sequence
        # recorder. Cleared at the start of every push().
        self._last_released_seqs: list[int] = []
        # Instrumentation: seq skipped by the most recent _skip_one_lost()
        # call (or None if no skip happened). Read by the receiver loop.
        self._last_lost_seq: int | None = None
        self._stats = {
            "received": 0,
            "duplicates": 0,
            "out_of_order": 0,
            "gaps": 0,
            "lost": 0,
            "late_released": 0,  # arrived after its seq was declared lost
        }

    # -- public API --------------------------------------------------------

    def push(self, seq: int, payload: bytes) -> list[bytes]:
        """Insert a packet. Returns ordered payloads ready to play.

        seq is interpreted as uint32. Wraparound is handled transparently.
        """
        # Normalize to uint32 (struct.unpack already gives unsigned int, but
        # be defensive against external callers).
        seq &= 0xFFFFFFFF
        self._stats["received"] += 1
        # Reset instrumentation per-call. The release list is filled in below
        # whenever we return a non-empty output.
        self._last_released_seqs = []
        self._last_lost_seq = None

        # First-ever packet: we have no reference for ordering, so accept
        # it as the "first sample". Set next_expected = seq+1 (we expect
        # the next packet in the stream to be seq+1). If the sender actually
        # started at seq=0 and we received a higher seq first (out-of-order
        # arrival of an early packet), the lower seq's will arrive later as
        # late arrivals and be counted as duplicates — which is the correct
        # behavior because we already moved past them.
        if self._next_expected is None:
            self._next_expected = (seq + 1) & 0xFFFFFFFF
            self._gap_started_at = None
            self._last_released_seqs = [seq]
            return [payload]

        # Forward distance from base to seq in the uint32 wrapped space.
        # 0       — same as next_expected → in-order
        # 1..2^31 — seq is in the future (delta of 1..2^31 packets ahead)
        # 2^31+1..2^32-1 — seq is in the past (late/duplicate)
        fwd = (seq - self._next_expected) % _SEQ_MOD

        # In-order (fwd == 0): close the gap if there was one, drain the
        # contiguous run. This is the only place that resets
        # _gap_started_at on successful delivery.
        if fwd == 0:
            self._gap_started_at = None
            output = [payload]
            released = [seq]
            next_seq = (seq + 1) & 0xFFFFFFFF
            while next_seq in self._buffer:
                output.append(self._buffer.pop(next_seq))
                released.append(next_seq)
                next_seq = (next_seq + 1) & 0xFFFFFFFF
            self._next_expected = next_seq
            self._last_released_seqs = released
            return output

        # fwd in [2^31+1 .. 2^32-1]: seq is behind (late/duplicate).
        if fwd > 0x80000000:
            # Same seq as a buffered future seq → duplicate of buffered.
            if seq in self._buffer:
                self._stats["duplicates"] += 1
                return []
            # Otherwise it's a true late arrival (already released).
            self._stats["duplicates"] += 1
            return []

        # fwd in [1 .. 2^31]: future seq. Open or extend a gap.
        # If this seq is already buffered, it's a duplicate.
        if seq in self._buffer:
            self._stats["duplicates"] += 1
            return []

        self._buffer[seq] = payload
        self._stats["out_of_order"] += 1
        if self._gap_started_at is None:
            self._gap_started_at = time.monotonic()
        return []

    def tick(self, now: float | None = None) -> list[bytes]:
        """Advance the timeout clock.

        Called periodically by the receiver loop (which already wakes every
        500 ms via socket timeout) so that gaps close even when no packets
        arrive. Returns any payloads that became ready as a result of
        skipping one or more genuinely-lost packets.
        """
        if self._gap_started_at is None or not self._buffer:
            return []
        if now is None:
            now = time.monotonic()
        if now - self._gap_started_at < self._reorder_window_s:
            return []
        return self._skip_one_lost()

    def _skip_one_lost(self) -> list[bytes]:
        """Declare exactly one seq lost, release whatever is now contiguous.

        Skips _next_expected (the seq that never showed up) by exactly 1,
        increments `lost` by 1, leaves the rest of the buffer intact, and
        drains contiguous buffered seqs starting from the new
        _next_expected.
        """
        self._stats["lost"] += 1
        self._stats["gaps"] += 1
        # Instrumentation: the seq we're about to declare lost is the one
        # that was the head of the gap (== _next_expected). The receiver
        # loop reads this back via _last_lost_seq.
        base = self._next_expected if self._next_expected is not None else 0
        lost_seq = base
        self._last_lost_seq = lost_seq
        new_next = (base + 1) & 0xFFFFFFFF
        output: list[bytes] = []
        released: list[int] = []
        while new_next in self._buffer:
            output.append(self._buffer.pop(new_next))
            released.append(new_next)
            new_next = (new_next + 1) & 0xFFFFFFFF
        # If the buffer is now empty, the gap is closed.
        # Otherwise, the gap now spans [new_next, next-future-seq).
        self._next_expected = new_next
        if self._buffer:
            # Reset the gap timer to "now" so the next missing seq gets a
            # fresh reorder window rather than inheriting the old one.
            self._gap_started_at = time.monotonic()
        else:
            self._gap_started_at = None
        self._last_released_seqs = released
        return output

    def stats(self) -> dict:
        return dict(self._stats)


class AudioReceiver:
    """Receives UDP packets and delivers ordered PCM bytes via on_pcm callback."""

    def __init__(
        self,
        on_pcm: Callable[[bytes], None],
        bind_host: str = "0.0.0.0",
        bind_port: int = 0,
        sock: socket.socket | None = None,
        jitter_buffer: JitterBuffer | None = None,
        inspector: UdpInspector | NullUdpInspector | None = None,
        sequence_recorder: SequenceIndexedPcmRecorder | NullSequenceRecorder | None = None,
        silence_inserter: SilenceInserter | NullSilenceInserter | None = None,
        reorder_window_ms: int = 200,
    ):
        self._on_pcm = on_pcm
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._external_sock = sock
        self._sock: socket.socket | None = None
        self._jitter = jitter_buffer or JitterBuffer(reorder_window_ms=reorder_window_ms)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "datagrams_received": 0,
            "decode_failures": 0,
            "pcm_bytes_delivered": 0,
            "packets_walked_out": 0,
            "silence_chunks_inserted": 0,
            "silence_bytes_inserted": 0,
        }
        self._inspector = inspector if inspector is not None else NullUdpInspector()
        self._sequence_recorder = (
            sequence_recorder if sequence_recorder is not None
            else NullSequenceRecorder()
        )
        self._silence_inserter = (
            silence_inserter if silence_inserter is not None
            else NullSilenceInserter()
        )

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self._external_sock is not None:
            self._sock = self._external_sock
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((self._bind_host, self._bind_port))
            self._sock.settimeout(0.5)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        log.info("AudioReceiver started on %s:%d", self._bind_host, self._local_port())

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._external_sock is None and self._sock is not None:
            self._sock.close()
        self._sock = None
        log.info("AudioReceiver stopped")

    def local_port(self) -> int:
        return self._local_port()

    def stats(self) -> dict:
        with self._stats_lock:
            merged = dict(self._stats)
        merged["jitter"] = self._jitter.stats()
        merged["silence"] = self._silence_inserter.stats()
        return merged

    # -- internal ---------------------------------------------------------

    def _local_port(self) -> int:
        if self._sock is None:
            return self._bind_port
        return self._sock.getsockname()[1]

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                # Heartbeat wakeup: socket timeout (0.5 s) lets us drive
                # the jitter-buffer timeout even when no packets arrive.
                released = self._jitter.tick()
                # tick() may have declared a packet lost; capture it.
                lost_seq = self._jitter._last_lost_seq
                if lost_seq is not None:
                    now = time.monotonic()
                    self._sequence_recorder.record_lost(lost_seq, now)
                    # Stage silence for the lost packet's slot — will be
                    # delivered by _deliver() in this same call, BEFORE the
                    # drained payloads that follow the gap.
                    self._silence_inserter.feed_loss(lost_seq, now)
                    self._jitter._last_lost_seq = None
                self._deliver(released)
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                log.exception("recv error")
                continue

            with self._stats_lock:
                self._stats["datagrams_received"] += 1

            decoded = decode_packet(data)
            if decoded is None:
                with self._stats_lock:
                    self._stats["decode_failures"] += 1
                self._inspector.record_decode_failure()
                continue

            seq, payload = decoded
            # Record what we received, including the declared length so the
            # inspector can flag any declared-vs-actual mismatch.
            # The header's payload_length field was already validated by
            # decode_packet, but we re-extract it here for the inspector.
            import struct as _struct
            if len(data) >= 8:
                _declared = _struct.unpack("!I", data[4:8])[0]
            else:
                _declared = 0
            self._inspector.record_packet(seq, _declared, len(payload))
            ordered = self._jitter.push(seq, payload)
            self._deliver(ordered)

    def _deliver(self, ordered: list[bytes]) -> None:
        """Forward jitter-released payloads to the injector callback.

        Ordering invariant: if the silence inserter has staged silence
        (because JitterBuffer._skip_one_lost() declared a loss in the same
        tick that produced `ordered`), the silence is delivered FIRST so
        the audio timeline places it in the missing packet's slot, BEFORE
        the buffered payload(s) that follow the gap.

        Silence is also delivered when `ordered` is empty — a loss may be
        declared without a drain (e.g. two consecutive missing seqs), and
        the silence for the lost slot must still reach the injector
        promptly so the playback timeline doesn't sit empty.
        """
        # 1) Inject any pending silence (loss-fill). Done unconditionally
        #    because silence represents audio-time that needs to advance the
        #    Injector's playback clock even when no real packets were drained.
        if self._silence_inserter.should_inject_silence():
            silence = self._silence_inserter.take_pending_silence()
            if silence is not None:
                now = time.monotonic()
                self._sequence_recorder.record_silence_injection(len(silence), now)
                with self._stats_lock:
                    self._stats["silence_chunks_inserted"] += 1
                    self._stats["silence_bytes_inserted"] += len(silence)
                self._inspector.record_pcm_chunk(silence)
                self._on_pcm(silence)
        if not ordered:
            return
        with self._stats_lock:
            self._stats["packets_walked_out"] += len(ordered)
            self._stats["pcm_bytes_delivered"] += sum(len(p) for p in ordered)
        # 2) Forward the received payloads in order. The jitter buffer
        #    populated _last_released_seqs (cleared at the start of push()).
        #    Pair each seq with its payload length so the recorder can
        #    compute PCM offsets correctly.
        released_seqs = list(self._jitter._last_released_seqs)
        if released_seqs and len(released_seqs) == len(ordered):
            self._sequence_recorder.record_delivered(
                released_seqs, [len(p) for p in ordered], time.monotonic()
            )
        for p in ordered:
            self._silence_inserter.observe_delivered_payload(p)
            self._inspector.record_pcm_chunk(p)
            self._on_pcm(p)
