"""Structured UDP/PCM inspector (TEMPORARY — observation only).

Sits between the AudioReceiver's on_pcm callback and the AudioSession's
_deliver_to_injector. Records every UDP packet (count, declared payload
length, sequence number), tracks sequence-number discontinuities, and
records the cumulative PCM byte stream. Emits a periodic summary.

Enabled only when AUDIO_DIAGNOSTIC_INSPECT=1. When disabled the class
is a no-op pass-through and costs nothing.

This module is part of a temporary diagnostic patch. It does not
participate in any control flow; the production path proceeds
identically whether the inspector is enabled or not.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter

log = logging.getLogger("audio-bridge")


def is_enabled() -> bool:
    return os.environ.get("AUDIO_DIAGNOSTIC_INSPECT") == "1"


class NullUdpInspector:
    """No-op pass-through when AUDIO_DIAGNOSTIC_INSPECT is unset."""

    def record_packet(self, seq: int, declared_length: int,
                      actual_payload_length: int) -> None:
        pass

    def record_pcm_chunk(self, pcm: bytes) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class UdpInspector:
    """Observes UDP packets, sequence numbers, and reconstructed PCM.

    Threading: record_packet() is called from the receiver thread;
    record_pcm_chunk() is called from the same thread (it's the on_pcm
    callback). The periodic summary runs on a daemon thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packets_total = 0
        self._decode_failures = 0
        self._payload_lengths = Counter()  # declared -> count
        self._payload_length_mismatches = 0  # declared != actual
        self._payload_lengths_actual = Counter()
        # Sequence-number tracking.
        self._seq_first: int | None = None
        self._seq_last: int | None = None
        self._seq_gaps: list[tuple[int, int]] = []  # (expected, got)
        self._seq_duplicates: list[int] = []
        # PCM tracking.
        self._pcm_chunks = 0
        self._pcm_bytes = 0
        self._pcm_chunk_lengths = Counter()  # chunk size -> count
        # First 5 reconstructed PCM chunks as bytes (raw, for inspection).
        self._first_chunks: list[bytes] = []
        self._first_chunk_count = 5
        # Decoded bytes that look like a packet header (uint32 seq, uint32 length)
        # accidentally appearing inside PCM — used to detect header leaks.
        self._header_in_pcm_suspicions = 0
        # Stop signal.
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    # -- recording ---------------------------------------------------------

    def record_packet(self, seq: int, declared_length: int,
                      actual_payload_length: int) -> None:
        """Called by the receiver thread for every successfully decoded packet.

        declared_length = what the header says the payload length is.
        actual_payload_length = len(payload) actually decoded.
        """
        with self._lock:
            self._packets_total += 1
            self._payload_lengths[declared_length] += 1
            self._payload_lengths_actual[actual_payload_length] += 1
            if declared_length != actual_payload_length:
                self._payload_length_mismatches += 1
            if self._seq_first is None:
                self._seq_first = seq
            else:
                expected = self._seq_last + 1
                if seq == self._seq_last:
                    self._seq_duplicates.append(seq)
                elif seq != expected:
                    # Gap or out-of-order. Record both directions.
                    self._seq_gaps.append((expected, seq))
            self._seq_last = seq

    def record_decode_failure(self) -> None:
        with self._lock:
            self._decode_failures += 1

    def record_pcm_chunk(self, pcm: bytes) -> None:
        """Called by the receiver thread for every ordered PCM chunk released
        from the jitter buffer (i.e. exactly what is passed to the injector).
        """
        with self._lock:
            self._pcm_chunks += 1
            self._pcm_bytes += len(pcm)
            self._pcm_chunk_lengths[len(pcm)] += 1
            if len(self._first_chunks) < self._first_chunk_count:
                # Save the first few raw chunks for postmortem inspection.
                self._first_chunks.append(pcm)
            # Heuristic: if a chunk's length is exactly 8 and the first 8 bytes
            # look like a plausible uint32 header (small int values), flag it.
            if len(pcm) == 8:
                self._header_in_pcm_suspicions += 1

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._summary_loop, daemon=True)
        self._thread.start()
        log.info("UdpInspector started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        # Final report — non-periodic, fires once on shutdown.
        self._emit_summary(final=True)
        log.info("UdpInspector stopped")

    def _summary_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            self._emit_summary(final=False)

    def _emit_summary(self, final: bool) -> None:
        with self._lock:
            elapsed = max(time.time() - self._start_time, 1e-6)
            packets = self._packets_total
            failures = self._decode_failures
            declared_dist = dict(self._payload_lengths)
            actual_dist = dict(self._payload_lengths_actual)
            mismatches = self._payload_length_mismatches
            seq_first = self._seq_first
            seq_last = self._seq_last
            gaps = list(self._seq_gaps[:20])
            gap_count = len(self._seq_gaps)
            dups = list(self._seq_duplicates[:20])
            dup_count = len(self._seq_duplicates)
            pcm_chunks = self._pcm_chunks
            pcm_bytes = self._pcm_bytes
            pcm_chunk_dist = dict(self._pcm_chunk_lengths)
            first_chunks = list(self._first_chunks)
            suspicions = self._header_in_pcm_suspicions

        tag = "FINAL " if final else ""
        pps = packets / elapsed
        bps = pcm_bytes / elapsed
        gap_summary = ""
        if gaps:
            shown = ", ".join(f"({e}->{g})" for e, g in gaps[:10])
            gap_summary = f" first 10 gaps (expected->got): [{shown}]"
        dup_summary = ""
        if dups:
            shown = ", ".join(str(s) for s in dups[:10])
            dup_summary = f" first 10 duplicate seqs: [{shown}]"
        log.info(
            "%sUdpInspector[%5.1fs]: packets=%d (%.1f/s) decode_failures=%d "
            "payload_declared=%s payload_actual=%s mismatches=%d "
            "seq=%s..%s gaps=%d%s dups=%d%s "
            "pcm_chunks=%d pcm_bytes=%d (%.0f B/s) pcm_chunk_sizes=%s "
            "8-byte-chunks=%d first_chunks=%d",
            tag, elapsed, packets, pps, failures,
            _truncate_dict(declared_dist), _truncate_dict(actual_dist),
            mismatches,
            seq_first, seq_last, gap_count, gap_summary,
            dup_count, dup_summary,
            pcm_chunks, pcm_bytes, bps, _truncate_dict(pcm_chunk_dist),
            suspicions, len(first_chunks),
        )
        # Also dump the first chunk bytes verbatim (hex) so we can spot
        # accidental header-in-pcm contamination.
        for i, chunk in enumerate(first_chunks):
            head = chunk[:32].hex()
            tail = chunk[-16:].hex() if len(chunk) > 32 else ""
            log.info("  first_chunk[%d]: len=%d head=%s%s",
                     i, len(chunk), head, f" tail={tail}" if tail else "")


def _truncate_dict(d: dict, max_items: int = 8) -> str:
    if len(d) <= max_items:
        return str(d)
    items = list(d.items())[:max_items]
    return f"{{{', '.join(f'{k}:{v}' for k, v in items)}, ...}}"
