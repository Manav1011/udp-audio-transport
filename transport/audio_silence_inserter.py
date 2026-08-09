"""Silence inserter for genuinely lost jitter-buffer packets.

When JitterBuffer._skip_one_lost() declares a sequence number lost, the
receiver normally emits NO PCM for that slot — so the next available chunk
immediately follows the previous one. This causes audible discontinuities
(clicks/static) at the Injector boundary because the playback timeline
"skips" the missing audio.

This module fills those missing slots with zero-valued interleaved stereo
float32 LE PCM (i.e., digital silence). The audio timeline is preserved.

The wire carries no signal that conveys the size of a packet that did NOT
arrive. We therefore INFER the lost packet's expected payload length from
context: use the most-recently-delivered payload length (typically1152
bytes = 144 stereo frames = 3 ms @ 48 kHz), falling back to MAX_PAYLOAD
from transport.audio_packet if no packet has been delivered yet. This
matches the sender's actual behavior: all packets are MAX_PAYLOAD except
the very last partial packet of a session.

DIAGNOSTIC-FIRST: enabled only when AUDIO_DIAGNOSTIC_SILENCE=1. When
disabled, NullSilenceInserter is a zero-overhead no-op and behavior is
identical to the pre-change receiver.

This module participates in the receiver's PCM output stream ONLY by
emitting silence bytes; it does NOT modify received PCM, does NOT alter
sequence ordering, and does NOT silence reordered packets that close their
gap within the jitter reorder window.

Constraints from the user (preserved verbatim):
  - DO NOT modify Android code, UDP packet format, packetization, PCM
    format, PipeWire, Injector, jitter-buffer reordering behavior.
  - DO NOT claim the audible static is fixed until a real Android test.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from transport.audio_packet import MAX_PAYLOAD

log = logging.getLogger("audio-bridge")

# Float32 stereo frame = 2 channels * 4 bytes. This is the same constant used
# by the sequence recorder and is locked in by the audio format (48 kHz, 2ch,
# float32 LE) configured in audio/injector.py and audio/capture.py.
STEREO_FRAME_BYTES = 8

JSON_PATH = "/tmp/audio_silence_log.json"
SUMMARY_PATH = "/tmp/audio_silence_summary.txt"


def is_enabled() -> bool:
    """Return True iff the silence inserter should run."""
    return os.environ.get("AUDIO_DIAGNOSTIC_SILENCE") == "1"


@dataclass
class LostPacketEvent:
    seq: int
    inferred_length: int
    missing_frames: int
    ts: float


@dataclass
class SilenceInjection:
    n_bytes: int
    n_frames: int
    ts: float


class NullSilenceInserter:
    """Zero-overhead no-op stand-in used when AUDIO_DIAGNOSTIC_SILENCE is unset."""

    def observe_delivered_payload(self, payload: bytes) -> None:
        pass

    def feed_loss(self, seq: int, now: float) -> None:
        pass

    def should_inject_silence(self) -> bool:
        return False

    def take_pending_silence(self) -> Optional[bytes]:
        return None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def stats(self) -> dict:
        return {}


class SilenceInserter:
    """Stages silence bytes for delivery between received PCM chunks.

    Lifecycle:
      1. start()                — record start time, open JSON/TXT paths.
      2. observe_delivered_payload(p) — called by the receiver after each
         delivered real chunk. Updates the inference context.
      3. feed_loss(seq, now)    — called by the receiver after JitterBuffer
         declares a packet lost. Stages the inferred-size silence bytes.
      4. take_pending_silence() — called by _deliver() to fetch the staged
         silence and clear the pending slot.
      5. stop()                 — writes JSON log and human summary.

    Counters exposed via stats() (required by the user):
      - lost_seq_count              — packets declared lost
      - missing_frame_count         — stereo frames inferred to be missing
      - inserted_silence_frame_count— silence frames actually delivered
      - cumulative_inserted_frames  — running total of inserted silence
      - cumulative_lost_packets     — running total of declared losses
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Inference context: most recent delivered payload length.
        self._last_delivered_payload_length: int | None = None
        # Staged silence bytes awaiting delivery by _deliver().
        self._pending_silence: bytes | None = None
        # Last inferred length, for diagnostics.
        self._last_inferred_payload_length: int | None = None
        # Event logs.
        self._lost_packets: list[LostPacketEvent] = []
        self._injections: list[SilenceInjection] = []
        # Counters (also accumulated via the lists, but kept as scalars for
        # O(1) stats() reads).
        self._lost_seq_count: int = 0
        self._missing_frame_count: int = 0
        self._inserted_silence_frame_count: int = 0
        self._cumulative_inserted_frames: int = 0
        self._cumulative_lost_packets: int = 0
        # Lifecycle.
        self._started_at: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._started_at = time.monotonic()
        log.info("SilenceInserter started — output %s", JSON_PATH)

    def stop(self) -> None:
        try:
            payload = {
                "lost_packets": [
                    {
                        "seq": ev.seq,
                        "inferred_length": ev.inferred_length,
                        "missing_frames": ev.missing_frames,
                        "ts": ev.ts,
                    }
                    for ev in self._lost_packets
                ],
                "injections": [
                    {
                        "n_bytes": inj.n_bytes,
                        "n_frames": inj.n_frames,
                        "ts": inj.ts,
                    }
                    for inj in self._injections
                ],
                "summary": self.stats(),
            }
            with open(JSON_PATH, "w") as f:
                json.dump(payload, f)
            log.info(
                "SilenceInserter wrote %d lost, %d injections to %s",
                len(self._lost_packets), len(self._injections), JSON_PATH,
            )
        except OSError as e:
            log.error("Failed to write silence log: %s", e)
        try:
            self._write_summary()
        except OSError as e:
            log.error("Failed to write silence summary: %s", e)
        log.info("SilenceInserter stopped")

    # -- recording ---------------------------------------------------------

    def observe_delivered_payload(self, payload: bytes) -> None:
        """Called by the receiver after each delivered real packet (NOT silence).
        The length is the inference context for any future loss.
        """
        with self._lock:
            self._last_delivered_payload_length = len(payload)

    def feed_loss(self, seq: int, now: float) -> None:
        """Called when JitterBuffer._skip_one_lost() declared a packet lost.
        Stages zero-valued PCM of the inferred length for the next _deliver().
        """
        length = self.infer_lost_payload_length()
        # Floor to whole frames; warn if length was not frame-aligned (this
        # should not happen because the sender never emits non-aligned
        # payloads, but be defensive).
        n_frames = length // STEREO_FRAME_BYTES
        if n_frames == 0:
            log.warning(
                "SilenceInserter: lost seq=%d inferred_length=%d is "
                "less than one stereo frame (%d bytes); emitting 1 frame",
                seq, length, STEREO_FRAME_BYTES,
            )
            n_frames = 1
            length = STEREO_FRAME_BYTES
        silence = b"\x00" * (n_frames * STEREO_FRAME_BYTES)
        with self._lock:
            self._pending_silence = silence
            self._last_inferred_payload_length = length
            self._lost_packets.append(LostPacketEvent(
                seq=seq, inferred_length=length,
                missing_frames=n_frames, ts=now,
            ))
            self._lost_seq_count += 1
            self._missing_frame_count += n_frames
            self._cumulative_lost_packets += 1

    def should_inject_silence(self) -> bool:
        """True iff take_pending_silence() will return non-None."""
        with self._lock:
            return self._pending_silence is not None

    def take_pending_silence(self) -> Optional[bytes]:
        """Called by AudioReceiver._deliver() to fetch the staged silence.
        Updates injection counters and clears the pending slot.
        """
        with self._lock:
            silence = self._pending_silence
            self._pending_silence = None
        if silence is None:
            return None
        n_frames = len(silence) // STEREO_FRAME_BYTES
        self._injections.append(SilenceInjection(
            n_bytes=len(silence), n_frames=n_frames, ts=time.monotonic(),
        ))
        self._inserted_silence_frame_count += n_frames
        self._cumulative_inserted_frames += n_frames
        return silence

    # -- inference ---------------------------------------------------------

    def infer_lost_payload_length(self) -> int:
        """Infer the expected payload length of the most-recently-declared-lost seq.

        Rule:
          - If we have already delivered at least one real packet, return its
            payload length (typical: 1152; trailing partial packets are rare
            and short, but if the trailing packet is lost the session is
            about to end and emitting 1152 bytes of silence at the tail is
            benign).
          - Else: return MAX_PAYLOAD (the single source of truth from
            transport/audio_packet.py). 1152 bytes = 144 stereo frames.
        """
        with self._lock:
            if self._last_delivered_payload_length is not None:
                return self._last_delivered_payload_length
        return MAX_PAYLOAD

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "lost_seq_count": self._lost_seq_count,
                "missing_frame_count": self._missing_frame_count,
                "inserted_silence_frame_count": self._inserted_silence_frame_count,
                "cumulative_inserted_frames": self._cumulative_inserted_frames,
                "cumulative_lost_packets": self._cumulative_lost_packets,
                "last_inferred_payload_length": self._last_inferred_payload_length,
            }

    def _write_summary(self) -> None:
        lines = []
        lines.append("Silence inserter — session summary")
        lines.append("=" * 60)
        lines.append(f"lost packets (declared):      {self._cumulative_lost_packets}")
        lines.append(f"missing frames (inferred):    {self._missing_frame_count}")
        lines.append(f"silence injections delivered: {len(self._injections)}")
        lines.append(f"silence frames inserted:      {self._cumulative_inserted_frames}")
        lines.append(f"distinct inferred lengths:    "
                     f"{sorted({ev.inferred_length for ev in self._lost_packets}) or '(none)'}")
        if self._lost_packets:
            lines.append(f"first lost seq: {self._lost_packets[0].seq}")
            lines.append(f"last lost seq:  {self._lost_packets[-1].seq}")
            lines.append(f"lost seqs (first 20): "
                         f"{[ev.seq for ev in self._lost_packets[:20]]}"
                         f"{' ...' if len(self._lost_packets) > 20 else ''}")
        with open(SUMMARY_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        log.info("Silence summary written to %s", SUMMARY_PATH)