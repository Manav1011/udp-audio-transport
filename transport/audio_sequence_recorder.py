"""Sequence-indexed PCM recorder (TEMPORARY — instrumentation only).

Sits alongside the diagnostic PCM writer and records metadata about every
PCM chunk released by the jitter buffer, keyed by UDP sequence number.
This lets us, after the session, reconstruct:

  - which sequence numbers were delivered (and in what order)
  - which sequence numbers were declared lost
  - the PCM byte / frame offset at which each chunk was delivered
  - the payload length of every packet
  - the wall-clock delivery timestamp of every released chunk

Outputs a JSON file at /tmp/audio_sequence_log.json with two arrays:
  - "delivered": [{seq, pcm_offset_bytes, pcm_offset_frames,
                    payload_length, delivery_ts}, ...]
  - "lost":      [{seq, lost_ts, near_delivered_seqs}, ...]

Also writes /tmp/audio_sequence_summary.txt with a human-readable summary.

The recorder is enabled only when AUDIO_DIAGNOSTIC_SEQUENCE=1. When
disabled it is a no-op pass-through with zero overhead.

This module is part of a temporary diagnostic patch. It does not
participate in any control flow; the production path proceeds identically
whether the recorder is enabled or not.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger("audio-bridge")

JSON_PATH = "/tmp/audio_sequence_log.json"
SUMMARY_PATH = "/tmp/audio_sequence_summary.txt"
STEREO_FRAME_BYTES = 8  # 2 channels * 4 bytes (float32)


def is_enabled() -> bool:
    return os.environ.get("AUDIO_DIAGNOSTIC_SEQUENCE") == "1"


@dataclass
class DeliveredChunk:
    seq: int
    pcm_offset_bytes: int
    pcm_offset_frames: int
    payload_length: int
    delivery_ts: float


@dataclass
class LostPacket:
    seq: int
    lost_ts: float
    near_delivered_seqs: list[int] = field(default_factory=list)


@dataclass
class SilenceInjection:
    """Records a silence chunk emitted by the SilenceInserter (instrumentation
    only — silence is NOT a 'delivered' packet and is NOT assigned a seq)."""
    n_bytes: int
    n_frames: int
    injection_ts: float


class NullSequenceRecorder:
    """No-op stand-in when AUDIO_DIAGNOSTIC_SEQUENCE is unset."""

    def record_delivered(self, seqs: Iterable[int], payload_lengths: Iterable[int],
                         now: float) -> None:
        pass

    def record_lost(self, seq: int, now: float) -> None:
        pass

    def record_silence_injection(self, n_bytes: int, now: float) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class SequenceIndexedPcmRecorder:
    """Records every delivered PCM chunk and every declared loss by seq."""

    def __init__(self) -> None:
        self._delivered: list[DeliveredChunk] = []
        self._lost: list[LostPacket] = []
        self._silence_injections: list[SilenceInjection] = []
        self._pcm_offset_bytes: int = 0
        self._started_at: float = 0.0

    def start(self) -> None:
        self._started_at = time.monotonic()
        log.info("SequenceIndexedPcmRecorder started — output %s", JSON_PATH)

    def stop(self) -> None:
        # Write JSON log
        try:
            payload = {
                "delivered": [
                    {
                        "seq": c.seq,
                        "pcm_offset_bytes": c.pcm_offset_bytes,
                        "pcm_offset_frames": c.pcm_offset_frames,
                        "payload_length": c.payload_length,
                        "delivery_ts": c.delivery_ts,
                    }
                    for c in self._delivered
                ],
                "lost": [
                    {
                        "seq": l.seq,
                        "lost_ts": l.lost_ts,
                        "near_delivered_seqs": l.near_delivered_seqs,
                    }
                    for l in self._lost
                ],
                "silence_injections": [
                    {
                        "n_bytes": s.n_bytes,
                        "n_frames": s.n_frames,
                        "injection_ts": s.injection_ts,
                    }
                    for s in self._silence_injections
                ],
            }
            with open(JSON_PATH, "w") as f:
                json.dump(payload, f)
            log.info(
                "SequenceIndexedPcmRecorder wrote %d delivered, %d lost, "
                "%d silence to %s",
                len(self._delivered), len(self._lost),
                len(self._silence_injections), JSON_PATH,
            )
        except OSError as e:
            log.error("Failed to write sequence log: %s", e)
        # Write human-readable summary
        try:
            self._write_summary()
        except OSError as e:
            log.error("Failed to write sequence summary: %s", e)
        log.info("SequenceIndexedPcmRecorder stopped")

    # -- recording ---------------------------------------------------------

    def record_delivered(self, seqs: Iterable[int], payload_lengths: Iterable[int],
                         now: float) -> None:
        """Called by the receiver loop after the jitter buffer releases
        a list of payloads (in order).
        """
        seqs = list(seqs)
        lens = list(payload_lengths)
        for seq, length in zip(seqs, lens):
            frames = length // STEREO_FRAME_BYTES
            self._delivered.append(DeliveredChunk(
                seq=seq,
                pcm_offset_bytes=self._pcm_offset_bytes,
                pcm_offset_frames=self._pcm_offset_bytes // STEREO_FRAME_BYTES,
                payload_length=length,
                delivery_ts=now,
            ))
            self._pcm_offset_bytes += length

    def record_lost(self, seq: int, now: float) -> None:
        """Called by the jitter buffer when it declares a packet lost."""
        # Capture the nearest delivered seqs around this loss for context.
        if self._delivered:
            tail = [c.seq for c in self._delivered[-3:]]
        else:
            tail = []
        self._lost.append(LostPacket(
            seq=seq,
            lost_ts=now,
            near_delivered_seqs=tail,
        ))

    def record_silence_injection(self, n_bytes: int, now: float) -> None:
        """Called by the silence inserter (via the receiver) after it
        delivers a silence chunk. The silence chunk advances the recorder's
        PCM-offset accounting the same way a real delivered packet would,
        because silence occupies a slot in the OUTPUT audio timeline.
        It is recorded as a separate event for diagnostics only — silence
        is NOT in the 'delivered' list because it has no seq."""
        n_frames = n_bytes // STEREO_FRAME_BYTES
        self._silence_injections.append(SilenceInjection(
            n_bytes=n_bytes, n_frames=n_frames, injection_ts=now,
        ))
        # Advance PCM offset so timeline accounting matches the OUTPUT stream.
        self._pcm_offset_bytes += n_bytes

    # -- summary -----------------------------------------------------------

    def _write_summary(self) -> None:
        n_del = len(self._delivered)
        n_lost = len(self._lost)
        n_silence = len(self._silence_injections)
        silence_frames = sum(s.n_frames for s in self._silence_injections)
        silence_bytes = sum(s.n_bytes for s in self._silence_injections)
        total_bytes = sum(c.payload_length for c in self._delivered)
        total_frames = total_bytes // STEREO_FRAME_BYTES
        delivered_seqs = [c.seq for c in self._delivered]
        lost_seqs = [l.seq for l in self._lost]
        # Detect payload-length inconsistencies.
        payload_lengths = sorted({c.payload_length for c in self._delivered})
        # Detect frame/byte alignment of every chunk.
        misaligned = [c for c in self._delivered
                      if c.payload_length % STEREO_FRAME_BYTES != 0]

        lines = []
        lines.append("Sequence-indexed PCM recorder — session summary")
        lines.append("=" * 60)
        lines.append(f"delivered chunks:  {n_del}")
        lines.append(f"lost packets:      {n_lost}")
        lines.append(f"silence injections:{n_silence}  "
                     f"({silence_bytes} bytes / {silence_frames} frames)")
        lines.append(f"total PCM bytes:   {total_bytes}")
        lines.append(f"total PCM frames:  {total_frames}")
        lines.append(f"distinct payload lengths: {payload_lengths}")
        lines.append(f"misaligned chunks (length not divisible by 8): {len(misaligned)}")
        if misaligned:
            for c in misaligned[:5]:
                lines.append(f"  seq={c.seq} len={c.payload_length}")
        if delivered_seqs:
            lines.append(f"first delivered seq: {delivered_seqs[0]}")
            lines.append(f"last delivered seq:  {delivered_seqs[-1]}")
        if lost_seqs:
            lines.append(f"first lost seq:      {lost_seqs[0]}")
            lines.append(f"last lost seq:       {lost_seqs[-1]}")
            lines.append(f"lost seqs:           {lost_seqs}")
        # Detect byte gaps: delivered chunks should be contiguous in PCM
        # offset (each chunk's offset = previous offset + previous length).
        # With the silence inserter enabled, gaps SHOULD remain zero because
        # silence is recorded separately and does NOT advance pcm_offset_bytes.
        gaps_in_pcm = []
        prev_offset = 0
        prev_length = 0
        for c in self._delivered:
            expected = prev_offset + prev_length
            if c.pcm_offset_bytes != expected:
                gaps_in_pcm.append((prev_offset, expected, c.pcm_offset_bytes))
            prev_offset = c.pcm_offset_bytes
            prev_length = c.payload_length
        lines.append(f"PCM-offset gaps in delivered-only list: {len(gaps_in_pcm)}")
        lines.append(f"  (with silence inserter OFF, should be 0;")
        lines.append(f"   with silence inserter ON, should equal lost_packets × 144)")
        for g in gaps_in_pcm[:5]:
            lines.append(f"  prev_end={g[1]} next_start={g[2]} delta={g[2]-g[1]}")
        with open(SUMMARY_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        log.info("Sequence summary written to %s", SUMMARY_PATH)
