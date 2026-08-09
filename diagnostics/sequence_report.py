"""Generate a sequence-indexed report from /tmp/audio_sequence_log.json.

Answers the user's questions:
  - total PCM frames expected vs received
  - exact missing seqs (gaps in delivered seqs)
  - frame ranges for missing packets (PCM-frame positions the lost
    chunks WOULD have occupied, had they been delivered)
  - whether backend PCM shifts after missing packets (alignment)
  - whether 1152-byte payload = 144 stereo frames
  - whether final PCM is Android PCM with missing ranges removed (or
    samples changed by resampling / pitch-shift)

Inputs:
  - /tmp/audio_sequence_log.json (sequence recorder output)
  - /home/manav1011/Documents/udp-audio-transport/phone_recorded.wav
  - /tmp/backend_received.wav

Outputs a human-readable report to stdout.
"""
from __future__ import annotations

import json
import os
import struct
import sys

JSON_PATH = "/tmp/audio_sequence_log.json"
PHONE_WAV = "/home/manav1011/Documents/udp-audio-transport/phone_recorded.wav"
BACKEND_WAV = "/tmp/backend_received.wav"
STEREO_FRAME_BYTES = 8
EXPECTED_PAYLOAD = 1152
EXPECTED_FRAMES_PER_PAYLOAD = 144  # 1152 / 8


def _read_wav_data(path: str) -> tuple[int, int, int, int, bytes]:
    """Read WAV header + raw PCM bytes. Returns (channels, sample_rate,
    bits_per_sample, total_frames, raw_bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", f"{path}: not RIFF/WAVE"
    i = 12
    channels = sample_rate = bits = audio_format = 0
    audio = b""
    while i < len(data):
        cid = data[i:i+4]
        csz = struct.unpack("<I", data[i+4:i+8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[i+8:i+8+csz])
            audio_format = fmt[0]
            channels = fmt[1]
            sample_rate = fmt[2]
            bits = fmt[5]
        elif cid == b"data":
            audio = data[i+8:i+8+csz]
            break
        i += 8 + csz
    total_frames = len(audio) // STEREO_FRAME_BYTES
    return channels, sample_rate, bits, total_frames, audio


def _gaps_in_seq(seqs: list[int]) -> list[tuple[int, int]]:
    """Return a list of (missing_seq, next_present_seq) for each gap in
    the monotonic sequence. Each tuple says 'missing X is followed by Y'.
    """
    out = []
    if not seqs:
        return out
    for a, b in zip(seqs, seqs[1:]):
        if b > a + 1:
            for missing in range(a + 1, b):
                out.append((missing, b))
    return out


def main():
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: {JSON_PATH} not found.")
        print("Run a session with AUDIO_DIAGNOSTIC_SEQUENCE=1 first.")
        sys.exit(1)
    with open(JSON_PATH) as f:
        log = json.load(f)
    delivered = log.get("delivered", [])
    lost = log.get("lost", [])

    print("=" * 72)
    print(" Sequence-Indexed Report — phone_recorded.wav vs backend_received.wav")
    print("=" * 72)
    print()

    # ------------------------------------------------------------
    # Q1: total PCM frames expected vs received
    # ------------------------------------------------------------
    deliv_seqs = [d["seq"] for d in delivered]
    total_delivered_bytes = sum(d["payload_length"] for d in delivered)
    total_delivered_frames = total_delivered_bytes // STEREO_FRAME_BYTES
    n_lost = len(lost)
    # We assume sender started at seq 0 and emits one packet per 6ms.
    # If we have seqs 0..N, that's N+1 packets attempted; -losses = received.
    if deliv_seqs:
        first_seq = deliv_seqs[0]
        last_seq = deliv_seqs[-1]
        attempted = (last_seq - first_seq + 1)
        lost_seqs = sorted(l["seq"] for l in lost)
    else:
        attempted = 0
        first_seq = last_seq = 0
        lost_seqs = []

    print(f"First delivered seq:    {first_seq}")
    print(f"Last delivered seq:     {last_seq}")
    print(f"Seqs attempted:         {attempted}  (= last_seq - first_seq + 1)")
    print(f"Seqs delivered:         {len(delivered)}")
    print(f"Seqs declared lost:     {n_lost}")
    print(f"Total PCM bytes rec'd:  {total_delivered_bytes}")
    print(f"Total PCM frames rec'd: {total_delivered_frames}")
    print()

    # ------------------------------------------------------------
    # Q2: exact missing seqs + their frame ranges
    # ------------------------------------------------------------
    print("-" * 72)
    print(" Missing sequence numbers and frame ranges")
    print("-" * 72)
    if not lost_seqs:
        print("  (none recorded)")
    else:
        for ls in lost_seqs:
            # Find the surrounding delivered chunks: those with seq < ls
            # contribute frames before; those with seq > ls contribute after.
            preceding = [d for d in delivered if d["seq"] < ls]
            following = [d for d in delivered if d["seq"] > ls]
            frame_start = preceding[-1]["pcm_offset_frames"] + (
                preceding[-1]["payload_length"] // STEREO_FRAME_BYTES
            ) if preceding else 0
            # 144 frames per missing packet (1152/8)
            frame_end = frame_start + EXPECTED_FRAMES_PER_PAYLOAD - 1
            print(f"  lost seq={ls}: PCM frames [{frame_start}..{frame_end}] "
                  f"(144 frames / 6ms @ 48kHz would have lived here)")
            if following:
                next_first_frame = following[0]["pcm_offset_frames"]
                gap_in_recording_frames = next_first_frame - frame_end - 1
                if gap_in_recording_frames != 0:
                    print(f"    -> backend PCM shifted by {gap_in_recording_frames} frames after this loss")
                else:
                    print(f"    -> backend PCM is contiguous (no shift after this loss)")
    print()

    # ------------------------------------------------------------
    # Q3: gaps in delivered seqs (more detailed than 'lost' list)
    # ------------------------------------------------------------
    print("-" * 72)
    print(" Gaps detected in delivered seq stream (out-of-order gaps NOT covered by timeout)")
    print("-" * 72)
    gaps = _gaps_in_seq(deliv_seqs)
    if not gaps:
        print("  (none — every delivered seq was contiguous to its predecessor)")
    else:
        for missing, next_present in gaps[:50]:
            print(f"  seq {missing} missing (followed by seq {next_present})")
        if len(gaps) > 50:
            print(f"  ... and {len(gaps) - 50} more gaps")
    print()

    # ------------------------------------------------------------
    # Q4: 1152-byte payload = 144 stereo frames?
    # ------------------------------------------------------------
    print("-" * 72)
    print(" Payload length → frame count check")
    print("-" * 72)
    lens = sorted({d["payload_length"] for d in delivered})
    print(f"  distinct payload lengths in delivered chunks: {lens}")
    if lens == [EXPECTED_PAYLOAD]:
        print(f"  ✓ all delivered packets are exactly {EXPECTED_PAYLOAD} bytes")
        print(f"  ✓ {EXPECTED_PAYLOAD} / 8 bytes-per-frame = "
              f"{EXPECTED_PAYLOAD // STEREO_FRAME_BYTES} stereo frames")
    else:
        for L in lens:
            if L % STEREO_FRAME_BYTES != 0:
                print(f"  ✗ payload length {L} is NOT divisible by 8 — CORRUPT")
            else:
                print(f"  ? payload length {L} → {L // STEREO_FRAME_BYTES} frames "
                      f"(expected {EXPECTED_FRAMES_PER_PAYLOAD})")
    print()

    # ------------------------------------------------------------
    # Q5: WAV analysis (phone vs backend) — alignment & sample equality
    # ------------------------------------------------------------
    print("-" * 72)
    print(" WAV file comparison")
    print("-" * 72)
    if os.path.exists(PHONE_WAV):
        pc, psr, pbits, pframes, praw = _read_wav_data(PHONE_WAV)
        print(f"  phone_recorded.wav:  {pframes} frames @ {psr}Hz × "
              f"{pc}ch × {pbits}bit = {pframes/psr:.3f}s")
    else:
        pc = psr = pbits = pframes = 0
        praw = b""
        print(f"  phone_recorded.wav:  (missing — {PHONE_WAV})")
    if os.path.exists(BACKEND_WAV):
        bc, bsr, bbits, bframes, braw = _read_wav_data(BACKEND_WAV)
        print(f"  backend_received.wav:{bframes} frames @ {bsr}Hz × "
              f"{bc}ch × {bbits}bit = {bframes/bsr:.3f}s")
    else:
        bc = bsr = bbits = bframes = 0
        braw = b""
        print(f"  backend_received.wav:(missing — {BACKEND_WAV})")
    if pc and bc and praw and braw:
        frame_delta = pframes - bframes
        print(f"  frame difference (phone - backend): {frame_delta} frames "
              f"({frame_delta/psr*1000:.1f} ms)")
    print()

    # ------------------------------------------------------------
    # Q6: Backend PCM-shift detection: after each loss, did the
    #     backend PCM continue at the same offset or skip?
    # ------------------------------------------------------------
    print("-" * 72)
    print(" Backend PCM alignment after missing packets")
    print("-" * 72)
    # Expected offset of seq N = (N - first_seq) * 1152
    # Actual offset = delivered's pcm_offset_bytes
    if deliv_seqs:
        for d in delivered[:5] + delivered[-5:]:
            expected = (d["seq"] - first_seq) * EXPECTED_PAYLOAD
            actual = d["pcm_offset_bytes"]
            drift = actual - expected
            if abs(drift) > 0:
                print(f"  seq {d['seq']}: actual={actual} expected={expected} "
                      f"drift={drift:+d} bytes "
                      f"({'PCM SHIFTED after missing packets' if drift != 0 else 'aligned'})")
        # Aggregate drift
        last = delivered[-1]
        expected_total = (last["seq"] - first_seq) * EXPECTED_PAYLOAD
        actual_total = last["pcm_offset_bytes"] + last["payload_length"]
        drift_total = actual_total - expected_total
        drift_frames = drift_total // STEREO_FRAME_BYTES
        print(f"  total PCM bytes vs seq-scaled expectation: "
              f"actual={actual_total} expected={expected_total} "
              f"drift={drift_total:+d} bytes "
              f"({drift_frames:+d} frames)")
    print()

    # ------------------------------------------------------------
    # Q7: Sample-level comparison: does backend PCM == phone PCM with
    #     missing-frame ranges removed (zero-padded or skipped)?
    # ------------------------------------------------------------
    print("-" * 72)
    print(" Sample-level comparison (phone vs backend, treating missing as skip)")
    print("-" * 72)
    if not (praw and braw and pc and bc):
        print("  (skipped — input WAVs missing)")
    else:
        # Build a "reconstructed" phone PCM by removing the byte ranges
        # corresponding to lost seqs.
        lost_byte_ranges = []
        cursor_bytes = 0  # running offset in expected PCM
        cursor_seq = first_seq
        all_seqs = sorted(set(deliv_seqs) | set(l["seq"] for l in lost))
        # Walk through all seqs in order. Each seq occupies 1152 bytes.
        reconstructed = bytearray()
        for s in all_seqs:
            byte_start = (s - first_seq) * EXPECTED_PAYLOAD
            byte_end = byte_start + EXPECTED_PAYLOAD
            if s in {l["seq"] for l in lost}:
                # Lost — skip
                continue
            # Delivered — slice from phone PCM
            if byte_end <= len(praw):
                reconstructed.extend(praw[byte_start:byte_end])
        # Now compare reconstructed vs backend
        n_cmp = min(len(reconstructed), len(braw))
        if n_cmp == 0:
            print("  (no comparable bytes)")
        else:
            n_match = sum(
                1 for i in range(n_cmp) if reconstructed[i] == braw[i]
            )
            match_pct = n_match / n_cmp * 100
            print(f"  compared {n_cmp} bytes ({n_cmp // STEREO_FRAME_BYTES} frames)")
            print(f"  byte-exact match: {n_match} ({match_pct:.2f}%)")
            # Also report on frames (groups of 8 bytes)
            n_frames = n_cmp // STEREO_FRAME_BYTES
            n_frame_match = 0
            for fr in range(n_frames):
                base = fr * STEREO_FRAME_BYTES
                if reconstructed[base:base+STEREO_FRAME_BYTES] == braw[base:base+STEREO_FRAME_BYTES]:
                    n_frame_match += 1
            print(f"  frame-exact match: {n_frame_match}/{n_frames} "
                  f"({n_frame_match/n_frames*100:.2f}%)")
            if match_pct == 100.0:
                print("  ✓ backend PCM is EXACTLY phone PCM with missing-frame ranges removed")
            elif match_pct >= 99.0:
                print("  ≈ backend PCM is ~phone PCM with missing-frame ranges removed (small drift)")
            else:
                print("  ✗ backend PCM is NOT phone PCM with missing-frame ranges removed")
                print("    (samples differ — likely resampling, mic-vs-speaker mismatch, or padding)")

    print()
    print("=" * 72)
    print(" END OF REPORT")
    print("=" * 72)


if __name__ == "__main__":
    main()