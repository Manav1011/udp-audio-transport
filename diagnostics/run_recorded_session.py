"""Integration harness: simulate a realistic ~8s Android-style session
directly through JitterBuffer + SequenceIndexedPcmRecorder.

Also writes the actual delivered PCM bytes to /tmp/backend_received.wav
so the comparison report has both files.
"""
import math
import os
import random
import struct
import sys
import tempfile
import time

os.environ["AUDIO_DIAGNOSTIC_SEQUENCE"] = "1"

import numpy as np

from transport.audio_receiver import JitterBuffer
from transport.audio_sequence_recorder import SequenceIndexedPcmRecorder

BACKEND_WAV = "/tmp/backend_received.wav"


def write_wav_float32(path: str, sample_rate: int, channels: int, raw_bytes: bytes) -> None:
    """Write a minimal IEEE-float (format=3) WAV file."""
    bits = 32
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    fmt_chunk = struct.pack(
        "<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, bits
    )
    data_size = len(raw_bytes)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + data_size)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw_bytes)


def make_deterministic_pcm(duration_s=8.0, sample_rate=48000, freq=440.0):
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    wave = 0.5 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    return stereo.tobytes()


def main():
    from transport import audio_sequence_recorder as mod
    tmp = tempfile.mkdtemp(prefix="seq_rec_")
    json_path = os.path.join(tmp, "seq.json")
    summary_path = os.path.join(tmp, "seq.txt")
    print(f"Output JSON:    {json_path}")
    print(f"Output summary: {summary_path}")
    print(f"Backend WAV:    {BACKEND_WAV}")
    orig_json = mod.JSON_PATH
    orig_summary = mod.SUMMARY_PATH
    mod.JSON_PATH = json_path
    mod.SUMMARY_PATH = summary_path

    random.seed(42)
    DROP_PCT = 0.015
    REORDER_WINDOW = 8
    INTER_ARRIVAL_S = 0.006

    pcm = make_deterministic_pcm(duration_s=8.0)
    payloads = [pcm[i:i + 1152] for i in range(0, len(pcm), 1152)]
    n_total = len(payloads)
    print(f"Total packets to send: {n_total}")

    reordered = []
    i = 0
    while i < len(payloads):
        window = payloads[i:i + REORDER_WINDOW]
        if random.random() < 0.5 and len(window) >= 2:
            random.shuffle(window)
        reordered.extend(window)
        i += REORDER_WINDOW

    arriving_seqs = []
    seq = 0
    for _payload in reordered:
        if random.random() >= DROP_PCT:
            arriving_seqs.append(seq)
        seq += 1

    n_arriving = len(arriving_seqs)
    print(f"Packets arriving: {n_arriving} ({n_total - n_arriving} dropped)")

    payload_by_seq = {seq: payloads[seq] for seq in arriving_seqs}

    buf = JitterBuffer(reorder_window_ms=200)
    rec = SequenceIndexedPcmRecorder()
    rec.start()

    # Also write the same PCM bytes that the recorder sees to backend_received.wav
    # so we have a real file to compare against.
    # We must apply jitter's release order — releases are NOT in seq order if
    # there was reordering, but in this simulation the jitter releases in seq
    # order (reorder window=200ms > 6ms × 8 packets = 48ms, so all reorders
    # close within the window). After losses, releases follow the seq order
    # with skipped seqs (those slots are NOT in the output).

    sim_time = [10000.0]
    import transport.audio_receiver as rx_mod
    real_monotonic = time.monotonic
    rx_mod.time.monotonic = lambda: sim_time[0]

    backend_pcm_chunks: list[bytes] = []

    try:
        for seq in arriving_seqs:
            sim_time[0] += INTER_ARRIVAL_S
            out = buf.push(seq, payload_by_seq[seq])
            if out:
                seqs = list(buf._last_released_seqs)
                rec.record_delivered(seqs, [len(payloads[s]) for s in seqs], sim_time[0])
                backend_pcm_chunks.extend(out)
            while True:
                ready = buf.tick(now=sim_time[0])
                if not ready:
                    break
                lost_seq = buf._last_lost_seq
                if lost_seq is not None:
                    rec.record_lost(lost_seq, sim_time[0])
                    buf._last_lost_seq = None
                seqs = list(buf._last_released_seqs)
                rec.record_delivered(seqs, [len(payloads[s]) for s in seqs], sim_time[0])
                backend_pcm_chunks.extend(ready)

        sim_time[0] += 1.0
        while True:
            ready = buf.tick(now=sim_time[0])
            if not ready:
                break
            lost_seq = buf._last_lost_seq
            if lost_seq is not None:
                rec.record_lost(lost_seq, sim_time[0])
                buf._last_lost_seq = None
            seqs = list(buf._last_released_seqs)
            rec.record_delivered(seqs, [len(payloads[s]) for s in seqs], sim_time[0])
            backend_pcm_chunks.extend(ready)
    finally:
        rx_mod.time.monotonic = real_monotonic
        rec.stop()

    # Write backend_received.wav
    backend_pcm = b"".join(backend_pcm_chunks)
    write_wav_float32(BACKEND_WAV, 48000, 2, backend_pcm)

    # Also write the ORIGINAL phone_recorded.wav from the same deterministic
    # generator, so the report can compare them.
    phone_path = "/home/manav1011/Documents/udp-audio-transport/phone_recorded.wav"
    # Only overwrite if it doesn't exist or if user opts in
    if "--regenerate-phone" in sys.argv or not os.path.exists(phone_path):
        write_wav_float32(phone_path, 48000, 2, pcm)
        print(f"Wrote phone recording to {phone_path}")
    else:
        print(f"Keeping existing phone recording at {phone_path}")

    # Stats
    print()
    print(f"Recorder delivered entries: {len(rec._delivered)}")
    print(f"Recorder lost entries:      {len(rec._lost)}")
    if rec._delivered:
        first = rec._delivered[0]
        last = rec._delivered[-1]
        print(f"First delivered seq: {first.seq} @ PCM frame {first.pcm_offset_frames}")
        print(f"Last delivered seq:  {last.seq} @ PCM frame {last.pcm_offset_frames}")
        lens = sorted({c.payload_length for c in rec._delivered})
        print(f"Distinct payload lengths: {lens}")
    print(f"backend PCM bytes: {len(backend_pcm)}")

    # Copy recorder outputs to /tmp canonical paths
    import shutil
    shutil.copy(json_path, "/tmp/audio_sequence_log.json")
    shutil.copy(summary_path, "/tmp/audio_sequence_summary.txt")
    print(f"Copied to /tmp/audio_sequence_log.json and /tmp/audio_sequence_summary.txt")

    # Print summary
    print()
    print("=" * 60)
    print(" Recorder summary:")
    print("=" * 60)
    with open(summary_path) as f:
        print(f.read())

    mod.JSON_PATH = orig_json
    mod.SUMMARY_PATH = orig_summary


if __name__ == "__main__":
    main()