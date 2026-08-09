"""End-to-end deterministic reference test.

Test A — Direct PipeWire
    Inject /tmp/deterministic-source.wav PCM bytes directly into
    Phone_Microphone via pw-cat -p and record from Phone_Microphone_Input
    via pw-cat -r.  Save to /tmp/direct-deterministic.wav.

Test B — UDP path
    Take the exact same PCM bytes, feed them through the existing
    transport.AudioSender -> transport.AudioReceiver -> injector.write_frames
    using the same config as production (loopback UDP), and record from
    Phone_Microphone_Input.  Save to /tmp/udp-deterministic.wav.

Compare both recordings against the exact same source WAV with
sample-level metrics (frame counts, offset, MAE, RMSE, max, exact %,
correlation, bit-exact).

Does NOT modify:
    - Android code, AudioRecord, microphone configuration
    - PipeWire configuration
    - UDP packet format, packet size, jitter buffer
    - production Injector

This script reads /tmp/deterministic-source.wav (built by
diag_make_deterministic.py).  Both tests reuse the same raw PCM bytes,
so the comparison is fair.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from config import BIND_HOST, BIND_PORT, DEST_HOST, DEST_PORT
from transport.audio_session import AudioSession
from diag_compare_pcm import (
    SR, CHANNELS, read_wav_raw_pcm, compare_pcm,
)

SOURCE = "/tmp/deterministic-source.wav"
OUT_DIRECT = "/tmp/direct-deterministic.wav"
OUT_UDP = "/tmp/udp-deterministic.wav"

SEGMENTS = [
    (0.0, 1.0, 440.0),
    (1.5, 2.5, 880.0),
    (3.0, 4.0, 1000.0),
]


def pw_cat_record(out_path: str, source: str = "Phone_Microphone_Input") -> subprocess.Popen:
    cmd = [
        "pw-cat", "-r",
        "--target", source,
        "--format", "f32",
        "--channels", str(CHANNELS),
        "--rate", str(SR),
        out_path,
    ]
    return subprocess.Popen(cmd, stderr=subprocess.DEVNULL)


def pw_cat_play(sink: str = "Phone_Microphone") -> subprocess.Popen:
    cmd = [
        "pw-cat", "-p",
        "--target", sink,
        "--format", "f32",
        "--channels", str(CHANNELS),
        "--rate", str(SR),
        "-",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def stream_pcm(proc: subprocess.Popen, pcm: bytes, real_time: bool = True) -> None:
    chunk_size = SR * 4 * CHANNELS // 10  # 100 ms
    written = 0
    t0 = time.time()
    while written < len(pcm):
        n = min(chunk_size, len(pcm) - written)
        try:
            proc.stdin.write(pcm[written:written + n])
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            break
        written += n
        if real_time:
            target = written / (SR * 4 * CHANNELS)
            elapsed = time.time() - t0
            if target - elapsed > 0.05:
                time.sleep(target - elapsed)
    try:
        proc.stdin.close()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=2)


def test_a_direct(pcm: bytes) -> None:
    print("=" * 72)
    print("TEST A — Direct PipeWire (pw-cat -p -> sink -> monitor -> source)")
    print("=" * 72)
    rec = pw_cat_record(OUT_DIRECT)
    time.sleep(0.3)
    pb = pw_cat_play()
    time.sleep(0.2)
    stream_pcm(pb, pcm)
    time.sleep(1.0)
    rec.terminate()
    try:
        rec.wait(timeout=2)
    except subprocess.TimeoutExpired:
        rec.kill()
    print(f"recorded -> {OUT_DIRECT}\n")


def test_b_udp(pcm: bytes) -> None:
    print("=" * 72)
    print(f"TEST B — UDP transport path ({BIND_HOST}:{BIND_PORT} -> "
          f"{DEST_HOST}:{DEST_PORT})")
    print("=" * 72)

    # 1) Bring up virtual sinks (Phone_Microphone, Phone_Microphone_Input).
    #    Production VirtualAudioManager creates these. We import it but do
    #    NOT modify it.
    from audio.virtual_audio import VirtualAudioManager
    vam = VirtualAudioManager()
    vam.start()
    print(f"virtual sink: {vam.sink_name()}  source: {vam.source_name()}")

    # 2) Bring up the injector (production pw-cat -p into Phone_Microphone).
    from audio.injector import Injector
    injector = Injector()
    injector.start_injection(tone=False)

    # 3) Bring up the UDP session, wiring its receiver into the injector.
    #    For this loopback diagnostic the sender and receiver must talk on
    #    the same port — production config separates bind/dest because the
    #    endpoints are on different machines. We use BIND_PORT for both.
    session = AudioSession(
        bind_host=BIND_HOST,
        bind_port=BIND_PORT,
        dest_host=DEST_HOST,
        dest_port=BIND_PORT,  # loopback: same port for sender + receiver
    )
    session.start()
    session.bind_injector(injector.write_frames)

    # 4) Start a recorder from the source.
    rec = pw_cat_record(OUT_UDP)
    time.sleep(0.3)

    # 5) Push the same deterministic PCM bytes through the sender.
    #    The sender fragments them, the receiver reassembles, and the
    #    injector writes the reconstructed PCM into Phone_Microphone.
    #    The AudioSender queue is 32 deep and ~100 ms chunks will fill it
    #    if we submit faster than the network drains. Pace submissions so
    #    each chunk is queued only after the previous one is gone.
    chunk_size = SR * 4 * CHANNELS // 10  # 100 ms chunks, like capture
    chunk_secs = chunk_size / (SR * 4 * CHANNELS)
    print(f"submitting {len(pcm)} PCM bytes through AudioSender in "
          f"{chunk_size}-byte chunks at ~{chunk_secs*1000:.0f}ms cadence...")
    t0 = time.time()
    submitted_bytes = 0
    while submitted_bytes < len(pcm):
        # Wait until the queue has at least 1 slot free.
        qsize = session.sender._queue.qsize()
        # Submit at most one chunk per loop; sleep to let drain.
        chunk = pcm[submitted_bytes:submitted_bytes + chunk_size]
        session.sender.submit(chunk)
        submitted_bytes += len(chunk)
        # Small sleep so we don't spin and so the receiver has time to drain.
        time.sleep(chunk_secs * 0.5)
    elapsed = time.time() - t0
    print(f"submitted in {elapsed:.2f}s")

    # 6) Wait for everything to drain.
    deadline = time.time() + 12.0
    while time.time() < deadline:
        s = session.stats()
        sent = s["sender"]["pcm_bytes_submitted"]
        recv = s["receiver"]["pcm_bytes_delivered"]
        if recv >= sent and sent > 0:
            break
        time.sleep(0.2)
    s = session.stats()
    print(f"sender: {s['sender']}")
    print(f"receiver: {s['receiver']}")
    print(f"session: {s['session']}")

    # 7) Stop recorder cleanly.
    time.sleep(1.0)
    rec.terminate()
    try:
        rec.wait(timeout=2)
    except subprocess.TimeoutExpired:
        rec.kill()

    # 8) Tear everything down.
    session.stop()
    injector.stop()
    vam.stop()
    print(f"recorded -> {OUT_UDP}\n")


def main() -> int:
    info, pcm = read_wav_raw_pcm(SOURCE)
    print(f"source: {SOURCE}")
    print(f"  frames={info['frames']}  bytes={len(pcm)}  duration={info['frames']/SR:.3f}s")
    print(f"  format={info['audio_format']}  sr={info['sample_rate']}  "
          f"ch={info['channels']}  bps={info['bits_per_sample']}\n")

    # Make sure virtual sink + source are up before Test A.
    # We start them here so Test A also has them; they survive both tests.
    from audio.virtual_audio import VirtualAudioManager
    vam = VirtualAudioManager()
    vam.start()
    print(f"virtual sink: {vam.sink_name()}  source: {vam.source_name()}\n")
    keep_vam = True

    try:
        # Test A
        test_a_direct(pcm)
        # Test B (creates its own session/injector but reuses the virtual sink)
        test_b_udp(pcm)
    finally:
        if keep_vam:
            vam.stop()

    # Compare
    print("=" * 72)
    print("COMPARISON A — source vs /tmp/direct-deterministic.wav")
    print("=" * 72)
    if os.path.exists(OUT_DIRECT):
        _, out_a = read_wav_raw_pcm(OUT_DIRECT)
        compare_pcm(pcm, out_a, segments=SEGMENTS,
                    src_path=SOURCE, out_path=OUT_DIRECT)
    else:
        print(f"missing {OUT_DIRECT}")

    print("\n" + "=" * 72)
    print("COMPARISON B — source vs /tmp/udp-deterministic.wav")
    print("=" * 72)
    if os.path.exists(OUT_UDP):
        _, out_b = read_wav_raw_pcm(OUT_UDP)
        compare_pcm(pcm, out_b, segments=SEGMENTS,
                    src_path=SOURCE, out_path=OUT_UDP)
    else:
        print(f"missing {OUT_UDP}")

    return 0


if __name__ == "__main__":
    sys.exit(main())