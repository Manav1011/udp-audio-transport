"""Standalone raw UDP receiver for Android → backend audio capture.

This script is deliberately DECOUPLED from the production backend. It
exists to answer ONE question:

    "Are the PCM bytes arriving from the Android app already noisy or
    corrupted BEFORE any receiver-side processing (jitter buffer,
    reordering, silence insertion, PipeWire) is applied?"

To answer that, this script:

    1. opens a UDP socket on a configurable port (default 5001)
    2. parses the 8-byte big-endian header (seq, payload_length)
    3. extracts the raw PCM payload
    4. appends the EXACT payload bytes to /tmp/android_raw_udp.pcm
    5. prints a 1-second status line with running statistics

It does NOT use AudioReceiver, JitterBuffer, AudioSession, Injector,
PipeWire, the silence inserter, the sequence recorder, or the
diagnostic WAV writer. It does NOT reorder packets. It writes payloads
in the exact order they arrive at the socket.

The recorded file is therefore a faithful byte-for-byte concatenation of
the PCM payloads as they crossed the UDP wire — nothing more, nothing
less. Analysis happens afterwards with diagnostics/analyze_raw_pcm.py.

PROTOCOL COMPATIBILITY

    UDP port:           5001 (overridable via --port)
    Listen address:     0.0.0.0
    Header:             8 bytes, big-endian
                          uint32 sequence_number
                          uint32 payload_length
    PCM:                48000 Hz, stereo, float32 little-endian,
                        interleaved [L, R, L, R, ...]
    Stereo frame:       8 bytes (2 channels * 4 bytes float32)
    MAX payload:        1152 bytes (144 stereo frames = 3 ms @ 48 kHz)
    Trailing partial:   allowed, frame-aligned

Header format constants are imported from transport.audio_packet so we
don't accidentally drift from the production protocol.

STRUCTURAL VALIDATION (per packet)

    1. datagram >= 8 bytes
    2. seq = uint32 big-endian
    3. payload_length = uint32 big-endian
    4. payload_length == len(datagram) - 8
    5. payload_length % 8 == 0  (frame-aligned)

If any check fails:
    - increment an error counter
    - log the problem (but do NOT spam: rate-limit to 5 per second)
    - DO NOT write that packet's payload

Do NOT attempt to repair malformed packets.

CONTROL TEST

    Run with Android Test Tone ON (440 Hz sine wave). The raw PCM file
    should be a clean 440 Hz signal with peak ~0.25 and RMS ~0.1768.
    The analyzer (diagnostics/analyze_raw_pcm.py) should report a
    dominant frequency at ~440 Hz.

    If the raw PCM is clean with Test Tone ON, the transport and
    PCM decoding are correct. The next test with the real microphone
    determines whether the defect is in Android capture.
"""
from __future__ import annotations

import argparse
import signal
import socket
import struct
import sys
import threading
import time
from collections import Counter, deque

# Reuse the production protocol constants. We deliberately do NOT
# import decode_packet (it enforces MAX_PAYLOAD which is fine, but we
# want to log the bigger picture) and we do NOT import anything else.
from transport.audio_packet import HEADER_FORMAT, HEADER_SIZE, MAX_PAYLOAD

DEFAULT_PORT = 5001
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_OUTPUT_PATH = "/tmp/android_raw_udp.pcm"
STATUS_INTERVAL_S = 1.0
ERROR_LOG_RATE_LIMIT_PER_S = 5  # don't flood stdout on a bad run

# Modular arithmetic for sequence numbers (uint32).
_SEQ_MOD = 1 << 32
_STEREO_FRAME_BYTES = 8  # 2 channels * 4 bytes float32


class _Stats:
    """Thread-safe-ish recorder statistics. Counters are plain ints; the
    UDP recv loop and the status thread both read/write them without a
    lock because all operations are atomic in CPython and eventual
    consistency is more than enough for a 1-second status line."""

    def __init__(self) -> None:
        self.packets_received = 0
        self.valid_packets = 0
        self.invalid_packets = 0
        self.payload_bytes = 0
        self.first_seq: int | None = None
        self.last_seq: int | None = None
        self.seq_seen: dict[int, int] = {}  # seq -> arrival count
        self.payload_size_hist: Counter[int] = Counter()
        # Recent payload sizes (RTT bounded) for the status line.
        self.recent_payload_sizes: deque[int] = deque(maxlen=20)
        # Errors, rate-limited.
        self.errors_short_header = 0
        self.errors_length_mismatch = 0
        self.errors_unaligned = 0
        self.errors_truncated = 0  # declared > MAX_PAYLOAD
        self.errors_other = 0
        self.start_monotonic = time.monotonic()


def _validate_and_extract(
    data: bytes, stats: _Stats
) -> tuple[int, bytes] | None:
    """Validate a single UDP datagram. Returns (seq, payload) or None.

    Does NOT raise; all errors are reported via stats counters. The
    caller decides what to do with None (we simply don't write).
    """
    if len(data) < HEADER_SIZE:
        stats.errors_short_header += 1
        return None
    seq, declared = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    if declared > MAX_PAYLOAD:
        # The sender caps at MAX_PAYLOAD but if we ever see a malformed
        # datagram with a bogus declared length, that's a corrupt packet.
        stats.errors_truncated += 1
        return None
    if len(data) - HEADER_SIZE != declared:
        stats.errors_length_mismatch += 1
        return None
    if declared % _STEREO_FRAME_BYTES != 0:
        stats.errors_unaligned += 1
        return None
    return seq, data[HEADER_SIZE:HEADER_SIZE + declared]


def _classify_arrival(
    seq: int, stats: _Stats
) -> str:
    """Update the running arrival stats and return a tag for the
    status line: 'dupe', 'gap', 'ooo' (out-of-order), or 'in-order'.

    Order is computed over the sequence of PACKETS RECEIVED, not in
    delivery order — the only ground truth here is the wire arrival
    order. The classification is for human diagnostics only and does
    NOT affect what gets written.
    """
    stats.seq_seen[seq] = stats.seq_seen.get(seq, 0) + 1
    if stats.first_seq is None:
        stats.first_seq = seq
        stats.last_seq = seq
        return "first"
    # Forward distance from last_seq to seq in the uint32 wrapped space.
    fwd = (seq - stats.last_seq) % _SEQ_MOD
    tag: str
    if fwd == 0:
        tag = "dupe"
    elif fwd == 1:
        tag = "in-order"
    elif fwd > 0x80000000:
        # Past (smaller modulo)
        tag = "ooo"
    else:
        # Future (>1 ahead)
        tag = "gap"
    stats.last_seq = seq
    return tag


def _format_status_line(stats: _Stats, elapsed_s: float) -> str:
    sizes = ", ".join(str(s) for s in stats.recent_payload_sizes)
    dupes = sum(c - 1 for c in stats.seq_seen.values() if c > 1)
    gap_count = sum(
        1 for arr in stats.seq_seen.values() if arr > 1
    )  # placeholder; refined below
    # Properly count gaps: in-order sequence numbers seen with skipped
    # numbers in between. We only consider the recent window to keep
    # this O(N) over recent seqs, not the whole session.
    recent = sorted(stats.seq_seen.keys())[-256:]
    gap_count = 0
    for i in range(1, len(recent)):
        # Treat uint32 wraparound conservatively (skip the count if wrap).
        if (recent[i] - recent[i - 1]) % _SEQ_MOD < 0x80000000:
            gap_count += int(recent[i] - recent[i - 1]) - 1
    return (
        f"[{elapsed_s:6.1f}s] "
        f"rx={stats.packets_received}  "
        f"valid={stats.valid_packets}  "
        f"invalid={stats.invalid_packets}  "
        f"bytes={stats.payload_bytes}  "
        f"last_sizes=[{sizes}]  "
        f"dupe={dupes}  gaps~{gap_count}  "
        f"errors: short={stats.errors_short_header} "
        f"mismatch={stats.errors_length_mismatch} "
        f"unaligned={stats.errors_unaligned} "
        f"truncated={stats.errors_truncated} "
        f"other={stats.errors_other}"
    )


def _print_final_report(stats: _Stats, output_path: str) -> None:
    total_seq = len(stats.seq_seen)
    dupes = sum(c - 1 for c in stats.seq_seen.values() if c > 1)
    # Calculate gaps across the full session (not just the recent window).
    gap_count = 0
    if total_seq > 1:
        seqs = sorted(stats.seq_seen.keys())
        for i in range(1, len(seqs)):
            fwd = (seqs[i] - seqs[i - 1]) % _SEQ_MOD
            if fwd < 0x80000000 and fwd > 1:
                gap_count += fwd - 1
    payload_frames = stats.payload_bytes // _STEREO_FRAME_BYTES
    print()
    print("=" * 60)
    print("RAW ANDROID UDP RECEIVER")
    print("=" * 60)
    print(f"packets received:       {stats.packets_received}")
    print(f"valid packets:          {stats.valid_packets}")
    print(f"invalid packets:        {stats.invalid_packets}")
    print(f"payload bytes:          {stats.payload_bytes}")
    print(f"payload frames:         {payload_frames}")
    print(f"first sequence:         {stats.first_seq}")
    print(f"last sequence:          {stats.last_seq}")
    print(f"distinct seqs seen:     {total_seq}")
    print(f"duplicates:             {dupes}")
    print(f"missing sequence nums:  {gap_count}")
    print(f"distinct payload sizes: {sorted(stats.payload_size_hist.keys())}")
    print(f"errors: short_header={stats.errors_short_header} "
          f"length_mismatch={stats.errors_length_mismatch} "
          f"unaligned={stats.errors_unaligned} "
          f"truncated={stats.errors_truncated} "
          f"other={stats.errors_other}")
    print("=" * 60)
    print(f"PCM saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone raw UDP receiver for Android audio"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"UDP port to listen on (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--bind", default=DEFAULT_BIND_HOST,
        help=f"Bind host (default {DEFAULT_BIND_HOST})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH,
        help=f"Output PCM file path (default {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-status", action="store_true",
        help="Disable the 1-second status line",
    )
    args = parser.parse_args()

    stats = _Stats()

    # Open the output file. We use an unbuffered binary file so each
    # write hits the disk promptly and the file is recoverable on crash.
    out = open(args.output, "wb", buffering=0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)  # 0.5s wakeup so Ctrl+C / signal thread reads

    stop = threading.Event()

    def _signal_handler(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"RAW ANDROID UDP RECEIVER")
    print(f"  bind:        {args.bind}:{args.port}")
    print(f"  output:      {args.output}")
    print(f"  protocol:    HEADER_FORMAT={HEADER_FORMAT!r} "
          f"HEADER_SIZE={HEADER_SIZE} MAX_PAYLOAD={MAX_PAYLOAD}")
    print(f"  press Ctrl+C to stop and write the final report")
    print()

    last_status = time.monotonic()
    err_log_window_start = time.monotonic()
    err_log_in_window = 0

    try:
        while not stop.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                pass
            else:
                stats.packets_received += 1
                result = _validate_and_extract(data, stats)
                if result is None:
                    # Rate-limit error logging on stdout.
                    now = time.monotonic()
                    if now - err_log_window_start > 1.0:
                        err_log_window_start = now
                        err_log_in_window = 0
                    if err_log_in_window < ERROR_LOG_RATE_LIMIT_PER_S:
                        err_log_in_window += 1
                        print(
                            f"  ! invalid packet: "
                            f"len={len(data)} "
                            f"reason counters "
                            f"short={stats.errors_short_header} "
                            f"mismatch={stats.errors_length_mismatch} "
                            f"unaligned={stats.errors_unaligned} "
                            f"truncated={stats.errors_truncated}",
                            flush=True,
                        )
                    continue
                seq, payload = result
                stats.valid_packets += 1
                stats.payload_bytes += len(payload)
                stats.payload_size_hist[len(payload)] += 1
                stats.recent_payload_sizes.append(len(payload))
                # Write the EXACT payload bytes. No modification, no
                # reorder, no decode beyond the header.
                out.write(payload)
                _classify_arrival(seq, stats)
            # Periodic status line.
            if not args.no_status:
                now = time.monotonic()
                if now - last_status >= STATUS_INTERVAL_S:
                    elapsed = now - stats.start_monotonic
                    print(_format_status_line(stats, elapsed), flush=True)
                    last_status = now
    finally:
        out.close()
        sock.close()
        _print_final_report(stats, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
