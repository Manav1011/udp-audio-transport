"""Standalone raw TCP receiver for Android microphone PCM.

This is a DELIBERATELY MINIMAL diagnostic. It exists to test one
specific hypothesis:

    Is the microphone noise caused by our UDP transport path?

The Android microphone PCM is already known to be clean when saved
locally on Android. The current raw UDP path produces noisy PCM on the
PC. To isolate the transport, this script receives the same Android
capture output via TCP — a byte stream — and writes the EXACT bytes
to disk.

The receiver has NOTHING to do with the production audio pipeline:

    - no AudioSession
    - no AudioReceiver
    - no JitterBuffer
    - no PipeWire
    - no Injector
    - no UDP
    - no AudioRecord
    - no sequence numbers
    - no headers
    - no resampling
    - no reformatting
    - no DSP of any kind

It literally does:

    data = conn.recv(...)
    output.write(data)

until EOF. That's the entire pipeline.

OUTPUT FORMAT
=============

The on-the-wire format is identical to the UDP path's transport
format (the Android sender does the same PCM conversion for both
transports):

    48000 Hz / stereo / Float32 LE / 8 bytes per frame

The receiver does NOT validate this format. It writes whatever bytes
arrive. After capture, the file's `bytes % 8` is reported so the
operator can sanity-check that the Android sender produced complete
frames, but the file is never modified.

START
=====

    python -m diagnostics.raw_android_tcp_receiver

Defaults:
    bind:  0.0.0.0
    port:  5002
    output: /tmp/android_raw_tcp.pcm

PLAYBACK
========

    ffplay -f f32le -ar 48000 -ac 2 /tmp/android_raw_tcp.pcm

ANALYSIS
========

    python -m diagnostics.analyze_raw_pcm --path /tmp/android_raw_tcp.pcm

COMPARISON
==========

Compare against the UDP capture:

    /tmp/android_raw_udp.pcm   (recorded via the raw UDP receiver)
    /tmp/android_raw_tcp.pcm   (recorded via this TCP receiver)

If the Android capture is the same in both cases, the only
difference is the transport. UDP-noisy + TCP-clean → UDP transport is
implicated. UDP-noisy + TCP-noisy → the issue is upstream of UDP.
"""
from __future__ import annotations

import argparse
import hashlib
import signal
import socket
import sys
import threading
import time

DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 5002
DEFAULT_OUTPUT = "/tmp/android_raw_tcp.pcm"

# How many bytes to recv per call. 64 KiB balances throughput and
# reporting granularity. The receiver is byte-faithful regardless of
# this value.
_RECV_CHUNK = 65536


def _format_int(n: int) -> str:
    return f"{n:,}"


def _serve_one_connection(
    sock: socket.socket,
    output_path: str,
    print_lock: threading.Lock,
) -> tuple[int, int, str, float]:
    """Accept one connection, stream its bytes to the output file
    until EOF, and return (bytes_received, recv_calls, sha256,
    duration_s)."""
    print_with_lock = lambda msg: print(msg, flush=True)  # noqa: E731
    with print_lock:
        print_with_lock(f"  waiting for connection on {sock.getsockname()}...")
    # block until a client connects
    conn, addr = sock.accept()
    started_at = time.monotonic()
    print_with_lock(f"  connection accepted from {addr[0]}:{addr[1]}")
    print_with_lock(f"  writing to {output_path}")
    sha = hashlib.sha256()
    bytes_received = 0
    recv_calls = 0
    # Open in binary mode, write only, unbuffered so a Ctrl+C still
    # leaves a usable file on disk.
    with open(output_path, "wb", buffering=0) as out, conn:
        conn.settimeout(None)
        while True:
            data = conn.recv(_RECV_CHUNK)
            recv_calls += 1
            if not data:
                # EOF: client closed the connection cleanly.
                break
            out.write(data)
            sha.update(data)
            bytes_received += len(data)
    duration = time.monotonic() - started_at
    return bytes_received, recv_calls, sha.hexdigest(), duration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone raw TCP receiver for Android microphone PCM"
    )
    parser.add_argument(
        "--bind", default=DEFAULT_BIND,
        help=f"Bind host (default {DEFAULT_BIND})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"TCP port to listen on (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output PCM file path (default {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("RAW ANDROID TCP RECEIVER")
    print(f"  bind:    {args.bind}:{args.port}")
    print(f"  output:  {args.output}")
    print(f"  protocol: raw TCP byte stream (no framing, no headers)")
    print(f"  press Ctrl+C to stop")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.listen(1)
    sock.settimeout(None)

    stop = threading.Event()

    def _signal_handler(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print_lock = threading.Lock()

    try:
        # Default mode: serve one connection, then exit.
        # Set a short accept timeout so Ctrl+C is responsive.
        sock.settimeout(0.5)
        result = None
        while not stop.is_set() and result is None:
            try:
                result = _serve_one_connection(
                    sock, args.output, print_lock
                )
            except socket.timeout:
                continue
            except OSError as e:
                if stop.is_set():
                    break
                print(f"  accept error: {e}", flush=True)
                continue
    finally:
        sock.close()

    if result is None:
        # No connection ever happened.
        print()
        print("=" * 52)
        print("RAW ANDROID TCP RECEIVER")
        print("=" * 52)
        print("  no connection received")
        print("=" * 52)
        return 0

    bytes_received, recv_calls, sha256_hex, duration = result

    # Final report
    print()
    print("=" * 52)
    print("RAW ANDROID TCP RECEIVER")
    print("=" * 52)
    print(f"connection time:   {duration:.3f} s")
    print(f"bytes received:    {_format_int(bytes_received)}")
    print(f"recv calls:        {_format_int(recv_calls)}")
    print(f"SHA256:            {sha256_hex}")
    print(f"output:            {args.output}")
    # Operator hint: the on-the-wire format is 8 bytes per stereo
    # frame. The receiver does not enforce this; we just print the
    # remainder so the operator can sanity-check.
    rem = bytes_received % 8
    print(f"bytes % 8:         {rem}  "
          f"({'frame-aligned' if rem == 0 else 'NOT frame-aligned'})")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    sys.exit(main())
