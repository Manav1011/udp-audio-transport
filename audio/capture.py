"""System audio capture — records PCM from the PipeWire default output monitor.

Uses pw-cat in record mode to capture the default sink's monitor source.
All pw-cat interaction lives here.
"""
from __future__ import annotations

import select
import signal
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(line_buffering=True)


class Capture:
    """Captures PCM bytes from the default PipeWire output monitor.

    Exposes start_capture(callback) — callback receives raw PCM bytes.
    """

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._sample_rate = 48000
        self._channels = 2
        self._frame_bytes = 4

    # -- device discovery --------------------------------------------------

    def _get_monitor(self) -> tuple[str, str] | None:
        """Get (sink_name, monitor_source) for the default output sink.

        Never selects a virtual microphone monitor.
        """
        # Query all sinks
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                sink_name = parts[1]
                # Skip virtual microphones
                if "mic" in sink_name.lower() or "microphone" in sink_name.lower():
                    continue
                monitor = f"{sink_name}.monitor"
                # Verify monitor exists by trying to query it
                try:
                    probe = subprocess.run(
                        ["pactl", "list", "sources", "short"],
                        capture_output=True, text=True, check=True,
                    )
                    if monitor in probe.stdout:
                        return sink_name, monitor
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
                # If we can't verify, just use it anyway
                return sink_name, monitor

        return None

    # -- lifecycle ---------------------------------------------------------

    def start_capture(self, callback):
        """Start capturing audio. callback(bytes) is invoked with each chunk."""
        monitor_info = self._get_monitor()
        if monitor_info is None:
            print("Could not determine default sink monitor")
            return
        sink, monitor = monitor_info
        print(f"Capturing: {monitor}")

        cmd = [
            "pw-cat", "-r",
            "--target", monitor,
            "--format", "f32",
            "--channels", str(self._channels),
            "--rate", str(self._sample_rate),
            "-",  # write to stdout
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("Error: pw-cat not found. Install pipewire-utils.")
            return

        chunk_size = self._sample_rate // 10 * self._channels * self._frame_bytes  # 100ms

        with self._lock:
            self._proc = proc

        print("------------------------------------")
        print(f"Device:      {monitor}")
        print(f"Sample Rate: {self._sample_rate}")
        print(f"Channels:    {self._channels}")
        print("------------------------------------\n")

        total_bytes = 0
        start_time = time.time()
        last_log_time = start_time
        frames_per_sec = 0
        bytes_per_sec = 0

        try:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    data = proc.stdout.read(chunk_size)
                    if not data:
                        break
                    total_bytes += len(data)
                    if callback:
                        callback(data)

                # Log stats every second
                now = time.time()
                if now - last_log_time >= 1.0:
                    if total_bytes > 0:
                        elapsed = now - last_log_time
                        bytes_per_sec = total_bytes / elapsed
                        frames_per_sec = int(bytes_per_sec / (self._channels * self._frame_bytes))
                        print("------------------------------------")
                        print(f"Frames/sec: {frames_per_sec}")
                        print(f"Bytes/sec:  {bytes_per_sec}")
                        print("------------------------------------\n")
                    total_bytes = 0
                    last_log_time = now
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._stop(proc)

    def _stop(self, proc):
        """Terminate the pw-cat process."""
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Stream closed.")

    def stop(self):
        """Stop any active capture."""
        with self._lock:
            proc = self._proc
        if proc is not None:
            self._stop(proc)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))
    print("Starting capture...")
    capture = Capture()
    capture.start_capture(callback=None)
