"""PCM injection — writes audio into a PipeWire virtual microphone sink.

Uses pw-cat in playback mode to write PCM into the target node.
All pw-cat interaction lives here.
"""
from __future__ import annotations

import logging
import math
import signal
import subprocess
import sys
import threading
import time

import numpy as np

log = logging.getLogger("audio-bridge")

sys.stdout.reconfigure(line_buffering=True)


class Injector:
    """Injects PCM bytes into a named PipeWire sink.

    Exposes start_injection(), write_frames(bytes), stop().
    In standalone mode, generates a continuous test tone.
    In passthrough mode (write_frames called), test tone is stopped.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 2,
                 dtype: str = "float32", freq: float = 440.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = np.dtype(dtype)
        self.freq = freq
        self._proc = None
        self._running = False
        self._passthrough = False
        self._total_samples = 0
        self._total_bytes_written = 0
        self._tone_thread = None
        self._lock = threading.Lock()


    # -- device discovery --------------------------------------------------

    def _find_virtual_mic(self) -> str | None:
        """Find the PipeWire sink for 'Phone Microphone'."""
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
                name = parts[1].lower()
                if "phone" in name and ("microphone" in name or "mic" in name):
                    return parts[1]
        return None

    # -- lifecycle ---------------------------------------------------------

    def start_injection(self, tone: bool = True):
        """Start injecting audio into the virtual microphone.

        tone: if True, also start the 440 Hz test-tone generator (used for
              standalone debugging). If False, only open the pw-cat pipe —
              real PCM is delivered via write_frames() only.
        """
        target = self._find_virtual_mic()
        if target is None:
            log.error("Could not find 'Phone Microphone' sink")
            return
        self._target = target

        cmd = [
            "pw-cat", "-p",
            "--target", target,
            "--format", "f32",
            "--channels", str(self.channels),
            "--rate", str(self.sample_rate),
            "-",
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.error("pw-cat not found. Install pipewire-utils.")
            return

        with self._lock:
            self._proc = proc
        self._running = True

        if tone:
            log.debug("Test tone active into %s (%.1f Hz, %d Hz, %d ch)",
                      target, self.freq, self.sample_rate, self.channels)
            self._tone_thread = threading.Thread(target=self._test_tone_loop, daemon=True)
            self._tone_thread.start()

    def _test_tone_loop(self):
        """Generate and write a continuous test tone. Used for standalone testing."""
        chunk_samples = self.sample_rate // 10

        try:
            while self._running and not self._passthrough:
                t = np.linspace(
                    self._total_samples / self.sample_rate,
                    (self._total_samples + chunk_samples) / self.sample_rate,
                    chunk_samples, endpoint=False,
                )
                wave = (0.3 * np.sin(2 * math.pi * self.freq * t)).astype(self.dtype)
                stereo = np.stack([wave, wave], axis=1).tobytes()

                with self._lock:
                    if self._proc is not None and self._proc.stdin and not self._proc.stdin.closed:
                        self._proc.stdin.write(stereo)
                        self._total_bytes_written += len(stereo)
                self._total_samples += chunk_samples
                time.sleep(chunk_samples / self.sample_rate)
        except KeyboardInterrupt:
            pass
        finally:
            # Just exit the loop. The actual pw-cat lifecycle is owned by
            # self.stop(), which is meant to be called externally. Calling
            # it here would tear down the pipe as soon as passthrough mode
            # is engaged, killing the pipeline.
            pass

    def write_frames(self, data: bytes):
        """Write raw PCM bytes to the injected stream.

        Switches from test tone to passthrough mode on first call.
        """
        if not self._running:
            return
        if not self._passthrough:
            self._passthrough = True
            if self._tone_thread is not None:
                self._tone_thread.join(timeout=1)
        with self._lock:
            if self._proc is not None and self._proc.stdin and not self._proc.stdin.closed:
                try:
                    self._proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    self._running = False

    def stop(self):
        """Stop injecting audio."""
        self._running = False
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))
    print("Starting injection...")
    injector = Injector()
    injector.start_injection()
    # Wait for the tone thread to complete (Ctrl+C)
    if injector._tone_thread is not None:
        injector._tone_thread.join()
