"""Speaker capture — records PCM from the application-owned
Phone_Speaker virtual sink's monitor source.

The capture source is `Phone_Speaker.monitor`, exposed by PipeWire for
every sink. The user explicitly selects `Phone_Speaker` from
GNOME Sound → Output Device, so its monitor carries whatever audio
the user is hearing through the phone call.

The monitor source is, by construction of the underlying
`module-null-sink`, already in the transport's native format:

    48 kHz
    stereo
    Float32 LE

We therefore capture it DIRECTLY in that format and forward the raw
bytes to AudioSender (UDP). No resampling, no mono->stereo
duplication, no s16->f32 conversion, no DSP — the speaker path is a
straight passthrough from the monitor to UDP.

This is the ONLY place where the speaker capture path lives. It is
intentionally separate from the microphone path (which uses
MicCapturePipeline to convert from a USB microphone's PCM16 mono
native format into the transport's float32 stereo 48 kHz format).

Test-tone mode is preserved separately in audio/injector.py and is not
routed through here.
"""
from __future__ import annotations

import logging
import select
import signal
import subprocess
import sys
import threading

log = logging.getLogger("audio-bridge")

sys.stdout.reconfigure(line_buffering=True)

# Default capture source — owned by PhoneSpeakerManager
# (Phone_Speaker.monitor). audio_main.py sets it explicitly via
# set_capture_source() before start_capture(); this default exists so
# the standalone demo and tests have a sensible value.
DEFAULT_CAPTURE_SOURCE = "Phone_Speaker.monitor"

# Transport-format constants — the monitor's native format.
TRANSPORT_SAMPLE_RATE = 48000
TRANSPORT_CHANNELS = 2
_TRANSPORT_FRAME_BYTES = TRANSPORT_CHANNELS * 4  # 2 channels * 4 bytes (f32)


class Capture:
    """Captures PCM bytes from the dedicated virtual capture source.

    The capture path is a straight passthrough:

        pw-cat (Float32 LE stereo, 48 kHz) -> callback(bytes)

    Exposes start_capture(callback) — callback receives transport-format
    float32 stereo bytes, ready for the UDP sender.
    """

    def __init__(self, capture_source: str = DEFAULT_CAPTURE_SOURCE):
        self._proc = None
        self._lock = threading.Lock()
        self._capture_source = capture_source

    # -- configuration ------------------------------------------------------

    def set_capture_source(self, source_name: str) -> None:
        """Override the capture source (default: Phone_Speaker.monitor)."""
        self._capture_source = source_name

    def set_capture_target(self, target: str) -> None:
        """Override the pw-cat --target argument directly.

        ``target`` is passed verbatim as ``--target`` to pw-cat — it
        may be either a node name (e.g. ``Phone_Speaker.monitor``) or
        a numeric Pulse source index (e.g. ``2581``). Use this when
        the string name does not resolve to a live PipeWire node; the
        numeric Pulse source index always does.

        This is an additive alternative to ``set_capture_source()``;
        the existing ``set_capture_source`` API is preserved.
        """
        self._capture_source = target

    # -- lifecycle ---------------------------------------------------------

    def start_capture(self, callback):
        """Start capturing audio. callback(bytes) is invoked with each
        transport-format (48 kHz / stereo / Float32 LE) chunk."""
        monitor = self._capture_source
        log.debug("Capturing from source: %s", monitor)

        # The monitor source for a module-null-sink is, by construction,
        # Float32 LE stereo at 48 kHz — the transport format. Capture it
        # directly with pw-cat; no resampling, no conversion.
        cmd = [
            "pw-cat", "-r",
            "--target", monitor,
            "--format", "f32",
            "--channels", str(TRANSPORT_CHANNELS),
            "--rate", str(TRANSPORT_SAMPLE_RATE),
            "-",  # write to stdout
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.error("pw-cat not found. Install pipewire-utils.")
            return

        # Native float32 stereo chunk size: 100ms of stereo audio.
        # 4 bytes per sample, 2 channels.
        chunk_frames = TRANSPORT_SAMPLE_RATE // 10  # 100ms in frames
        chunk_size = chunk_frames * _TRANSPORT_FRAME_BYTES

        with self._lock:
            self._proc = proc

        try:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    raw = proc.stdout.read(chunk_size)
                    if not raw:
                        break
                    # Raw bytes are already in transport format —
                    # forward them directly to the sender.
                    if callback:
                        callback(raw)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop(proc)

    def _stop(self, proc):
        """Terminate the pw-cat process."""
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

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
