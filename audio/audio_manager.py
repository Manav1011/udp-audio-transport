"""Audio manager — top-level coordinator for capture and injection.

Owns one Capture and one Injector instance. The rest of the backend
never calls pw-cat or subprocess directly.
"""
from __future__ import annotations

import logging

from audio.capture import Capture
from audio.injector import Injector

log = logging.getLogger("audio-bridge")


class AudioManager:
    """Coordinates audio capture and injection."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.capture = Capture()
        self.injector = Injector(sample_rate=sample_rate, channels=channels)
        self._callback = None

    def set_capture_callback(self, callback):
        """Set callback(bytes) invoked when PCM is captured."""
        self._callback = callback

    def write_microphone_frames(self, data: bytes):
        """Write PCM bytes to the microphone injection stream."""
        self.injector.write_frames(data)

    def start(self):
        """Start audio — begin capture and injection."""
        if self._callback is not None:
            self.capture.start_capture(self._callback)
        else:
            log.error("No capture callback set — capture will not start")
        self.injector.start_injection()

    def stop(self):
        """Stop audio — halt capture and injection."""
        self.capture.stop()
        self.injector.stop()


def main():
    """Standalone demo: capture + injection."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    mgr = AudioManager()

    def on_capture(data: bytes):
        log.info("captured %d bytes", len(data))

    mgr.set_capture_callback(on_capture)
    mgr.start()

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        mgr.stop()


if __name__ == "__main__":
    main()
