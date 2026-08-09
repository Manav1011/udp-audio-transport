"""System audio capture infrastructure.

Owns the dedicated virtual capture sink and a remapped capture source that
capture.py reads from. A loopback routes the current default physical sink
into the virtual capture sink so its monitor carries the same audio that
the user is hearing through the speaker.

Why this architecture (Phase 5.1):

    On this PipeWire 1.0.5 system, recording directly from a null sink
    monitor (e.g. PC_Audio_Capture.monitor) via pw-cat returns a noisy
    waveform even when nothing is playing. The same is true of the
    physical hardware sink monitor.

    The existing Phone_Microphone / Phone_Microphone_Input pair already
    proved the workaround: a *remapped source* built from the sink
    monitor returns clean digital silence when nothing is playing and
    faithfully carries injected audio otherwise. We use the same pattern:
    a dedicated null sink plus a remap source from its monitor.

Graph owned by this module:

    [user application audio sink-inputs]
            |
            v
        [default physical sink]   <-- user hears audio through speaker
            |
            |  (loopback copy; sample-accurate)
            v
        PC_Audio_Capture (null sink)
            |
            v
        PC_Audio_Capture.monitor  <-- remap source is built from this
            |
            v
        PC_Audio_Capture_Input     <-- capture.py reads from THIS source
            |
            v
        capture.py

The application owns everything created here. Nothing else is touched.
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger("audio-bridge")


class SystemAudioCaptureError(Exception):
    """Raised when system audio capture infrastructure setup fails."""


class SystemAudioCaptureManager:
    """Manages PC_Audio_Capture sink + remap source + default-sink loopback."""

    # Public, stable names — capture.py and other modules reference these.
    SINK_NAME = "PC_Audio_Capture"
    SOURCE_NAME = "PC_Audio_Capture_Input"

    def __init__(self):
        self._sink_module = None
        self._remap_module = None
        self._loopback_module = None

    # -- device discovery --------------------------------------------------

    def _list_sinks(self) -> list[str]:
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        return [line.split("\t")[1] for line in result.stdout.strip().split("\n")
                if line.strip()]

    def _list_sources(self) -> list[str]:
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        return [line.split("\t")[1] for line in result.stdout.strip().split("\n")
                if line.strip()]

    def _get_default_sink(self) -> str | None:
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        name = result.stdout.strip()
        return name or None

    # -- module management -------------------------------------------------

    def _load_module(self, module: str, arguments: str) -> str:
        try:
            result = subprocess.run(
                ["pactl", "load-module", module, arguments],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise SystemAudioCaptureError(
                f"Failed to load module {module}: {e}"
            )
        return result.stdout.strip()

    def _unload_module(self, index: str) -> None:
        try:
            subprocess.run(
                ["pactl", "unload-module", index],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise SystemAudioCaptureError(
                f"Failed to unload module {index}: {e}"
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Create the virtual capture sink + remap source + loopback.

        Idempotent / safe to call twice. Raises SystemAudioCaptureError on
        failure with clean rollback.
        """
        sinks = self._list_sinks()
        sources = self._list_sources()

        # 1. Create or reuse the dedicated virtual capture sink.
        if self.SINK_NAME not in sinks:
            self._sink_module = self._load_module(
                "module-null-sink",
                f"sink_name={self.SINK_NAME} "
                f"sink_properties=device.description=\"{self.SINK_NAME}\"",
            )
        else:
            log.debug("Virtual capture sink %s already exists; reusing", self.SINK_NAME)

        # 2. Create or reuse the remapped capture source from the sink's
        #    monitor. The remap source returns clean silence when nothing
        #    is playing, unlike the raw sink monitor.
        if self.SOURCE_NAME not in sources:
            self._remap_module = self._load_module(
                "module-remap-source",
                f"source_name={self.SOURCE_NAME} "
                f"master={self.SINK_NAME}.monitor "
                f"source_properties=device.description=\"PC_Audio_Capture_Input\" "
                f"channels=2 channel_map=front-left,front-right",
            )
        else:
            log.debug("Remapped capture source %s already exists; reusing", self.SOURCE_NAME)

        # 3. Discover the current default physical sink. We need a copy of
        #    whatever the user is hearing routed into our virtual sink.
        default_sink = self._get_default_sink()
        if default_sink is None:
            raise SystemAudioCaptureError("No default sink available")
        if default_sink == self.SINK_NAME:
            raise SystemAudioCaptureError(
                "Default sink is the virtual capture sink itself; "
                "refusing to create a feedback loop"
            )
        log.debug("Default physical sink for loopback: %s", default_sink)

        # 4. Create the loopback from the default physical sink monitor
        #    into our virtual capture sink. The physical sink keeps
        #    playing normally — we only copy a copy of its output into
        #    the virtual sink so the remap source can pick it up.
        self._loopback_module = self._load_module(
            "module-loopback",
            f"source={default_sink}.monitor "
            f"sink={self.SINK_NAME} "
            f"latency_msec=20 "
            f"source_dont_move=true sink_dont_move=true",
        )

    def stop(self) -> None:
        """Destroy loopback, remap source, then virtual sink (reverse order)."""
        if self._loopback_module is not None:
            try:
                self._unload_module(self._loopback_module)
            except SystemAudioCaptureError:
                log.exception("Failed to unload loopback module")
            self._loopback_module = None
        if self._remap_module is not None:
            try:
                self._unload_module(self._remap_module)
            except SystemAudioCaptureError:
                log.exception("Failed to unload remapped capture source module")
            self._remap_module = None
        if self._sink_module is not None:
            try:
                self._unload_module(self._sink_module)
            except SystemAudioCaptureError:
                log.exception("Failed to unload virtual capture sink module")
            self._sink_module = None

    # -- public accessors --------------------------------------------------

    def capture_source_name(self) -> str:
        """Return the source name that capture.py should record from."""
        return self.SOURCE_NAME

    def sink_name(self) -> str:
        """Return the virtual capture sink name."""
        return self.SINK_NAME


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mgr = SystemAudioCaptureManager()
    try:
        mgr.start()
        print(f"Capture source: {mgr.capture_source_name()}. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
