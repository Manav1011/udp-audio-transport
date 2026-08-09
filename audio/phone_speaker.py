"""Application-owned speaker sink lifecycle manager.

Owns the creation and destruction of the Phone_Speaker null sink. The
user explicitly selects Phone_Speaker from GNOME Sound → Output Device.
audio/capture.py reads from Phone_Speaker.monitor and forwards bytes
to AudioSender (UDP).

The module does NOT create any loopback from the default physical sink.
The previous PC_Audio_Capture + module-loopback path is gone; the user
selects the sink directly, so we only need to provide the device.

The sink is tagged with an ownership property
(`audio-bridge.owned=true` in sink_properties / device.properties) so
that stale-device cleanup can identify it unambiguously on next
startup.
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger("audio-bridge")

# Property name used as an ownership marker on all devices we create.
OWNERSHIP_PROPERTY = "audio-bridge.owned"
OWNERSHIP_VALUE = "true"


class PhoneSpeakerError(Exception):
    """Raised when Phone_Speaker creation, destruction, or monitor
    resolution fails."""


# -- pactl source-resolution helpers ----------------------------------------
#
# Phone_Speaker is a module-null-sink. On this system PipeWire does NOT
# expose its monitor as a standalone Audio/Source node — the pulse-server
# synthesizes a proxy Pulse source over the sink's monitor output ports.
#
# Passing the string name `Phone_Speaker.monitor` to pw-cat --target does
# not establish a link (the string resolves to no live PipeWire node);
# passing the numeric Pulse source index works. The index is not stable
# across runs, so it must be resolved at runtime AFTER Phone_Speaker has
# been created.


def _list_sources_short_output() -> str:
    """Return raw stdout of `pactl list sources short`. Empty string on error."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout


def _find_source_index_by_name(source_name: str, pactl_output: str) -> str | None:
    """Return column-0 index for the row whose column-1 name equals
    ``source_name`` in tab-separated ``pactl list sources short`` output.

    Returns ``None`` if no such row exists. Lines with fewer than 2
    tab-separated columns are skipped. Match on column 1 is exact
    (substring matches like ``Phone_Speaker.monitor.extra`` do not match
    ``Phone_Speaker.monitor``).

    Pure function — no I/O. Easy to unit-test.
    """
    for line in pactl_output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        if parts[1] == source_name:
            return parts[0]
    return None


class PhoneSpeakerManager:
    """Manages the Phone_Speaker null sink that the user selects
    explicitly from GNOME Sound → Output Device."""

    SINK_NAME = "Phone_Speaker"

    def __init__(self):
        self._module = None

    # -- device discovery --------------------------------------------------

    def _list_sinks(self) -> list[str]:
        """Return list of existing sink names."""
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        return [
            line.split("\t")[1]
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]

    # -- module management -------------------------------------------------

    def _load_module(self, module: str, arguments: str) -> str:
        """Load a PulseAudio/PipeWire module and return its index."""
        try:
            result = subprocess.run(
                ["pactl", "load-module", module, arguments],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise PhoneSpeakerError(f"Failed to load module: {e}")
        return result.stdout.strip()

    def _unload_module(self, index: str) -> None:
        """Unload a module by index."""
        try:
            subprocess.run(
                ["pactl", "unload-module", index],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise PhoneSpeakerError(f"Failed to unload module: {e}")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Create Phone_Speaker null sink if absent.

        Idempotent: if the sink already exists, leave it alone. The sink
        is tagged with `audio-bridge.owned=true` so stale-device cleanup
        can identify it across restarts.
        """
        sinks = self._list_sinks()
        if self.SINK_NAME in sinks:
            log.debug("Speaker sink %s already exists; reusing", self.SINK_NAME)
            return

        self._module = self._load_module(
            "module-null-sink",
            f"sink_name={self.SINK_NAME} "
            f"sink_properties="
            f"device.description=\"{self.SINK_NAME}\" "
            f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}",
        )

    def stop(self) -> None:
        """Destroy the Phone_Speaker null sink (if we created it)."""
        if self._module is None:
            return
        try:
            self._unload_module(self._module)
        except PhoneSpeakerError:
            log.exception("Failed to unload Phone_Speaker module")
        self._module = None

    # -- public accessors --------------------------------------------------

    def sink_name(self) -> str:
        """Return the sink name (user selects this in GNOME Sound)."""
        return self.SINK_NAME

    def monitor_source_name(self) -> str:
        """Return the monitor source name that capture.py records from.

        PipeWire exposes every sink's audio on a parallel `.monitor`
        source; this is the source the capture path subscribes to.
        """
        return f"{self.SINK_NAME}.monitor"

    def monitor_source_index(self) -> str:
        """Resolve the runtime Pulse source index for Phone_Speaker.monitor.

        Must be called AFTER ``start()`` so the monitor exists. Raises
        ``PhoneSpeakerError`` whose message contains the full
        ``pactl list sources short`` output if the monitor is not
        found — never silently falls back to a different device.

        The string name ``Phone_Speaker.monitor`` does NOT work as a
        pw-cat ``--target`` value on this system (no live PipeWire
        Audio/Source node with that name exists). The numeric Pulse
        source index DOES work, so callers should pass the returned
        index to pw-cat.
        """
        stdout = _list_sources_short_output()
        index = _find_source_index_by_name(self.monitor_source_name(), stdout)
        if index is None:
            raise PhoneSpeakerError(
                f"Pulse source '{self.monitor_source_name()}' not found. "
                f"`pactl list sources short` returned:\n{stdout}"
            )
        return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mgr = PhoneSpeakerManager()
    try:
        mgr.start()
        print(
            f"Phone_Speaker ready. "
            f"Sink: {mgr.sink_name()} "
            f"Monitor source: {mgr.monitor_source_name()}. "
            f"Press Ctrl+C to stop."
        )
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()