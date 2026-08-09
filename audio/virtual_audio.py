"""Virtual audio device lifecycle manager (microphone side).

Owns the creation and destruction of Phone_Microphone (null sink) and
Phone_Microphone_Input (remapped source). No other module may invoke
pactl/pw-cli/pw-cat/wpctl directly except capture.py and injector.py.

Ownership:
    Both devices are tagged with `audio-bridge.owned=true` in their
    PulseAudio sink_properties / source_properties. Stale-device
    cleanup at startup unloads any pre-existing instance of these
    names ONLY if the ownership marker is present. Devices that share
    a name but lack the marker are considered user/system devices and
    are never touched.
"""
from __future__ import annotations

import json
import logging
import subprocess

from audio.phone_speaker import (
    OWNERSHIP_PROPERTY,
    OWNERSHIP_VALUE,
)

log = logging.getLogger("audio-bridge")


class VirtualAudioError(Exception):
    """Raised when virtual device creation or destruction fails."""


class VirtualAudioManager:
    """Manages Phone_Microphone sink + Phone_Microphone_Input remapped source.

    injector.py writes PCM into Phone_Microphone sink.
    Applications record from Phone_Microphone_Input source.
    """

    SINK_NAME = "Phone_Microphone"
    SOURCE_NAME = "Phone_Microphone_Input"

    def __init__(self):
        self._mic_module = None
        self._remap_module = None

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
        return [line.split("\t")[1] for line in result.stdout.strip().split("\n")
                if line.strip()]

    def _list_sources(self) -> list[str]:
        """Return list of existing source names."""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        return [line.split("\t")[1] for line in result.stdout.strip().split("\n")
                if line.strip()]

    def _list_sink_modules(self) -> list[dict]:
        """Return list of loaded modules with their index, name, and
        arguments as parsed dict keys. Each entry is
        ``{"index": str, "name": str, "args": dict}``.
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "modules"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        modules = []
        current = None
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Module #"):
                if current is not None:
                    modules.append(current)
                current = {
                    "index": stripped.split("#", 1)[1].strip(),
                    "name": "",
                    "args": {},
                }
            elif current is not None:
                if stripped.startswith("Name:"):
                    current["name"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Argument:"):
                    arg_str = stripped.split(":", 1)[1].strip()
                    current["args"] = _parse_module_arguments(arg_str)
        if current is not None:
            modules.append(current)
        return modules

    # -- module management -------------------------------------------------

    def _load_module(self, module: str, arguments: str) -> str:
        """Load a PulseAudio/PipeWire module and return its index."""
        try:
            result = subprocess.run(
                ["pactl", "load-module", module, arguments],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise VirtualAudioError(f"Failed to load module: {e}")
        return result.stdout.strip()

    def _unload_module(self, index: str) -> None:
        """Unload a module by index."""
        try:
            subprocess.run(
                ["pactl", "unload-module", index],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise VirtualAudioError(f"Failed to unload module: {e}")

    # -- stale-device cleanup ---------------------------------------------

    def _unload_orphans(self) -> None:
        """Unload any previously-created instances of our owned devices.

        Scans loaded modules for `module-null-sink` and `module-remap-source`
        entries whose arguments carry the `audio-bridge.owned=true`
        marker. Those are leftover devices from a previous (possibly
        crashed) run; unload them so we can recreate them cleanly.

        Devices that share the SINK_NAME / SOURCE_NAME but lack the
        ownership marker are NOT touched — they are presumed to belong
        to the user or the system and would be a destructive action
        to remove.
        """
        modules = self._list_sink_modules()
        for m in modules:
            args = m["args"]
            if args.get(OWNERSHIP_PROPERTY) != OWNERSHIP_VALUE:
                continue
            # Is this an instance of one of our owned device names?
            target = None
            if (
                m["name"] == "module-null-sink"
                and args.get("sink_name") == self.SINK_NAME
            ):
                target = "sink"
            elif (
                m["name"] == "module-remap-source"
                and args.get("source_name") == self.SOURCE_NAME
            ):
                target = "source"
            if target is None:
                continue
            log.info(
                "Unloading stale %s (module #%s, owned by previous run)",
                target, m["index"],
            )
            try:
                self._unload_module(m["index"])
            except VirtualAudioError:
                log.exception("Failed to unload stale %s", target)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Create Phone_Microphone sink and Phone_Microphone_Input remapped source.

        Before creating, unload any prior-run orphans that match our
        ownership marker. Then create fresh devices (idempotent: if
        non-orphan devices with these names already exist, reuse them
        — though normally the cleanup above removes our leftovers).

        Raises VirtualAudioError on failure with clean rollback.
        """
        # Stale-device cleanup before anything else.
        self._unload_orphans()

        sinks = self._list_sinks()
        sources = self._list_sources()

        # 1. Create the null sink
        if self.SINK_NAME not in sinks:
            self._mic_module = self._load_module(
                "module-null-sink",
                f"sink_name={self.SINK_NAME} "
                f"sink_properties=device.description=\"{self.SINK_NAME}\" "
                f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}",
            )
        else:
            log.debug("Virtual mic sink %s already exists; reusing", self.SINK_NAME)

        # 2. Create the remapped source
        if self.SOURCE_NAME not in sources:
            self._remap_module = self._load_module(
                "module-remap-source",
                f"source_name={self.SOURCE_NAME} "
                f"master={self.SINK_NAME}.monitor "
                f"source_properties=device.description=\"Phone Microphone\" "
                f"channels=2 channel_map=front-left,front-right "
                f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}",
            )
        else:
            log.debug("Remapped mic source %s already exists; reusing", self.SOURCE_NAME)

    def stop(self) -> None:
        """Destroy remapped source then null sink (reverse order)."""
        if self._remap_module is not None:
            try:
                self._unload_module(self._remap_module)
            except VirtualAudioError:
                log.exception("Failed to unload remapped microphone module")
            self._remap_module = None
        if self._mic_module is not None:
            try:
                self._unload_module(self._mic_module)
            except VirtualAudioError:
                log.exception("Failed to unload virtual microphone module")
            self._mic_module = None

    def sink_name(self) -> str:
        """Return the sink name (injector writes here)."""
        return self.SINK_NAME

    def source_name(self) -> str:
        """Return the remapped source name (applications record from here)."""
        return self.SOURCE_NAME


def _parse_module_arguments(arg_str: str) -> dict:
    """Parse a PulseAudio/PipeWire module Argument: string into a dict.

    The Argument: value is a sequence of whitespace-separated tokens.
    When PipeWire echoes the arguments, simple key=value pairs are
    emitted with the value verbatim — values that contained spaces at
    load time are emitted as plain words without quote protection,
    so a multi-word value (e.g. ``device.description="Phone Speaker Sink"``)
    appears as multiple space-separated tokens after the parser.

    For our ownership-detection purpose we only need:
        - the simple tokens ``audio-bridge.owned``, ``sink_name``,
          ``source_name`` (no spaces in key or value).

    To handle multi-word values without losing them, the parser adopts
    a two-pass strategy:
        1. Tokenize on whitespace.
        2. For each token that matches ``KEY=VALUE`` where both KEY
           and VALUE are simple identifiers (no whitespace inside),
           store it as a key in the result dict. A token that looks
           like a bare word is treated as part of the value of the
           most recent key (and ignored — we don't need it).

    This avoids depending on quote-escaping semantics that the pulse
    daemon does not preserve in its output.
    """
    args: dict = {}
    for token in arg_str.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        # Both key and value must be simple identifiers (no spaces,
        # no quotes) — otherwise we are looking at a fragment of a
        # multi-word value and we don't care about it.
        if not k or not v:
            continue
        if any(c.isspace() for c in k) or any(c.isspace() for c in v):
            continue
        args[k] = v
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mgr = VirtualAudioManager()
    try:
        mgr.start()
        print("Virtual devices ready. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()