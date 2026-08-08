"""Virtual audio device lifecycle manager.

Owns the creation and destruction of Phone_Microphone (null sink) and
Phone_Microphone_Input (remapped source). No other module may invoke
pactl/pw-cli/pw-cat/wpctl directly except capture.py and injector.py.
"""
from __future__ import annotations

import subprocess


class VirtualAudioError(Exception):
    """Raised when virtual device creation or destruction fails."""


class VirtualAudioManager:
    """Manages Phone_Microphone sink + Phone_Microphone_Input remapped source.

    injector.py writes PCM into Phone_Microphone sink.
    Applications record from Phone_Microphone_Input source.
    capture.py reads PCM from the real default output monitor (no change).
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

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Create Phone_Microphone sink and Phone_Microphone_Input remapped source.

        Raises VirtualAudioError on failure with clean rollback.
        """
        # 1. Create or reuse the null sink
        if self.SINK_NAME not in self._list_sinks():
            print("Creating virtual microphone sink...")
            self._mic_module = self._load_module(
                "module-null-sink",
                f"sink_name={self.SINK_NAME} "
                f"sink_properties=device.description=\"{self.SINK_NAME}\"",
            )
        else:
            print("Reusing existing virtual microphone sink...")
        print("Virtual microphone sink ready.")

        # 2. Create or reuse the remapped source
        if self.SOURCE_NAME not in self._list_sources():
            print("Creating remapped microphone source...")
            self._remap_module = self._load_module(
                "module-remap-source",
                f"source_name={self.SOURCE_NAME} "
                f"master={self.SINK_NAME}.monitor "
                f"source_properties=device.description=\"Phone Microphone\" "
                f"channels=2 channel_map=front-left,front-right",
            )
        else:
            print("Reusing existing remapped microphone source...")
        print("Remapped microphone source ready.")

    def stop(self) -> None:
        """Destroy remapped source then null sink (reverse order)."""
        if self._remap_module is not None:
            print(f"Destroying remapped microphone ({self.SOURCE_NAME})...")
            self._unload_module(self._remap_module)
            self._remap_module = None
            print("Remapped microphone destroyed.")
        if self._mic_module is not None:
            print(f"Destroying virtual microphone sink ({self.SINK_NAME})...")
            self._unload_module(self._mic_module)
            self._mic_module = None
            print("Virtual microphone sink destroyed.")

    def sink_name(self) -> str:
        """Return the sink name (injector writes here)."""
        return self.SINK_NAME

    def source_name(self) -> str:
        """Return the remapped source name (applications record from here)."""
        return self.SOURCE_NAME


if __name__ == "__main__":
    mgr = VirtualAudioManager()
    try:
        mgr.start()
        print("\nVirtual devices ready. Press Ctrl+C to stop.\n")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mgr.stop()
