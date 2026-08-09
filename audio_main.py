"""Audio entry point — wires PipeWire audio + dual transport (TCP mic, UDP speaker).

FINAL ARCHITECTURE:

    [Android mic] --TCP--> AudioTcpMicReceiver (port 5002)
                                  -> injector.write_frames
                                  -> Phone_Microphone sink
                                  -> apps recording from Phone_Microphone_Input

    [Phone_Speaker sink, selected by user from GNOME Sound → Output Device]
        -> Phone_Speaker.monitor
        -> Capture (pw-cat record)
        -> AudioSender (UDP)
        -> Android speaker UDP listener (port 5000)

The TCP microphone transport is the ONLY mic path. UDP is reserved
for the speaker path.

The application owns and lifecycle-manages three virtual devices:
    Phone_Microphone          (null sink, mic path)
    Phone_Microphone_Input    (remapped source, apps record from this)
    Phone_Speaker             (null sink, user selects as Output Device)

On startup the application unloads any stale instances of these
devices (only those with the `audio-bridge.owned=true` marker), then
creates fresh ones. On shutdown it removes them in reverse order.

Run:
    python -m audio_main
    # env vars override defaults (see config.py)
"""
from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading

from audio.audio_manager import AudioManager
from audio.phone_speaker import PhoneSpeakerManager
from audio.virtual_audio import VirtualAudioManager
from config import (
    MICROPHONE_TCP_HOST,
    MICROPHONE_TCP_PORT,
    SPEAKER_UDP_HOST,
    SPEAKER_UDP_PORT,
)
from transport.audio_session import AudioSession
from utils.logger import log


def _get_active_default_sink_name() -> str | None:
    """Return the system default sink's node.name, or None if unknown.

    Used only to emit a one-line startup hint when Phone_Speaker exists
    but is not the active default. We never modify the default here —
    the user selects Phone_Speaker manually in GNOME Sound or by running
    `wpctl set-default <id>`.
    """
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True, text=True, timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()
    return None


def _get_speaker_sink_id(sink_name: str) -> int | None:
    """Return the wpctl numeric id for a sink by node.name, or None."""
    try:
        result = subprocess.run(
            ["wpctl", "status"],
            capture_output=True, text=True, timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        # Lines look like: "*   51. Family 17h/19h..." or "   183. Phone_Speaker..."
        if sink_name in line:
            digits = ""
            for ch in line:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    pass
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    log.info("Audio bridge starting")

    # ---- 1. Virtual devices (mic side) -----------------------------------
    # VirtualAudioManager.start() unloads any stale
    # Phone_Microphone / Phone_Microphone_Input modules that carry the
    # audio-bridge.owned marker, then creates fresh ones.
    vam = VirtualAudioManager()
    vam.start()

    # ---- 2. Virtual device (speaker side) -------------------------------
    # PhoneSpeakerManager creates the Phone_Speaker null sink with the
    # audio-bridge.owned marker. The user must select Phone_Speaker
    # from GNOME Sound → Output Device; capture reads Phone_Speaker.monitor.
    psm = PhoneSpeakerManager()
    psm.start()

    # ---- 3. Verify all three devices are present ------------------------
    log.info(
        "Virtual devices ready: %s, %s, %s",
        vam.sink_name(), vam.source_name(), psm.sink_name(),
    )

    # ---- 3b. Hint if Phone_Speaker is not the active default sink -----
    # The capture path is correct: pw-cat reads Phone_Speaker.monitor
    # and the backend sends those bytes to UDP. But the capture only
    # carries real audio when the user (or GNOME) has set Phone_Speaker
    # as the system default sink. If it isn't, emit a one-line hint
    # telling the user exactly what to do. The backend never modifies
    # the default itself — that's a user decision via GNOME Settings
    # → Sound → Output Device.
    active_default = _get_active_default_sink_name()
    if active_default is not None and active_default != psm.sink_name():
        sink_id = _get_speaker_sink_id(psm.sink_name())
        if sink_id is not None:
            log.warning(
                "Phone_Speaker exists but is not the default sink "
                "(current default: %s). PC applications will play to the "
                "current default, not to Phone_Speaker, so nothing will "
                "reach Android. Select Phone_Speaker in GNOME Settings "
                "→ Sound → Output Device, or run: wpctl set-default %d",
                active_default, sink_id,
            )
        else:
            log.warning(
                "Phone_Speaker exists but is not the default sink "
                "(current default: %s). Select Phone_Speaker in GNOME "
                "Settings → Sound → Output Device.",
                active_default,
            )

    # ---- 4. Network transports ------------------------------------------
    session = AudioSession(
        mic_bind_host=MICROPHONE_TCP_HOST,
        mic_bind_port=MICROPHONE_TCP_PORT,
        speaker_dest_host=SPEAKER_UDP_HOST,
        speaker_dest_port=SPEAKER_UDP_PORT,
    )
    session.start()

    log.info(
        "Microphone TCP server listening on %s:%d",
        MICROPHONE_TCP_HOST, MICROPHONE_TCP_PORT,
    )
    log.info(
        "Speaker UDP destination: %s:%d",
        SPEAKER_UDP_HOST, SPEAKER_UDP_PORT,
    )
    log.info("Speaker capture source: %s", psm.monitor_source_name())
    log.info("Audio bridge ready")

    mgr = AudioManager()
    # The TCP mic receiver -> injector callback is wired via
    # session.bind_injector(mgr.write_microphone_frames). The speaker
    # capture path feeds the session's sender submit function.
    session.bind_injector(mgr.write_microphone_frames)

    def _start_audio():
        # Open the injector pipe in passthrough mode (tone=False) so no
        # test tone is generated. write_frames() delivers real PCM from
        # TCP.
        mgr.injector.start_injection(tone=False)
        # Resolve the runtime Pulse source index for Phone_Speaker.monitor.
        # The string name `Phone_Speaker.monitor` does not resolve to a
        # live PipeWire node on this system (the pulse-server synthesizes
        # a proxy over the sink's monitor ports); the numeric Pulse source
        # index wires up correctly. The index changes between runs, so we
        # must look it up after Phone_Speaker has been created. If the
        # monitor is absent, PhoneSpeakerError propagates out of this
        # daemon thread with the full pactl output in its message.
        monitor_index = psm.monitor_source_index()
        log.info(
            "Speaker capture monitor: %s (Pulse source index %s)",
            psm.monitor_source_name(), monitor_index,
        )
        mgr.capture.set_capture_target(monitor_index)
        mgr.set_capture_callback(session.sender.submit)
        mgr.capture.start_capture(mgr._callback)

    audio_thread = threading.Thread(target=_start_audio, daemon=True)
    audio_thread.start()

    shutdown = {"done": False}

    def _shutdown(*_args):
        if shutdown["done"]:
            return
        shutdown["done"] = True
        log.info("Shutting down...")
        # Reverse order: network -> audio -> speaker device -> mic devices.
        try:
            session.stop()
        except Exception:
            log.exception("Failed to stop AudioSession")
        try:
            mgr.stop()
        except Exception:
            log.exception("Failed to stop AudioManager")
        try:
            psm.stop()
        except Exception:
            log.exception("Failed to stop PhoneSpeakerManager")
        try:
            vam.stop()
        except Exception:
            log.exception("Failed to stop VirtualAudioManager")
        log.info("Audio bridge stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Idle. All audio work happens in background threads.
    try:
        signal.pause()
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()