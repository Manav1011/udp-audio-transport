"""Audio entry point — wires PipeWire audio + dual TCP transport (mic + speaker).

FINAL ARCHITECTURE (both transports are TCP):

    [Android mic] --TCP--> AudioTcpMicReceiver (port 5002)
                                  -> injector.write_frames
                                  -> Phone_Microphone sink
                                  -> apps recording from Phone_Microphone_Input

    [Phone_Speaker sink, selected by user from GNOME Sound → Output Device]
        -> Phone_Speaker.monitor
        -> Capture (pw-cat record)
        -> AudioTcpSpeakerSender (TCP client)
        -> Android TCP speaker server (port 5000)

Both transports are TCP. Mic and speaker each have a dedicated TCP
connection on a separate port — they are NOT multiplexed.

SPEAKER STARTUP CONTRACT — capture must not start before the TCP
connection is up:

    1. Create the Phone_Speaker virtual device and resolve its monitor
       Pulse source index.
    2. Start the AudioTcpSpeakerSender; it begins attempting to connect.
    3. Wait until the connection is established. Do NOT start pw-cat
       until this happens.
    4. Start pw-cat to feed the sender.

    On disconnect: stop pw-cat (no stale PCM is generated). On
    reconnect: restart pw-cat from the current live monitor.

The application owns and lifecycle-manages three virtual devices:
    Phone_Microphone          (null sink, mic path)
    Phone_Microphone_Input    (remapped source, apps record from this)
    Phone_Speaker             (null sink, user selects as Output Device)

On startup the application unloads any stale instances of these
devices (only those with the `audio-bridge.owned=true` marker), then
creates fresh ones. On shutdown it removes them in reverse order.

Run:
    SPEAKER_TCP_HOST=<phone-ip> python -m audio_main
    # env vars override defaults (see config.py)
"""
from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
from typing import Callable

from audio.audio_manager import AudioManager
from audio.phone_speaker import PhoneSpeakerManager
from audio.virtual_audio import VirtualAudioManager
from config import (
    MICROPHONE_TCP_HOST,
    MICROPHONE_TCP_PORT,
    SPEAKER_TCP_HOST,
    SPEAKER_TCP_PORT,
)
from transport.audio_session import AudioSession
from utils.logger import log

# How long to wait for the Android TCP speaker server to accept our
# connection on startup. After this we still proceed (capture will
# start later on reconnect), but emit a warning.
_SPEAKER_CONNECT_TIMEOUT_S = 30.0


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


class _SpeakerCaptureController:
    """Owns the lifecycle of the pw-cat speaker capture subprocess.

    The pw-cat read loop is blocking and runs on its own thread. To
    "pause" capture we stop the subprocess (the read loop sees EOF
    and returns); to resume we spawn a new thread. We never queue
    audio while paused — there is no audio to queue.

    State callbacks from AudioTcpSpeakerSender drive this controller:

        connected=True   -> start_capture_thread()
        connected=False  -> stop_capture_thread()  (and no more PCM
                            is generated, so the sender stays idle
                            until reconnect)

    The controller is single-threaded with respect to its own state:
    the sender invokes our callbacks on the sender's thread, and we
    use a lock plus a generation counter to ensure that a stop/start
    pair never overlaps. Each generation is its own capture thread.
    """

    def __init__(
        self,
        audio_manager: AudioManager,
        sender_submit,
        is_connected: Callable[[], bool] | None = None,
    ):
        self._mgr = audio_manager
        self._submit = sender_submit
        # Optional predicate consulted in start() so the controller
        # cannot miss an already-connected sender. The sender's state
        # callback only fires on transitions; if it transitioned to
        # True before we registered (or while _desired_running was
        # False) the callback was either never invoked or invoked
        # and dropped, and no further transition is coming to
        # retrigger it. ``is_connected`` lets start() close that
        # race directly. ``audio_main.py`` passes
        # ``session.sender.is_connected``.
        self._is_connected = is_connected
        self._lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._generation = 0
        self._desired_running = False
        self._submit_set = False

    def install_submit(self, submit) -> None:
        """Wire the sender's submit() into the capture callback. Idempotent."""
        with self._lock:
            self._submit = submit
            self._submit_set = True

    def start(self) -> None:
        """Begin tracking the sender's connection state.

        Sets ``_desired_running = True`` so future state callbacks
        are honored, and — if ``is_connected`` was supplied at
        construction and reports True right now — kicks off capture
        immediately. This closes the race where the sender connected
        before the controller was ready.
        """
        with self._lock:
            self._desired_running = True
        if self._is_connected is not None and self._is_connected():
            self._start_capture_thread()

    def stop(self) -> None:
        """Stop the capture subprocess and refuse to start it again."""
        with self._lock:
            self._desired_running = False
        self._stop_capture_locked()

    def on_state(self, connected: bool) -> None:
        """Sender state callback. Invoked on the sender thread."""
        with self._lock:
            desired = self._desired_running
        if not desired:
            return
        if connected:
            self._start_capture_thread()
        else:
            self._stop_capture_locked()

    # -- internal ---------------------------------------------------------

    def _start_capture_thread(self) -> None:
        """Start (or no-op if already running) the pw-cat read loop."""
        with self._lock:
            existing = self._capture_thread
            if existing is not None and existing.is_alive():
                return
            self._generation += 1
            gen = self._generation
            submit = self._submit
            submit_set = self._submit_set

        if not submit_set or submit is None:
            log.error(
                "Speaker capture cannot start: sender submit not wired"
            )
            return

        log.info("Speaker capture starting (generation %d)", gen)

        def _run_capture():
            try:
                self._mgr.capture.start_capture(submit)
            except Exception:
                log.exception(
                    "Speaker capture thread crashed (generation %d)", gen
                )
            finally:
                # The capture thread is done. If this was the current
                # generation, clear it; otherwise it's a stale thread
                # that we replaced.
                with self._lock:
                    if self._generation == gen:
                        self._capture_thread = None
                log.info("Speaker capture thread exited (generation %d)", gen)

        t = threading.Thread(
            target=_run_capture,
            name=f"speaker-capture-{gen}",
            daemon=True,
        )
        with self._lock:
            self._capture_thread = t
        t.start()

    def _stop_capture_locked(self) -> None:
        """Kill the current pw-cat subprocess (read loop returns)."""
        with self._lock:
            t = self._capture_thread
        if t is None:
            return
        try:
            self._mgr.capture.stop()
        except Exception:
            log.exception("Speaker capture stop failed")
        # Don't join — the thread exits on its own when the subprocess
        # pipe closes (within ~2s of stop()). join() would block the
        # sender callback thread on shutdown.


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
        speaker_dest_host=SPEAKER_TCP_HOST,
        speaker_dest_port=SPEAKER_TCP_PORT,
    )
    session.start()

    log.info(
        "Microphone TCP server listening on %s:%d",
        MICROPHONE_TCP_HOST, MICROPHONE_TCP_PORT,
    )
    log.info(
        "Speaker TCP destination: %s:%d",
        SPEAKER_TCP_HOST, SPEAKER_TCP_PORT,
    )

    # ---- 5. Speaker capture lifecycle -----------------------------------
    # The TCP sender's start() has already been invoked by
    # session.start() — it is now attempting to connect on a daemon
    # thread. We do NOT start pw-cat yet. Instead we wait until the
    # connection is up, and then start the capture subprocess. On
    # subsequent disconnect/reconnect we pause and resume capture via
    # the sender's state callback so we never generate PCM that has
    # nowhere to go.
    monitor_index = psm.monitor_source_index()
    log.info(
        "Speaker capture monitor: %s (Pulse source index %s)",
        psm.monitor_source_name(), monitor_index,
    )

    mgr = AudioManager()
    session.bind_injector(mgr.write_microphone_frames)
    mgr.injector.start_injection(tone=False)
    mgr.capture.set_capture_target(monitor_index)

    speaker_controller = _SpeakerCaptureController(
        audio_manager=mgr,
        sender_submit=None,  # wired below
        # Pass the sender's is_connected() so the controller's
        # start() can detect the case where the sender already
        # connected before the controller was ready. The speaker
        # sender's state callback only fires on transitions; if the
        # initial connect happened before the callback was registered
        # (or while _desired_running was False), the callback was
        # either never invoked or invoked and dropped, and no
        # transition is coming to retrigger it. Without this, the
        # very first Android START AUDIO STREAM would wait ~30
        # seconds for the sender's idle MSG_PEEK probe to time out
        # before the reconnect actually started capture.
        is_connected=session.sender.is_connected,
    )
    speaker_controller.install_submit(session.sender.submit)
    # Register the state callback BEFORE we wait on the sender. If
    # the sender's daemon thread already connected (Android pressed
    # START AUDIO STREAM during the few-millisecond window between
    # sender.start() and now), _notify_state(True) fired with no
    # listener and was lost. Registering here means we are ready for
    # the next transition; the is_connected predicate above closes
    # the gap for the connection that already happened.
    session.sender.add_state_callback(speaker_controller.on_state)

    if session.sender.wait_until_connected(timeout=_SPEAKER_CONNECT_TIMEOUT_S):
        log.info("Speaker TCP connected on startup; starting capture")
    else:
        log.warning(
            "Speaker TCP not connected after %.1fs; capture will start "
            "automatically when the Android speaker server accepts the "
            "connection",
            _SPEAKER_CONNECT_TIMEOUT_S,
        )

    speaker_controller.start()

    log.info("Audio bridge ready")

    shutdown = {"done": False}

    def _shutdown(*_args):
        if shutdown["done"]:
            return
        shutdown["done"] = True
        log.info("Shutting down...")
        # Reverse order: capture -> network -> speaker device -> mic devices.
        try:
            speaker_controller.stop()
        except Exception:
            log.exception("Failed to stop speaker capture")
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
