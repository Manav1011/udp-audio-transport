"""Regression test: speaker startup race.

The production wiring in audio_main.py does, in order:

    1. AudioTcpSpeakerSender.start()  -> the sender's daemon thread dials
       the Android speaker server. If the server is already up, the
       connection is established almost immediately.
    2. _SpeakerCaptureController is constructed; its state callback is
       registered with the sender.
    3. sender.wait_until_connected(timeout=...) blocks until the
       connection is up. Because the sender may have already connected
       in step 1, this returns promptly without firing any NEW state
       transition.
    4. speaker_controller.start() flips _desired_running = True.

The bug this test guards against: if the sender connected in step 1
before the state callback was registered (or before start() flipped
_desired_running), the controller's on_state(True) callback either
never fired or fired with _desired_running = False and returned early.
After step 4 there is no incoming transition to retrigger it, so the
pw-cat capture thread is NEVER spawned. The sender's idle MSG_PEEK
probe then hits the 30 s socket timeout, which we mis-classify as a
disconnect, and only on the *next* connection does the capture
finally start. The Android side gets silence for ~30 s.

The exact real-world sequence is:

    backend starts -> Android presses START AUDIO STREAM for the first
    time -> speaker TCP connects immediately -> speaker capture must
    start immediately. No reconnect, no 30-second wait.

We verify the contract by exercising the production controller
(_SpeakerCaptureController inside audio_main.py) directly against a
real LoopbackServer-backed AudioTcpSpeakerSender. We also exercise
the production wiring order: install_submit + add_state_callback,
then wait_until_connected, then start, and finally assert that the
capture thread is alive WITHOUT any reconnect having occurred.
"""
from __future__ import annotations

import threading
import time

import pytest

import audio_main
from transport.audio_tcp_speaker_sender import AudioTcpSpeakerSender


# A minimal LoopbackServer that accepts exactly one connection. Borrowed
# in spirit from test_audio_tcp_speaker_sender.py but kept inline so
# this test is self-contained and does not depend on test ordering.
class _OneShotServer:
    def __init__(self) -> None:
        import socket

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self._sock.settimeout(5.0)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._accepted = threading.Event()
        self._conn = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        import socket

        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._conn = conn
            self._accepted.set()
            # Read forever (or until the test ends) so the sender's
            # MSG_PEEK probe sees a live peer and does not time out.
            conn.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return

    def close(self) -> None:
        self._stop.set()
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass


class _FakeCapture:
    """Stand-in for audio.capture.Capture used by the controller.

    The controller calls .capture.start_capture(submit) to spawn its
    pw-cat read loop. We provide a stub that records the call and
    exposes a stop() method so the controller's lifecycle works.

    Each start_capture() call creates a NEW local stop-event so that
    a prior stop() does not immediately unblock a freshly-spawned
    thread. This mirrors the real pw-cat subprocess model where each
    generation is a fresh child process.
    """

    def __init__(self) -> None:
        self.start_count = 0
        self.start_count_lock = threading.Lock()
        self._stop_for_current: threading.Event | None = None
        self._stop_lock = threading.Lock()
        self.submit = None

    def wait_for_new_start(self, seen_count: int, timeout: float) -> bool:
        """Block until start_count strictly exceeds ``seen_count``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.start_count_lock:
                if self.start_count > seen_count:
                    return True
            time.sleep(0.005)
        return False

    def wait_for_first_start(self, timeout: float) -> bool:
        """Block until start_count is at least 1."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.start_count_lock:
                if self.start_count >= 1:
                    return True
            time.sleep(0.005)
        return False

    def start_capture(self, submit) -> None:
        with self.start_count_lock:
            self.start_count += 1
        # Each invocation gets its own stop event so stop() called
        # before this start (or for an earlier generation) does NOT
        # affect us.
        my_stop = threading.Event()
        with self._stop_lock:
            self._stop_for_current = my_stop
        self.submit = submit
        while not my_stop.is_set():
            time.sleep(0.02)

    def stop(self) -> None:
        # Signal whichever stop event the current generation owns.
        with self._stop_lock:
            ev = self._stop_for_current
        if ev is not None:
            ev.set()


class _FakeAudioManager:
    """Stand-in for AudioManager exposing .capture."""

    def __init__(self) -> None:
        self.capture = _FakeCapture()

    # AudioManager exposes write_microphone_frames, but the controller
    # never calls it. Stub it for attribute completeness.
    def write_microphone_frames(self, data: bytes) -> None:  # pragma: no cover
        return None


def _wait_connected(sender: AudioTcpSpeakerSender, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sender.is_connected():
            return
        time.sleep(0.01)
    raise AssertionError("sender did not connect within timeout")


def _wait_capture_started(capture: _FakeCapture, timeout: float = 1.0) -> None:
    """Block until capture.start_capture() has been invoked at least once."""
    assert capture.wait_for_first_start(timeout=timeout), (
        "speaker capture never started within timeout — the controller "
        "missed the already-connected state"
    )


def _wait_capture_restarted(
    capture: _FakeCapture, after_count: int, timeout: float = 1.0,
) -> None:
    """Block until capture.start_capture() is invoked AGAIN after
    ``after_count``. Used to verify a reconnect spawned a fresh
    capture thread. Caller must pass the count it observed last.
    """
    assert capture.wait_for_new_start(after_count, timeout=timeout), (
        "speaker capture did not restart on reconnect — the controller "
        "did not spawn a fresh capture thread"
    )


def test_speaker_capture_starts_when_sender_already_connected():
    """The exact production race: sender connects before the controller
    is ready, then controller.start() is called. Capture MUST start
    without waiting for a disconnect/reconnect.

    Real-world scenario: backend starts, Android immediately presses
    START AUDIO STREAM, speaker TCP connects before the controller's
    state callback can act on it.

    This test mirrors the production wiring in audio_main.py:
    speaker_controller is constructed with is_connected=session.sender.is_connected,
    the state callback is registered, and start() is called. The
    is_connected predicate makes start() close the race automatically;
    audio_main.py does NOT manually invoke on_state(True) after start.
    """
    server = _OneShotServer()
    server.start()
    sender = AudioTcpSpeakerSender(host="127.0.0.1", port=server.port)

    # ---- Mirror the production wiring order from audio_main.py ----------
    # Step 1: start the sender (it will connect on its daemon thread).
    sender.start()
    # Step 2: register the controller's state callback BEFORE waiting
    # on the sender. The callback is now in place for any future
    # transition; the sender's _notify_state(True) that fires on the
    # initial connection still has no listener (controller not yet
    # constructed in production order — but here we add the listener
    # on a freshly-constructed controller, so the listener IS present
    # and WILL fire when the sender connects a moment from now).
    _wait_connected(sender)
    assert sender.is_connected()

    # Now construct the controller, wire its submit/callback, and
    # call start(). At this moment the sender is ALREADY connected,
    # so any state callback that fired during step 1 either never
    # ran (no callback registered yet, in production) or ran while
    # _desired_running was False and was dropped (also in production).
    mgr = _FakeAudioManager()
    Controller = audio_main._SpeakerCaptureController
    # Production wiring: pass the sender's is_connected so the
    # controller's start() can detect an already-connected sender and
    # spawn capture immediately. This is what makes the race-free
    # property hold without audio_main.py having to manually
    # reconcile the state.
    controller = Controller(
        audio_manager=mgr,
        sender_submit=None,
        is_connected=sender.is_connected,
    )
    controller.install_submit(sender.submit)
    sender.add_state_callback(controller.on_state)
    controller.start()

    # The contract: capture must start immediately. NO disconnect,
    # NO reconnect, NO 30-second wait, NO manual on_state(True) call.
    _wait_capture_started(mgr.capture, timeout=1.0)
    assert sender.is_connected(), (
        "sender disconnected before capture started — the reconnect "
        "path was taken, which is exactly the bug"
    )
    # And the capture thread on the controller should be alive (not
    # None, not exited).
    assert controller._capture_thread is not None
    assert controller._capture_thread.is_alive()

    # Cleanup.
    controller.stop()
    sender.stop()
    server.close()


def test_speaker_capture_restarts_on_reconnect_after_disconnect():
    """Verify the existing reconnect path still works.

    This is the path the stop/start Android cycle exercises: sender
    is connected, controller is running, then disconnects, then
    reconnects. The state callback fires True on the reconnect and
    spawns a fresh capture thread. We must not regress this.

    We exercise the controller's on_state(True) directly (without
    standing up a second TCP server): the production reconnect path
    is already covered by
    test_audio_tcp_speaker_sender.test_e2e_connected_then_disconnect_
    then_idle_then_reconnect. Here we only assert that the
    controller's on_state(True) handler spawns capture when fired
    while _desired_running is True and the previous capture thread
    has exited.
    """
    server = _OneShotServer()
    server.start()
    sender = AudioTcpSpeakerSender(host="127.0.0.1", port=server.port)

    mgr = _FakeAudioManager()
    Controller = audio_main._SpeakerCaptureController
    controller = Controller(audio_manager=mgr, sender_submit=None)
    controller.install_submit(sender.submit)

    # Simulate the controller already running with capture active.
    controller.start()
    controller.on_state(True)  # first connect
    _wait_capture_started(mgr.capture, timeout=1.0)
    first_start_count = mgr.capture.start_count
    first_thread = controller._capture_thread
    assert first_thread is not None and first_thread.is_alive()

    # Disconnect: controller stops the capture thread. Wait for the
    # previous capture thread to fully exit before triggering the
    # reconnect — otherwise on_state(True) will see the existing
    # (still-exiting) thread and skip spawning a new one, which is
    # the controller's correct no-double-spawn behavior.
    controller.on_state(False)
    first_thread.join(timeout=2.0)
    # The controller's _run_capture finally clears _capture_thread
    # once the body returns; give that a moment.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and controller._capture_thread is not None:
        time.sleep(0.01)
    assert controller._capture_thread is None, (
        "controller._capture_thread not cleared after disconnect"
    )

    # Reconnect: a fresh capture thread must spawn.
    controller.on_state(True)
    _wait_capture_restarted(mgr.capture, after_count=first_start_count, timeout=1.0)
    second_thread = controller._capture_thread
    assert second_thread is not None and second_thread.is_alive()
    assert second_thread is not first_thread, (
        "reconnect did not spawn a fresh capture thread"
    )

    # Cleanup.
    controller.stop()
    sender.stop()
    server.close()