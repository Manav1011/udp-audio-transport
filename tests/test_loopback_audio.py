"""Integration test: real capture -> UDP -> real injector on localhost.

Pipeline under test (FINAL ARCHITECTURE — speaker path):
  pw-cat tone -> Phone_Speaker sink (selected as system Output Device)
  Phone_Speaker.monitor -> pw-cat record -> AudioSender -> UDP -> test listener
  -> injector.write_frames -> Phone_Microphone -> Phone_Microphone.monitor
  -> Phone_Microphone_Input

In production, the UDP listener is the Android speaker app. In this
test, we stand up a tiny in-process UDP forwarder that mimics what the
Android app does: receive UDP packets and forward their payloads to
the injector. This isolates the speaker pipeline from the now-TCP mic
path.

We simultaneously:
  1. Play a 440 Hz sine into Phone_Speaker (so capture has something to read)
  2. Run the capture -> UDP -> injector pipeline (with the test forwarder)
  3. Record from Phone_Microphone_Input via pw-record
  4. Verify the recorded WAV contains a tone (peak > threshold)

Requires:
  - pw-cat, pw-record (pipewire-utils)
  - PulseAudio/PipeWire running
  - Virtual mic devices (Phase 4 setup). The test creates them if absent.
  - Phone_Speaker sink (Phase 6 setup). The test creates it if absent.
"""
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import wave

import numpy as np
import pytest

from audio.audio_manager import AudioManager
from audio.phone_speaker import PhoneSpeakerManager
from audio.virtual_audio import VirtualAudioManager
from transport.audio_packet import decode_packet
from transport.audio_session import AudioSession


def _pw_cat_available() -> bool:
    return shutil.which("pw-cat") is not None and shutil.which("pw-record") is not None


def _play_tone(stop_event: threading.Event, target_sink: str, freq: float = 440.0):
    """Generate a 440 Hz float32 stereo signal and feed it to the
    named sink (--target)."""
    proc = subprocess.Popen(
        ["pw-cat", "-p", "--target", target_sink,
         "--format", "f32", "--channels", "2",
         "--rate", "48000", "-"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    sr = 48000
    chunk = sr // 20  # 50 ms
    phase = 0.0
    try:
        while not stop_event.is_set():
            t = np.arange(chunk) / sr
            s = (0.3 * np.sin(2 * math.pi * freq * (phase + t))).astype(np.float32)
            phase += chunk / sr
            stereo = np.stack([s, s], axis=1).tobytes()
            try:
                proc.stdin.write(stereo)
                proc.stdin.flush()
            except BrokenPipeError:
                break
            time.sleep(chunk / sr)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def _udp_forwarder(stop_event: threading.Event, listen_port: int,
                   forward_to):
    """Receive UDP datagrams on listen_port, decode the wire format,
    and forward each payload's PCM bytes to forward_to(). Mimics the
    role the Android speaker app plays in production."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", listen_port))
    sock.settimeout(0.2)
    try:
        while not stop_event.is_set():
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            decoded = decode_packet(data)
            if decoded is not None:
                _seq, pcm = decoded
                forward_to(pcm)
    finally:
        sock.close()


@pytest.mark.skipif(not _pw_cat_available(), reason="pw-cat / pw-record not installed")
def test_capture_to_injector_loopback_carries_signal():
    vam = VirtualAudioManager()
    vam.start()
    psm = PhoneSpeakerManager()
    psm.start()
    try:
        # Ephemeral UDP port for the speaker transport (sender -> forwarder).
        bind_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        bind_sock.bind(("127.0.0.1", 0))
        udp_port = bind_sock.getsockname()[1]
        bind_sock.close()

        # Ephemeral TCP port for the mic transport (unused for this test
        # but the session needs both arguments).
        tcp_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_probe.bind(("127.0.0.1", 0))
        tcp_port = tcp_probe.getsockname()[1]
        tcp_probe.close()

        session = AudioSession(
            mic_bind_host="127.0.0.1",
            mic_bind_port=tcp_port,
            speaker_dest_host="127.0.0.1",
            speaker_dest_port=udp_port,
        )
        session.start()

        mgr = AudioManager()
        # Speaker path: capture from Phone_Speaker.monitor -> session.sender
        # -> UDP -> forwarder -> injector -> Phone_Microphone
        mgr.capture.set_capture_source(psm.monitor_source_name())
        mgr.set_capture_callback(session.sender.submit)
        session.bind_injector(mgr.write_microphone_frames)

        # Inline audio start (capture is blocking — use a thread)
        def _start_audio():
            mgr.injector.start_injection(tone=False)
            mgr.capture.start_capture(mgr._callback)

        audio_thread = threading.Thread(target=_start_audio, daemon=True)
        audio_thread.start()

        # Stand-in for the Android speaker app: receive UDP packets on the
        # speaker port and forward the payload PCM to the injector.
        stop_forwarder = threading.Event()
        forwarder_thread = threading.Thread(
            target=_udp_forwarder,
            args=(stop_forwarder, udp_port, mgr.write_microphone_frames),
            daemon=True,
        )
        forwarder_thread.start()

        # Start the tone source playing into Phone_Speaker so its
        # monitor carries the signal.
        stop_tone = threading.Event()
        tone_thread = threading.Thread(
            target=_play_tone,
            args=(stop_tone, psm.sink_name()),
            daemon=True,
        )
        tone_thread.start()

        # Give everything a moment to come up
        time.sleep(1.0)

        # Record from Phone_Microphone_Input while audio flows
        out_wav = "/tmp/audio_test_loopback.wav"
        if os.path.exists(out_wav):
            os.remove(out_wav)

        rec = subprocess.Popen(
            ["pw-record", "--target", "Phone_Microphone_Input", out_wav],
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3.0)
        rec.terminate()
        try:
            rec.wait(timeout=3)
        except subprocess.TimeoutExpired:
            rec.kill()

        # Stop tone, then cleanly shut down the pipeline
        stop_tone.set()
        tone_thread.join(timeout=2)
        stop_forwarder.set()
        forwarder_thread.join(timeout=2)

        mgr.stop()
        session.stop()

    finally:
        psm.stop()
        vam.stop()

    # Verify the WAV
    assert os.path.exists(out_wav), f"pw-record did not produce {out_wav}"
    with wave.open(out_wav, "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        assert nchannels == 2, f"expected 2 channels, got {nchannels}"
        assert sampwidth == 2, f"expected 16-bit, got {sampwidth}"
        assert framerate == 48000, f"expected 48000 Hz, got {framerate}"
        assert nframes >= 48000 * 2, f"too short: {nframes} frames"

        raw = w.readframes(nframes)
        pcm = np.frombuffer(raw, dtype=np.int16)
        peak = int(np.max(np.abs(pcm)))
        assert peak > 1000, f"audio is silent (peak={peak}) — pipeline not carrying signal"