"""AudioSession — wires mic (TCP) and speaker (UDP) end-to-end.

FINAL ARCHITECTURE:

  Microphone:  Android → TCP → AudioTcpMicReceiver → on_pcm callback
                                               → injector.write_frames

  Speaker:     PC capture → AudioSender → UDP → Android speaker

The receiver is the TCP listener (TCP SERVER). The Android mic app is
the TCP CLIENT and connects to <backend_ip>:5002.

The sender is the UDP packer (UDP CLIENT). The Android speaker app is
the UDP receiver/listener.

Final-architecture decisions:
  - The TCP mic path is byte-faithful: no framing, no reordering, no
    jitter buffer, no resampling, no DSP. We trust the Android sender
    to produce the agreed PCM format (48000 Hz / stereo / Float32 LE).
  - The UDP speaker path uses the existing
    transport.audio_packet.encode_packet wire format and the existing
    AudioSender that fragments PCM into MAX_PAYLOAD-sized datagrams.
    This is unchanged.
  - The previous mic-over-UDP path (AudioReceiver + JitterBuffer +
    silence inserter + sequence recorder + UDP inspector) is no
    longer active in the production session. The modules remain in
    the codebase for tests and diagnostic scripts that reference
    them.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from transport.audio_diagnostic import (
    DiagnosticWavWriter,
    NullDiagnosticWavWriter,
    is_enabled as diagnostic_is_enabled,
)
from transport.audio_sender import AudioSender
from transport.audio_tcp_mic_receiver import AudioTcpMicReceiver

log = logging.getLogger("audio-bridge")


class AudioSession:
    """Glues an AudioTcpMicReceiver (mic) and an AudioSender (speaker)."""

    def __init__(
        self,
        mic_bind_host: str,
        mic_bind_port: int,
        speaker_dest_host: str,
        speaker_dest_port: int,
    ):
        self.mic_bind_host = mic_bind_host
        self.mic_bind_port = mic_bind_port
        self.speaker_dest = (speaker_dest_host, speaker_dest_port)

        self._injector_write: Callable[[bytes], None] | None = None
        self._diagnostic = (
            DiagnosticWavWriter() if diagnostic_is_enabled()
            else NullDiagnosticWavWriter()
        )
        self._stats = {
            "injector_chunks_dropped": 0,
            "injector_bytes_delivered": 0,
            "injector_chunks_delivered": 0,
        }
        self._stats_lock = threading.Lock()

        # Speaker = UDP, mic = TCP. The mic receiver's on_pcm callback
        # points at the same internal _deliver_to_injector funnel used
        # by the legacy AudioReceiver path, so the injector contract is
        # unchanged.
        self.mic_receiver = AudioTcpMicReceiver(
            on_pcm=self._deliver_to_injector,
            bind_host=mic_bind_host,
            bind_port=mic_bind_port,
        )
        self.sender = AudioSender(dest=self.speaker_dest)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        self.mic_receiver.start()
        self.sender.start()

    def stop(self) -> None:
        self.sender.stop()
        self.mic_receiver.stop()
        self._diagnostic.close()

    def bind_injector(self, write_frames: Callable[[bytes], None]) -> None:
        """Wire the receiver's ordered PCM output to injector.write_frames."""
        self._injector_write = write_frames

    def stats(self) -> dict:
        with self._stats_lock:
            session = dict(self._stats)
        return {
            "session": session,
            "mic": self.mic_receiver.stats(),
            "sender": self.sender.stats(),
        }

    # -- internal ---------------------------------------------------------

    def _deliver_to_injector(self, pcm: bytes) -> None:
        if self._injector_write is None:
            return
        # Tee: observe the exact bytes before they reach the injector.
        # No-op unless AUDIO_DIAGNOSTIC_RECORD=1.
        self._diagnostic.write(pcm)
        try:
            self._injector_write(pcm)
        except Exception:
            with self._stats_lock:
                self._stats["injector_chunks_dropped"] += 1
            log.exception("injector write failed")
            return
        with self._stats_lock:
            self._stats["injector_chunks_delivered"] += 1
            self._stats["injector_bytes_delivered"] += len(pcm)
