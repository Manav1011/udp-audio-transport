"""Unit tests for the SequenceIndexedPcmRecorder and its wiring.

The recorder is instrumentation-only and observes jitter-buffer releases
and losses via AudioReceiver. These tests verify:

- Direct recorder: offsets, frames, summary file
- JitterBuffer instrumentation fields (_last_released_seqs, _last_lost_seq)
- AudioReceiver wires recorder correctly (losses + deliveries)
- End-to-end loopback: sender → receiver → recorder sees monotonic seqs
"""
import json
import os
import socket
import tempfile
import time

from transport.audio_packet import encode_packet
from transport.audio_receiver import AudioReceiver, JitterBuffer
from transport.audio_sender import AudioSender
from transport.audio_sequence_recorder import (
    STEREO_FRAME_BYTES,
    SequenceIndexedPcmRecorder,
)


# ---------------------------------------------------------------------------
# Direct recorder tests
# ---------------------------------------------------------------------------

def test_recorder_disabled_when_env_unset(monkeypatch):
    from transport.audio_sequence_recorder import is_enabled, NullSequenceRecorder
    monkeypatch.delenv("AUDIO_DIAGNOSTIC_SEQUENCE", raising=False)
    assert is_enabled() is False
    n = NullSequenceRecorder()
    # All methods should be no-ops
    n.record_delivered([1, 2], [8, 8], 0.0)
    n.record_lost(3, 0.0)
    n.start()
    n.stop()


def test_recorder_records_pcm_offsets():
    """Each delivered chunk must record PCM byte/frame offset = sum of prior
    payload_lengths, and frames = bytes/8."""
    rec = SequenceIndexedPcmRecorder()
    rec.start()
    rec.record_delivered([100, 101, 102], [1152, 1152, 1152], 1.0)
    rec.stop()
    assert len(rec._delivered) == 3
    assert rec._delivered[0].pcm_offset_bytes == 0
    assert rec._delivered[1].pcm_offset_bytes == 1152
    assert rec._delivered[2].pcm_offset_bytes == 2304
    assert rec._delivered[0].pcm_offset_frames == 0
    assert rec._delivered[1].pcm_offset_frames == 144
    assert rec._delivered[2].pcm_offset_frames == 288
    assert rec._delivered[0].payload_length == 1152


def test_recorder_records_lost_packets_with_context():
    """Lost packets must capture seq + ts + nearby delivered seqs for context."""
    rec = SequenceIndexedPcmRecorder()
    rec.start()
    # Some delivered chunks first
    rec.record_delivered([10, 11, 12], [100, 100, 100], 0.0)
    # Then a loss
    rec.record_lost(13, 1.0)
    rec.record_lost(14, 1.1)
    assert len(rec._lost) == 2
    assert rec._lost[0].seq == 13
    assert rec._lost[1].seq == 14
    # nearby seqs (tail of delivered list) should be [10, 11, 12]
    assert rec._lost[0].near_delivered_seqs == [10, 11, 12]


def test_recorder_writes_json_and_summary(monkeypatch):
    """stop() flushes JSON to JSON_PATH and summary to SUMMARY_PATH."""
    from transport import audio_sequence_recorder as mod
    tmp = tempfile.mkdtemp()
    json_target = os.path.join(tmp, "seq.json")
    summary_target = os.path.join(tmp, "seq.txt")
    orig_json = mod.JSON_PATH
    orig_summary = mod.SUMMARY_PATH
    monkeypatch.setattr(mod, "JSON_PATH", json_target)
    monkeypatch.setattr(mod, "SUMMARY_PATH", summary_target)
    try:
        rec = SequenceIndexedPcmRecorder()
        rec.start()
        rec.record_delivered([1, 2, 3], [800, 800, 800], 0.0)
        rec.record_lost(4, 1.0)
        rec.stop()
        with open(json_target) as f:
            data = json.load(f)
        assert len(data["delivered"]) == 3
        assert len(data["lost"]) == 1
        assert data["delivered"][0]["seq"] == 1
        # 2400 bytes / 8 bytes per frame = 200 frames at start of chunk 3
        assert data["delivered"][2]["pcm_offset_frames"] == 200
        with open(summary_target) as f:
            summary = f.read()
        assert "delivered chunks:  3" in summary
        assert "lost packets:      1" in summary
    finally:
        monkeypatch.setattr(mod, "JSON_PATH", orig_json)
        monkeypatch.setattr(mod, "SUMMARY_PATH", orig_summary)


def test_summary_detects_misaligned_payload_lengths():
    """Chunks whose payload_length is not divisible by 8 (stereo-frame size)
    must be flagged in the summary."""
    rec = SequenceIndexedPcmRecorder()
    rec.start()
    # 1153 is not divisible by 8 — should be flagged as misaligned
    rec.record_delivered([1, 2], [1153, 1152], 0.0)
    rec.record_lost(3, 1.0)
    rec.stop()
    misaligned = [c for c in rec._delivered
                  if c.payload_length % STEREO_FRAME_BYTES != 0]
    assert len(misaligned) == 1
    assert misaligned[0].seq == 1


# ---------------------------------------------------------------------------
# JitterBuffer instrumentation fields
# ---------------------------------------------------------------------------

def test_jitter_records_released_seqs_in_order():
    buf = JitterBuffer(reorder_window_ms=100)
    # Single in-order
    out = buf.push(0, b"x")
    assert out == [b"x"]
    assert buf._last_released_seqs == [0]
    # In-order release following gap-fill
    buf.push(1, b"y")
    assert buf._last_released_seqs == [1]
    buf.push(3, b"a")  # gap of 1
    assert buf._last_released_seqs == []  # buffered, not released
    buf.push(2, b"b")
    # Should have released [2, 3]
    assert buf._last_released_seqs == [2, 3]


def test_jitter_records_lost_seq_on_skip():
    """After a timeout-induced skip, _last_lost_seq should equal the seq
    we declared lost, and _last_released_seqs should list whatever
    buffered packets we drained."""
    buf = JitterBuffer(reorder_window_ms=100)
    buf.push(100, b"p100")  # establishes next_expected = 101
    buf.push(102, b"p102")  # buffered
    buf.push(103, b"p103")  # buffered
    # Pin gap start time deterministically
    buf._gap_started_at = 5000.0
    # Tick past the window — should skip seq 101
    out = buf.tick(now=5000.0 + 0.20)
    assert out == [b"p102", b"p103"]
    assert buf._last_lost_seq == 101
    assert buf._last_released_seqs == [102, 103]


# ---------------------------------------------------------------------------
# End-to-end loopback: sender → receiver → recorder
# ---------------------------------------------------------------------------

def _bind_ephemeral():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    return s


def test_loopback_records_all_packets(monkeypatch):
    """Send 50 packets, observe them all in the recorder's delivered list."""
    from transport import audio_sequence_recorder as mod
    tmp = tempfile.mkdtemp()
    json_path = os.path.join(tmp, "seq.json")
    summary_path = os.path.join(tmp, "seq.txt")
    orig_json = mod.JSON_PATH
    orig_summary = mod.SUMMARY_PATH
    monkeypatch.setattr(mod, "JSON_PATH", json_path)
    monkeypatch.setattr(mod, "SUMMARY_PATH", summary_path)

    bind_sock = _bind_ephemeral()
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    rec = SequenceIndexedPcmRecorder()
    received_pcm: list[bytes] = []

    sender = AudioSender(dest=("127.0.0.1", port))
    receiver = AudioReceiver(
        on_pcm=lambda b: received_pcm.append(b),
        bind_host="127.0.0.1",
        bind_port=port,
        sequence_recorder=rec,
    )
    rec.start()
    receiver.start()
    sender.start()
    try:
        # Send 50 packets × 1152 bytes = 57600 bytes = 7200 stereo frames
        pcm = b"\x00" * (1152 * 50)
        sender.submit(pcm)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if len(received_pcm) >= 50:
                break
            time.sleep(0.05)
    finally:
        sender.stop()
        receiver.stop()
        rec.stop()

    assert len(received_pcm) == 50, f"expected 50 packets, got {len(received_pcm)}"
    assert len(rec._delivered) == 50
    # Seqs should be 0..49
    seqs = [c.seq for c in rec._delivered]
    assert seqs == list(range(50))
    # PCM offsets should grow monotonically by 1152 each
    offsets = [c.pcm_offset_bytes for c in rec._delivered]
    assert offsets == [i * 1152 for i in range(50)]
    # Payload lengths should be uniform
    lens = {c.payload_length for c in rec._delivered}
    assert lens == {1152}
    # Each 1152-byte chunk = 144 stereo frames
    frames = [c.pcm_offset_frames for c in rec._delivered]
    assert frames == [i * 144 for i in range(50)]
    # No losses in a perfect loopback
    assert len(rec._lost) == 0

    # Restore module paths
    monkeypatch.setattr(mod, "JSON_PATH", orig_json)
    monkeypatch.setattr(mod, "SUMMARY_PATH", orig_summary)


def test_loopback_with_simulated_loss_records_lost_seq(monkeypatch):
    """Drop seqs 17 and 35 by patching the sender's _send_fragmented.
    The receiver/recorder should declare those seqs lost after the timeout."""
    from transport import audio_sequence_recorder as mod
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(mod, "JSON_PATH", os.path.join(tmp, "seq.json"))
    monkeypatch.setattr(mod, "SUMMARY_PATH", os.path.join(tmp, "seq.txt"))

    bind_sock = _bind_ephemeral()
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    rec = SequenceIndexedPcmRecorder()
    received_pcm: list[bytes] = []

    sender = AudioSender(dest=("127.0.0.1", port))

    def dropping_fragment(pcm: bytes) -> None:
        """Same as _send_fragmented but skips seqs 17 and 35."""
        assert sender._sock is not None
        seq = sender._seq
        for offset in range(0, len(pcm), 1152):
            payload = pcm[offset:offset + 1152]
            if seq in (17, 35):
                seq += 1
                continue
            datagram = encode_packet(seq, payload)
            sender._sock.sendto(datagram, sender.dest)
            sender._stats["packets_sent"] += 1
            sender._stats["bytes_sent"] += len(datagram)
            seq += 1
        sender._seq = seq

    sender._send_fragmented = dropping_fragment
    receiver = AudioReceiver(
        on_pcm=lambda b: received_pcm.append(b),
        bind_host="127.0.0.1",
        bind_port=port,
        sequence_recorder=rec,
        reorder_window_ms=80,  # short so the test doesn't have to wait long
    )
    rec.start()
    receiver.start()
    sender.start()
    try:
        # 40 packets' worth. We drop 17 and 35, so 38 should arrive.
        pcm = b"\x00" * (1152 * 40)
        sender.submit(pcm)
        # Wait long enough for the recv loop's 500ms heartbeat ticks to
        # drive jitter timeouts past 80ms gaps. With only 38 packets
        # arriving at the receiver's normal pace (well under 1 second),
        # the loopback is done within ~1 second. We poll until both losses
        # show up.
        deadline = time.time() + 6.0
        while time.time() < deadline:
            time.sleep(0.05)
            if len(received_pcm) >= 38 and len(rec._lost) >= 2:
                break
    finally:
        sender.stop()
        receiver.stop()
        rec.stop()

    # 38 packets delivered (we sent 40, dropped 2)
    assert len(received_pcm) == 38, f"expected 38 delivered, got {len(received_pcm)}"
    # Both 17 and 35 must be declared lost
    lost_seqs = sorted(l.seq for l in rec._lost)
    assert lost_seqs == [17, 35], f"expected lost=[17,35], got {lost_seqs}"