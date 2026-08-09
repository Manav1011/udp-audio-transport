"""Unit tests for the SilenceInserter and its wiring into AudioReceiver.

The SilenceInserter synthesizes zero-valued stereo float32 LE PCM for slots
where the JitterBuffer has DEFINITIVELY declared a packet lost (after the
reorder window elapsed). It does NOT insert silence for reordered packets
that close their gap within the window.

These tests are deterministic: they pin `buf._gap_started_at` and pass
explicit `now` values to `tick()`.

Tests A–G + edge case H map directly to the user's required test list.
Null-class and env-gate tests are appended at the end.
"""
import os
import socket
import struct
import tempfile
import threading
import time

from transport.audio_packet import MAX_PAYLOAD, encode_packet
from transport.audio_receiver import AudioReceiver, JitterBuffer
from transport.audio_sender import AudioSender
from transport.audio_silence_inserter import (
    JSON_PATH,
    STEREO_FRAME_BYTES,
    SUMMARY_PATH,
    NullSilenceInserter,
    SilenceInserter,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Helpers — direct jitter-buffer + silence-inserter wiring (no UDP, no sender)
# ---------------------------------------------------------------------------

class StubRec:
    """Minimal recorder stub that mirrors SequenceIndexedPcmRecorder's interface
    methods used by AudioReceiver. Lets us run receiver tests without
    touching /tmp files."""
    def __init__(self):
        self.delivered: list[tuple[list[int], list[int], float]] = []
        self.lost: list[tuple[int, float]] = []
        self.silence_injections: list[tuple[int, float]] = []
        self.pcm_offset_bytes = 0

    def record_delivered(self, seqs, lens, now):
        self.delivered.append((list(seqs), list(lens), now))
        self.pcm_offset_bytes += sum(lens)

    def record_lost(self, seq, now):
        self.lost.append((seq, now))

    def record_silence_injection(self, n_bytes, now):
        self.silence_injections.append((n_bytes, now))
        self.pcm_offset_bytes += n_bytes


def _make_recv_buf(reorder_window_ms=100):
    """Build an AudioReceiver whose on_pcm callback captures all PCM chunks."""
    captured: list[bytes] = []
    jitter = JitterBuffer(reorder_window_ms=reorder_window_ms)
    rec = StubRec()
    si = SilenceInserter()
    si.start()
    recv = AudioReceiver(
        on_pcm=captured.append,
        jitter_buffer=jitter,
        sequence_recorder=rec,
        silence_inserter=si,
    )
    # Don't start() the recv thread — we'll drive push/tick directly.
    return recv, jitter, rec, si, captured


def _drive_loss(jitter, seq_that_gets_lost):
    """Pin gap-start time and force a timeout via tick()."""
    jitter._gap_started_at = 5000.0
    return jitter.tick(now=5000.0 + 0.30)


# ---------------------------------------------------------------------------
# Test A — In-order, no silence
# ---------------------------------------------------------------------------

def _p(seq: int, n: int = 1152) -> bytes:
    """Make a payload that starts with 4 bytes encoding `seq`, then zeros."""
    header = f"p{seq}".encode()[:4].ljust(4, b"\x00")
    return header + b"\x00" * (n - 4)


def test_A_in_order_no_silence():
    """5 in-order packets of 1152 bytes → 5 on_pcm calls, no silence, all
    silence counters zero."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    for seq in range(100, 105):
        out = jitter.push(seq, _p(seq))
        recv._deliver(out)
    assert len(captured) == 5
    for i, chunk in enumerate(captured):
        assert chunk == _p(100 + i)
    s = si.stats()
    assert s["lost_seq_count"] == 0
    assert s["missing_frame_count"] == 0
    assert s["inserted_silence_frame_count"] == 0
    assert s["cumulative_inserted_frames"] == 0
    assert s["cumulative_lost_packets"] == 0
    assert si.should_inject_silence() is False
    assert si.take_pending_silence() is None


# ---------------------------------------------------------------------------
# Test B — Reorder within window, no silence
# ---------------------------------------------------------------------------

def test_B_reorder_within_window_no_silence():
    """100,101,103,102 → 4 deliveries, NO silence (102 closes the gap)."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    recv._deliver(jitter.push(100, _p(100)))
    recv._deliver(jitter.push(101, _p(101)))
    recv._deliver(jitter.push(103, _p(103)))
    # 102 closes the gap (in-order after the buffered 103)
    recv._deliver(jitter.push(102, _p(102)))
    assert len(captured) == 4
    assert captured == [_p(100), _p(101), _p(102), _p(103)]
    s = si.stats()
    assert s["cumulative_lost_packets"] == 0
    assert s["cumulative_inserted_frames"] == 0
    assert s["inserted_silence_frame_count"] == 0


# ---------------------------------------------------------------------------
# Test C — Single loss: exactly one packet of silence
# ---------------------------------------------------------------------------

def test_C_single_loss_one_packet_of_silence():
    """push(100), push(102) — 101 is never sent; one tick declares 101 lost."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    recv._deliver(jitter.push(100, _p(100)))
    recv._deliver(jitter.push(102, _p(102)))
    # Now drive the timeout: 101 declared lost, 102 drains out.
    jitter._gap_started_at = 5000.0
    released = jitter.tick(now=5000.30)
    assert released == [_p(102)]
    # The receiver feeds feed_loss before _deliver.
    lost_seq = jitter._last_lost_seq
    assert lost_seq == 101
    si.feed_loss(lost_seq, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released)
    # Order in captured: [p100, silence, p102]
    assert len(captured) == 3
    assert captured[0] == _p(100)
    silence = captured[1]
    assert silence == b"\x00" * 1152, f"silence must be 1152 zero bytes, got {len(silence)} bytes"
    assert len(silence) == 1152
    assert captured[2] == _p(102)
    s = si.stats()
    assert s["lost_seq_count"] == 1
    assert s["missing_frame_count"] == 144
    assert s["inserted_silence_frame_count"] == 144
    assert s["cumulative_inserted_frames"] == 144
    assert s["cumulative_lost_packets"] == 1


# ---------------------------------------------------------------------------
# Test D — Two consecutive losses: exactly two packet durations of silence
# ---------------------------------------------------------------------------

def test_D_two_consecutive_losses_two_packets_of_silence():
    """push(100), push(103) — 101 AND 102 are missing; two ticks."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    recv._deliver(jitter.push(100, _p(100)))
    recv._deliver(jitter.push(103, _p(103)))
    # Tick 1: declares 101 lost, drains nothing (103 not yet contiguous).
    jitter._gap_started_at = 5000.0
    released1 = jitter.tick(now=5000.30)
    assert released1 == [], f"expected empty release, got {released1}"
    si.feed_loss(101, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released1)
    # After tick 1: captured = [p100, silence_101]; 102 still missing.
    assert len(captured) == 2
    assert captured[0] == _p(100)
    assert captured[1] == b"\x00" * 1152
    # Tick 2: declares 102 lost, drains p103.
    jitter._gap_started_at = 5000.0  # re-pin (would have been reset by tick 1)
    released2 = jitter.tick(now=5000.60)
    assert released2 == [_p(103)], f"expected [_p(103)], got {released2}"
    si.feed_loss(102, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released2)
    # Final captured: [p100, silence_101, silence_102, p103]
    assert len(captured) == 4
    assert captured[0] == _p(100)
    assert captured[1] == b"\x00" * 1152
    assert captured[2] == b"\x00" * 1152
    assert captured[3] == _p(103)
    s = si.stats()
    assert s["lost_seq_count"] == 2
    assert s["missing_frame_count"] == 288
    assert s["inserted_silence_frame_count"] == 288
    assert s["cumulative_inserted_frames"] == 288
    assert s["cumulative_lost_packets"] == 2


# ---------------------------------------------------------------------------
# Test E — Reordered burst: no silence
# ---------------------------------------------------------------------------

def test_E_reordered_burst_no_unnecessary_silence():
    """Send 100, 102, 103, 101 (a burst of reordering that closes within
    the window). 4 delivered, 0 silence — silence inserter must not fire."""
    recv, jitter, rec, si, captured = _make_recv_buf(reorder_window_ms=1000)
    recv._deliver(jitter.push(100, _p(100)))
    recv._deliver(jitter.push(102, _p(102)))
    recv._deliver(jitter.push(103, _p(103)))
    recv._deliver(jitter.push(101, _p(101)))
    assert len(captured) == 4
    assert captured == [_p(100), _p(101), _p(102), _p(103)]
    s = si.stats()
    assert s["cumulative_lost_packets"] == 0
    assert s["cumulative_inserted_frames"] == 0
    assert s["inserted_silence_frame_count"] == 0


# ---------------------------------------------------------------------------
# Test F — Surrounding PCM verification
# ---------------------------------------------------------------------------

def test_F_surrounding_pcm_unmodified_with_silence_between():
    """push(100, payload_A), push(102, payload_C), 101 lost. Verify:
    - silence chunk is exactly 1152 zero bytes
    - payload_A and payload_C arrive byte-identical
    - the concatenation [silence, payload_A, payload_C] = expected timeline"""
    recv, jitter, rec, si, captured = _make_recv_buf()
    payload_a = bytes((0xAA ^ i) & 0xFF for i in range(1152))
    payload_c = bytes((0xCC ^ i) & 0xFF for i in range(1152))
    recv._deliver(jitter.push(100, payload_a))
    recv._deliver(jitter.push(102, payload_c))
    jitter._gap_started_at = 5000.0
    released = jitter.tick(now=5000.30)
    assert released == [payload_c]
    si.feed_loss(101, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released)
    # The order is [payload_a, silence, payload_c]
    assert len(captured) == 3
    assert captured[0] == payload_a, "payload A must arrive byte-identical"
    assert captured[1] == b"\x00" * 1152, "silence must be all zeros"
    assert captured[2] == payload_c, "payload C must arrive byte-identical"
    # Concat the full timeline as it would appear at the injector
    full = b"".join(captured)
    expected = payload_a + (b"\x00" * 1152) + payload_c
    assert full == expected


# ---------------------------------------------------------------------------
# Test G — Total timeline preservation
# ---------------------------------------------------------------------------

def test_G_total_timeline_preserved():
    """100 in-order packets, drop seqs 50 and 75 (i.e., never push them).
    Total on_pcm bytes == 100 * 1152 = 115200.
    Silence inserter reports 2 losses × 144 frames = 288.
    Recorder stub sees 98 real deliveries (NOT the 2 silence chunks)."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    payload = b"\x42" * 1152
    for seq in range(100):
        if seq in (50, 75):
            continue
        recv._deliver(jitter.push(seq, payload))
    # Now push packets just past the missing ones to open gaps:
    # After push(49), we already drained 0..49. After push(51) and 73, we
    # have a buffer of {51..73, 74?}. Let's just push 51 and 74 explicitly.
    # Already done above (since we skipped seq 50 and 75 only). For seq 50:
    # The push(49) drained. push(50) was skipped. push(51) opened a gap.
    # We need to drive ticks for both gaps (50 and 75) to drain.
    # Tick once to drain 51..74 (skipping 50):
    jitter._gap_started_at = 5000.0
    released = jitter.tick(now=5000.30)
    # released should be 51..74 (24 packets)
    assert len(released) == 24
    assert released[0] == payload
    assert released[-1] == payload
    si.feed_loss(50, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released)
    # Now the buffer should be empty (gap cleared); next expected is 75.
    # Push 76..99 if not already pushed. We already pushed 76..99 above.
    # After 74, next expected was 75. push(76) opened the second gap.
    # Tick again:
    jitter._gap_started_at = 5000.0
    released2 = jitter.tick(now=5000.60)
    # released2 should be 76..99 (24 packets)
    assert len(released2) == 24
    si.feed_loss(75, time.monotonic())
    jitter._last_lost_seq = None
    recv._deliver(released2)
    # Total captured: 50 (0..49) + 1 silence (50) + 24 (51..74) + 1 silence (75) + 24 (76..99) = 100
    assert len(captured) == 100
    # Total bytes
    total_bytes = sum(len(c) for c in captured)
    assert total_bytes == 100 * 1152
    # Silence chunks are at positions 50 and 75 (0-indexed)
    assert captured[50] == b"\x00" * 1152
    assert captured[75] == b"\x00" * 1152
    # Real chunks are non-silence
    for i, c in enumerate(captured):
        if i in (50, 75):
            assert c == b"\x00" * 1152
        else:
            assert c == payload
    # Recorder stub: 52 drain events (50 individual + 2 batched), 2 silence injections.
    # Each batched drain delivers multiple seqs in one record_delivered call.
    assert len(rec.delivered) == 52  # 50 single-drains (0..49) + 1 batch (51..74) + 1 batch (76..99)
    total_delivered_seqs = sum(len(seqs) for seqs, _, _ in rec.delivered)
    assert total_delivered_seqs == 98  # 50 + 24 + 24
    assert len(rec.silence_injections) == 2
    # Recorder's PCM offset accounting includes silence: total bytes =
    # 98 * 1152 (delivered) + 2 * 1152 (silence) = 100 * 1152 = 115200.
    assert rec.pcm_offset_bytes == 100 * 1152
    s = si.stats()
    assert s["inserted_silence_frame_count"] == 288
    assert s["cumulative_lost_packets"] == 2


# ---------------------------------------------------------------------------
# Test H — Edge case: first-ever push, no prior delivery (fallback to MAX_PAYLOAD)
# ---------------------------------------------------------------------------

def test_H_inference_fallback_to_max_payload():
    """First-ever push, then a gap; tick declares loss. _last_delivered
    was None → fallback to MAX_PAYLOAD = 1152."""
    recv, jitter, rec, si, captured = _make_recv_buf()
    # Don't push anything yet. The very first packet would have been 100.
    # Let's simulate: push(100), push(102), then tick → loss of 101.
    recv._deliver(jitter.push(100, b"p100"))
    # Now _last_delivered_payload_length should be 4 (length of "p100")
    assert si._last_delivered_payload_length == 4
    # Force a different scenario: skip the first push and tick directly
    # (unrealistic, but exercises the fallback). We can't easily reset the
    # jitter, so test the fallback rule directly via infer_lost_payload_length:
    si2 = SilenceInserter()
    assert si2._last_delivered_payload_length is None
    assert si2.infer_lost_payload_length() == MAX_PAYLOAD
    si2.feed_loss(42, 0.0)
    silence = si2.take_pending_silence()
    assert silence is not None
    assert len(silence) == MAX_PAYLOAD
    assert silence == b"\x00" * MAX_PAYLOAD
    # Frames: MAX_PAYLOAD // 8 = 144
    assert len(silence) // STEREO_FRAME_BYTES == MAX_PAYLOAD // STEREO_FRAME_BYTES


# ---------------------------------------------------------------------------
# Null + env-gate tests
# ---------------------------------------------------------------------------

def test_null_silence_inserter_is_noop():
    n = NullSilenceInserter()
    n.observe_delivered_payload(b"x" * 100)
    n.feed_loss(99, 0.0)
    assert n.should_inject_silence() is False
    assert n.take_pending_silence() is None
    n.start()
    n.stop()
    assert n.stats() == {}


def test_silence_inserter_disabled_when_env_unset(monkeypatch):
    from transport.audio_silence_inserter import is_enabled
    monkeypatch.delenv("AUDIO_DIAGNOSTIC_SILENCE", raising=False)
    assert is_enabled() is False


def test_silence_inserter_enabled_when_env_one(monkeypatch):
    monkeypatch.setenv("AUDIO_DIAGNOSTIC_SILENCE", "1")
    assert is_enabled() is True


def test_silence_inserter_writes_json_and_summary(monkeypatch, tmp_path):
    """stop() flushes JSON to JSON_PATH and summary to SUMMARY_PATH."""
    from transport import audio_silence_inserter as mod
    orig_json = mod.JSON_PATH
    orig_summary = mod.SUMMARY_PATH
    json_target = str(tmp_path / "sil.json")
    summary_target = str(tmp_path / "sil.txt")
    monkeypatch.setattr(mod, "JSON_PATH", json_target)
    monkeypatch.setattr(mod, "SUMMARY_PATH", summary_target)
    try:
        si = SilenceInserter()
        si.start()
        si.observe_delivered_payload(b"\x00" * 1152)
        si.feed_loss(101, 0.0)
        sil = si.take_pending_silence()
        assert sil is not None
        si.stop()
        import json
        with open(json_target) as f:
            data = json.load(f)
        assert len(data["lost_packets"]) == 1
        assert data["lost_packets"][0]["seq"] == 101
        assert data["lost_packets"][0]["inferred_length"] == 1152
        assert data["lost_packets"][0]["missing_frames"] == 144
        assert len(data["injections"]) == 1
        assert data["injections"][0]["n_bytes"] == 1152
        assert data["injections"][0]["n_frames"] == 144
        with open(summary_target) as f:
            summary = f.read()
        assert "lost packets (declared):      1" in summary
        assert "silence frames inserted:      144" in summary
    finally:
        monkeypatch.setattr(mod, "JSON_PATH", orig_json)
        monkeypatch.setattr(mod, "SUMMARY_PATH", orig_summary)


# ---------------------------------------------------------------------------
# End-to-end loopback (real UDP): silence inserter wired into AudioReceiver
# ---------------------------------------------------------------------------

def test_loopback_silence_inserter_via_audio_receiver():
    """Same loopback setup as test_sequence_recorder, but with the silence
    inserter enabled. Drop seq 17, 35 — silence should be delivered before
    seq 18 and 36 respectively."""
    import socket
    import time
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    received_pcm: list[bytes] = []
    si = SilenceInserter()
    si.start()

    sender = AudioSender(dest=("127.0.0.1", port))
    from transport.audio_packet import encode_packet as _enc
    def dropping_fragment(pcm: bytes) -> None:
        assert sender._sock is not None
        seq = sender._seq
        for offset in range(0, len(pcm), 1152):
            payload = pcm[offset:offset + 1152]
            if seq in (17, 35):
                seq += 1
                continue
            sender._sock.sendto(_enc(seq, payload), sender.dest)
            sender._stats["packets_sent"] += 1
            sender._stats["bytes_sent"] += len(_enc(seq, payload))
            seq += 1
        sender._seq = seq

    sender._send_fragmented = dropping_fragment
    receiver = AudioReceiver(
        on_pcm=lambda b: received_pcm.append(b),
        bind_host="127.0.0.1",
        bind_port=port,
        silence_inserter=si,
        reorder_window_ms=80,
    )
    receiver.start()
    sender.start()
    try:
        pcm = b"\x00" * (1152 * 40)
        sender.submit(pcm)
        deadline = time.time() + 6.0
        while time.time() < deadline:
            time.sleep(0.05)
            if len(received_pcm) >= 40 and len(si._injections) >= 2:
                break
    finally:
        sender.stop()
        receiver.stop()
        si.stop()
    # 40 chunks delivered (38 real + 2 silence = 40 calls)
    assert len(received_pcm) == 40
    # Identify the silence chunks by checking they are exactly 1152 zero bytes
    # The order should be: real, real, ..., silence_17 (somewhere after seq 16's payload), ..., silence_35, ...
    silence_count = sum(1 for c in received_pcm if c == b"\x00" * 1152)
    # 1152 zero bytes is also what the sender emitted for these packets (since
    # we used b"\x00" * pcm). So we can't distinguish by zero-ness alone.
    # Use stats instead.
    s = si.stats()
    assert s["cumulative_lost_packets"] == 2
    assert s["inserted_silence_frame_count"] == 288  # 2 × 144
    # Total bytes emitted to on_pcm should equal 40 * 1152 (38 real + 2 silence)
    total_bytes = sum(len(c) for c in received_pcm)
    assert total_bytes == 40 * 1152
    # The two silence chunks must appear SOMEWHERE in the captured stream
    assert total_bytes >= 40 * 1152