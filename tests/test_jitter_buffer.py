"""Unit tests for the JitterBuffer.

The JitterBuffer uses a time-based reorder window (default 200 ms). When a
gap opens (an out-of-order packet arrives before its predecessor), it is
held until either the missing seq arrives or the window elapses, at which
point exactly one packet is declared lost and the rest of the buffer is
preserved.

These tests exercise deterministic scenarios by passing `now` explicitly
to tick(). reorder_window_ms=100 keeps the math clean.
"""
from transport.audio_receiver import JitterBuffer


def _seq(buf: JitterBuffer, n: int, payload: bytes | None = None) -> list[bytes]:
    return buf.push(n, payload if payload is not None else f"p{n}".encode())


# ---------------------------------------------------------------------------
# Initial state and basic in-order delivery
# ---------------------------------------------------------------------------

def test_initial_state():
    buf = JitterBuffer(reorder_window_ms=100)
    assert buf.stats() == {
        "received": 0, "duplicates": 0, "out_of_order": 0,
        "gaps": 0, "lost": 0, "late_released": 0,
    }


def test_in_order_returns_immediately():
    buf = JitterBuffer(reorder_window_ms=100)
    # The very first arriving packet is always accepted (no predecessor
    # to wait for). After that, in-order packets release immediately.
    assert _seq(buf, 0) == [b"p0"]
    assert _seq(buf, 1) == [b"p1"]
    assert _seq(buf, 2) == [b"p2"]
    assert buf.stats()["received"] == 3
    assert buf.stats()["out_of_order"] == 0
    assert buf.stats()["gaps"] == 0
    assert buf.stats()["lost"] == 0


def test_long_run_in_order():
    buf = JitterBuffer(reorder_window_ms=100)
    for i in range(1000):
        out = _seq(buf, i)
        assert out == [f"p{i}".encode()]
    assert buf.stats()["received"] == 1000
    assert buf.stats()["out_of_order"] == 0
    assert buf.stats()["gaps"] == 0


# ---------------------------------------------------------------------------
# Reordering within the reorder window
# ---------------------------------------------------------------------------

def test_out_of_order_buffers():
    """100, 102, 103 — 102 and 103 buffered; need 101 to release."""
    buf = JitterBuffer(reorder_window_ms=100)
    assert _seq(buf, 100) == [b"p100"]
    # 102, 103 are buffered (out of order)
    assert _seq(buf, 102) == []
    assert _seq(buf, 103) == []
    # 101 arrives — releases 101, 102, 103 in order
    out = _seq(buf, 101)
    assert out == [b"p101", b"p102", b"p103"]
    assert buf.stats()["gaps"] == 0
    assert buf.stats()["lost"] == 0


def test_required_reorder_scenario_100_101_103_102():
    """Required test: 100,101,103,102 -> output 100,101,102,103."""
    buf = JitterBuffer(reorder_window_ms=100)
    assert _seq(buf, 100) == [b"p100"]
    assert _seq(buf, 101) == [b"p101"]
    assert _seq(buf, 103) == []
    assert _seq(buf, 102) == [b"p102", b"p103"]


def test_burst_reorder_larger_than_old_64_packet_limit():
    """A burst of reordering up to 200 packets (3× the old 64-slot limit)
    must still deliver correctly as long as the missing predecessor
    arrives within the reorder window.
    """
    buf = JitterBuffer(reorder_window_ms=1000)  # generous window for this test
    # Start with packet 0
    assert _seq(buf, 0) == [b"p0"]
    # Send 200, 199, ..., 2 in reverse order. None are in-order (next_expected=1).
    # 200 is the first to arrive; it opens a gap.
    for seq in range(200, 1, -1):
        out = _seq(buf, seq)
        assert out == [], f"push({seq}) should have buffered; got {out}"
    assert buf.stats()["out_of_order"] == 199
    assert buf.stats()["gaps"] == 0
    # Now the missing 1 arrives — release 1..200 contiguously
    out = _seq(buf, 1)
    assert out == [f"p{i}".encode() for i in range(1, 201)]
    assert buf.stats()["gaps"] == 0
    assert buf.stats()["lost"] == 0


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_duplicate_of_released_seq_dropped():
    buf = JitterBuffer(reorder_window_ms=100)
    _seq(buf, 0)
    _seq(buf, 1)
    assert _seq(buf, 0) == []
    assert _seq(buf, 1) == []
    assert buf.stats()["duplicates"] == 2


def test_duplicate_of_buffered_seq_dropped():
    buf = JitterBuffer(reorder_window_ms=100)
    _seq(buf, 0)
    _seq(buf, 2)  # buffered
    out = _seq(buf, 2)  # duplicate
    assert out == []
    assert buf.stats()["duplicates"] == 1


# ---------------------------------------------------------------------------
# Permanent loss with time-based skip
# ---------------------------------------------------------------------------

def test_permanently_missing_after_timeout():
    """After the reorder window elapses, the missing seq is declared lost
    and the buffered packets release in order."""
    buf = JitterBuffer(reorder_window_ms=100)
    # Adopt seq=100 as the start
    assert _seq(buf, 100) == [b"p100"]
    # 102 arrives — opens a gap (missing 101). Capture the time so the
    # test can deterministically advance the clock.
    t0 = 5000.0
    assert _seq(buf, 102) == []
    gap_start = buf._gap_started_at
    # Force the gap start time to a known value so the test is deterministic.
    buf._gap_started_at = t0
    # 103, 104 also buffered (still within reorder window)
    assert _seq(buf, 103) == []
    assert _seq(buf, 104) == []
    # Before the window, tick returns []
    assert buf.tick(now=t0 + 0.05) == []
    # At t=200ms after the gap opened, tick skips 101 and releases 102,103,104
    out = buf.tick(now=t0 + 0.20)
    assert out == [b"p102", b"p103", b"p104"]
    assert buf.stats()["lost"] == 1
    assert buf.stats()["gaps"] == 1
    # And we can keep advancing: 105 should release in-order because the
    # buffer was drained and next_expected was advanced to 105.
    assert _seq(buf, 105) == [b"p105"]
    assert buf.stats()["lost"] == 1  # still just one lost


def test_buffered_packets_NOT_discarded_on_loss():
    """Required test: when seq 102 is declared lost, packets 103/104/105
    that are already buffered MUST be preserved and delivered, NOT wiped.
    This is the exact failure mode of the old 64-slot buffer.clear() path.
    """
    buf = JitterBuffer(reorder_window_ms=100)
    _seq(buf, 100)
    _seq(buf, 101)
    # Buffer 103, 104, 105 in order; 102 is missing
    _seq(buf, 103)
    _seq(buf, 104)
    _seq(buf, 105)
    # Pin the gap start so the test is deterministic.
    t0 = 5000.0
    buf._gap_started_at = t0
    # Before timeout, nothing releases
    assert buf.tick(now=t0 + 0.05) == []
    # After timeout: skip exactly 102, release 103, 104, 105 in order
    out = buf.tick(now=t0 + 0.20)
    assert out == [b"p103", b"p104", b"p105"]
    # 102 was the ONLY thing declared lost
    assert buf.stats()["lost"] == 1
    assert buf.stats()["gaps"] == 1
    # And the buffer is empty — nothing was wiped
    assert buf._buffer == {}


def test_old_implementation_would_have_cleared_buffer():
    """Reproduction of the OLD failure mode. We construct a tiny buffer
    size manually (mimicking the old JitterBuffer(max_buffer=64) logic)
    and demonstrate that the OLD approach would discard 103/104/105 when
    102 never arrives. Then we assert the NEW implementation preserves
    them.
    """
    # OLD-style behavior (re-implemented inline for the test):
    class OldStyleBuffer:
        def __init__(self, max_buffer=64):
            self._buffer = {}
            self._next_expected = 0
            self._max_buffer = max_buffer
            self._stats = {"received": 0, "duplicates": 0, "out_of_order": 0,
                           "gaps": 0, "lost": 0}

        def push(self, seq, payload):
            self._stats["received"] += 1
            if seq < self._next_expected:
                self._stats["duplicates"] += 1
                return []
            if seq == self._next_expected:
                self._next_expected += 1
                out = [payload]
                while self._next_expected in self._buffer:
                    out.append(self._buffer.pop(self._next_expected))
                    self._next_expected += 1
                return out
            if seq in self._buffer:
                self._stats["duplicates"] += 1
                return []
            self._buffer[seq] = payload
            self._stats["out_of_order"] += 1
            if len(self._buffer) > self._max_buffer:
                lost = seq - self._next_expected
                self._stats["lost"] += lost
                self._stats["gaps"] += 1
                self._next_expected = seq + 1
                self._buffer.clear()
                return []
            return []

    old = OldStyleBuffer(max_buffer=2)  # tiny cap so overflow is easy to hit
    old.push(100, b"p100")
    old.push(101, b"p101")
    old.push(103, b"p103")
    old.push(104, b"p104")
    # 105 triggers overflow because buffer would be {103,104,105}, len=3 > 2
    old.push(105, b"p105")
    # The OLD code cleared everything and advanced past 101. The bytes
    # for 103/104/105 that were in the buffer were thrown away.
    assert old._buffer == {}, "old style should have wiped the buffer"
    assert old._next_expected == 106, "old style should have jumped past gap"
    assert old._stats["lost"] >= 3, "old style counts 102/103/104/105 as lost"
    # 103 was DISCARDED even though its bytes were physically in memory.

    # Now demonstrate the NEW implementation preserves them.
    new = JitterBuffer(reorder_window_ms=100)
    new.push(100, b"p100")
    new.push(101, b"p101")
    new.push(103, b"p103")
    new.push(104, b"p104")
    new.push(105, b"p105")
    # Pin the gap start so the test is deterministic.
    new._gap_started_at = 9000.0
    # Tick past the reorder window
    out = new.tick(now=9000.0 + 0.20)
    assert out == [b"p103", b"p104", b"p105"], \
        f"NEW implementation must preserve buffered packets, got {out}"
    assert new._buffer == {}, "buffer should be drained but not wiped-empty-then-lost"
    assert new.stats()["lost"] == 1, "only seq=102 should be lost"
    assert new.stats()["gaps"] == 1


# ---------------------------------------------------------------------------
# Late arrivals after the reorder window
# ---------------------------------------------------------------------------

def test_late_arrival_after_loss_dropped():
    """After a seq is declared lost, a late copy of it arriving is a duplicate."""
    buf = JitterBuffer(reorder_window_ms=100)
    _seq(buf, 100)
    _seq(buf, 101)
    _seq(buf, 103)
    # Pin gap-start time so the test is deterministic
    buf._gap_started_at = 7000.0
    # Timeout expires — 102 declared lost, 103 released
    out = buf.tick(now=7000.0 + 0.20)
    assert out == [b"p103"]
    # Now seq 102 finally arrives (late)
    out = _seq(buf, 102)
    assert out == []
    assert buf.stats()["duplicates"] >= 1


# ---------------------------------------------------------------------------
# Sequence-number wraparound
# ---------------------------------------------------------------------------

def test_sequence_wraparound_in_order():
    """Packets arriving through the uint32 wraparound boundary should
    deliver in order."""
    buf = JitterBuffer(reorder_window_ms=100)
    # Adopt a seq near the wrap boundary
    near_max = 0xFFFFFFFF - 2
    assert _seq(buf, near_max) == [f"p{near_max}".encode()]
    assert _seq(buf, near_max + 1) == [f"p{near_max+1}".encode()]
    # Continue through the wrap: next seq is 0xFFFFFFFF, then 0, then 1
    assert _seq(buf, 0xFFFFFFFF) == [f"p{0xFFFFFFFF}".encode()]
    assert _seq(buf, 0) == [b"p0"]
    assert _seq(buf, 1) == [b"p1"]
    assert buf.stats()["gaps"] == 0
    assert buf.stats()["lost"] == 0


def test_sequence_wraparound_with_reorder():
    """A packet right after the wrap boundary arrives before its predecessor."""
    buf = JitterBuffer(reorder_window_ms=100)
    near_max = 0xFFFFFFFF - 1
    assert _seq(buf, near_max) == [f"p{near_max}".encode()]
    # In-order: next expected is 0xFFFFFFFF, send it
    assert _seq(buf, 0xFFFFFFFF) == [f"p{0xFFFFFFFF}".encode()]
    # Now next_expected = 0. Skip 0, buffer seq=1 (gap of 1)
    assert _seq(buf, 1) == []
    # Now the missing seq=0 arrives
    out = _seq(buf, 0)
    assert out == [b"p0", b"p1"]
    assert buf.stats()["gaps"] == 0
    assert buf.stats()["lost"] == 0


def test_sequence_wraparound_late_packet():
    """A packet whose seq is numerically small on the wire but conceptually
    LATE because we've advanced past the wrap boundary."""
    buf = JitterBuffer(reorder_window_ms=100)
    # Adopt seq near wrap boundary
    near_max = 0xFFFFFFF0  # 16 below wrap
    _seq(buf, near_max)
    # Walk forward past the wrap into normal seq space
    for seq in range(near_max + 1, 0xFFFFFFFF):
        out = _seq(buf, seq)
        assert out != [], f"push({seq}) should have released; got {out}"
    # After releasing 0xFFFFFFFE, next_expected = 0xFFFFFFFF.
    # In-order: release 0xFFFFFFFF.
    out = _seq(buf, 0xFFFFFFFF)
    assert out == [f"p{0xFFFFFFFF}".encode()]
    # Now next_expected = 0. We're well past the wrap.
    # A duplicate of near_max (= 0xFFFFFFF0) arrives. Conceptually late by
    # (0xFFFFFFFF - 0xFFFFFFF0) + 1 = 16 packets — way behind us.
    out = _seq(buf, near_max)
    assert out == []
    assert buf.stats()["duplicates"] >= 1
