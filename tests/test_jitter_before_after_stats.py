"""Simulated reorder/loss benchmark for the jitter buffer.

Runs the SAME packet stream through the OLD jitter-buffer implementation
(max_buffer=64, buffer.clear() on overflow) and the NEW implementation
(reorder_window_ms=200, per-seq skip). Reports the resulting stats so we
can see the difference in behavior on a realistic workload.
"""
import random
import time

from transport.audio_receiver import JitterBuffer


class OldStyleJitterBuffer:
    """Re-implementation of the OLD jitter buffer for comparison."""

    def __init__(self, max_buffer: int = 64):
        self._buffer: dict[int, bytes] = {}
        self._next_expected: int = 0
        self._max_buffer = max_buffer
        self._stats = {
            "received": 0, "duplicates": 0, "out_of_order": 0,
            "gaps": 0, "lost": 0,
        }

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

    def stats(self):
        return dict(self._stats)


def simulate_workload(reorder_window_ms=200, seed=42):
    """Send 5066 packets (matching the real session) with reorder windows
    up to 16 packets (Wi-Fi scale), 1% drop, occasional late dupes.

    Sender starts at seq=0 (Android behavior). Packet arrival interval is
    ~6 ms (167 pps for 48 kHz mono audio at MAX_PAYLOAD=1152).
    """
    random.seed(seed)
    n_sent = 5066
    sent = []
    i = 0
    while i < n_sent:
        # 1% drop probability
        if random.random() < 0.01:
            i += 1
            continue
        sent.append(i)
        i += 1
    # Apply reordering: shuffle windows of 16 packets with 50% probability.
    # (Skips the first packet so the sender's seq=0 lands first.)
    arriving = []
    if sent:
        arriving.append(sent[0])
    j = 1
    while j < len(sent):
        window = sent[j:j+16]
        if random.random() < 0.5 and len(window) >= 2:
            random.shuffle(window)
        arriving.extend(window)
        j += 16
    # 0.5% duplicate arrivals (late resends)
    duplicates = random.sample(arriving, k=max(1, len(arriving) // 200))
    arriving.extend(duplicates)

    INTER_ARRIVAL_S = 0.006  # 6 ms per packet = 167 pps

    # Run OLD buffer
    old = OldStyleJitterBuffer(max_buffer=64)
    old_delivered = 0
    for seq in arriving:
        out = old.push(seq, b"X")
        old_delivered += len(out)

    # Run NEW buffer with simulated time advancing at INTER_ARRIVAL_S per packet.
    # We monkey-patch time.monotonic via the module attribute so push() and
    # tick() agree on the same simulated timeline.
    import transport.audio_receiver as _rx_mod
    real_monotonic = _rx_mod.time.monotonic
    sim_time = [10000.0]
    _rx_mod.time.monotonic = lambda: sim_time[0]
    try:
        new = JitterBuffer(reorder_window_ms=reorder_window_ms)
        new_delivered = 0
        for seq in arriving:
            sim_time[0] += INTER_ARRIVAL_S
            out = new.push(seq, b"X")
            new_delivered += len(out)
            while True:
                ready = new.tick(now=sim_time[0])
                if not ready:
                    break
                new_delivered += len(ready)
        # Final drain
        sim_time[0] += 1.0
        while True:
            ready = new.tick(now=sim_time[0])
            if not ready:
                break
            new_delivered += len(ready)
    finally:
        _rx_mod.time.monotonic = real_monotonic

    return {
        "packets_sent": n_sent,
        "packets_arrived": len(arriving),
        "duplicates_injected": len(duplicates),
        "old": {
            "delivered": old_delivered,
            "stats": old.stats(),
        },
        "new": {
            "delivered": new_delivered,
            "stats": new.stats(),
        },
    }


def main():
    result = simulate_workload()
    print("=" * 70)
    print(" Jitter Buffer Before/After Benchmark (deterministic, seed=42)")
    print("=" * 70)
    print(f" packets sent:                 {result['packets_sent']}")
    print(f" packets arrived:              {result['packets_arrived']}")
    print(f" duplicates injected:          {result['duplicates_injected']}")
    print()
    print(" OLD (max_buffer=64, buffer.clear() on overflow):")
    for k, v in result["old"]["stats"].items():
        print(f"   {k:<14} {v}")
    print(f"   {'delivered':<14} {result['old']['delivered']}")
    print()
    print(" NEW (reorder_window=200ms, per-seq skip, buffer preserved):")
    for k, v in result["new"]["stats"].items():
        print(f"   {k:<14} {v}")
    print(f"   {'delivered':<14} {result['new']['delivered']}")
    print()
    old_lost = result["old"]["stats"]["lost"]
    new_lost = result["new"]["stats"]["lost"]
    diff = old_lost - new_lost
    print(f" PCM packets saved by new impl: {diff} "
          f"({(diff/old_lost*100 if old_lost else 0):.0f}% of old loss)")
    print(f" Continuity of delivery stream:")
    print(f"   OLD: {result['old']['stats']['gaps']} forced buffer wipes")
    print(f"   NEW: {result['new']['stats']['gaps']} per-seq gaps (no wipes)")


if __name__ == "__main__":
    main()
