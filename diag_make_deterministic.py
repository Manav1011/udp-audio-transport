"""Build a deterministic reference WAV for end-to-end testing.

The signal is constructed sample-by-sample so the exact expected PCM
bytes are known. Both channels carry the same mono waveform (left =
right = waveform).

Layout (5.00 seconds @ 48 kHz stereo float32 LE = 480000 bytes PCM):

    0.000 – 1.000 s  : 440 Hz sine, amplitude 0.50
    1.000 – 1.500 s  : digital silence
    1.500 – 2.500 s  : 880 Hz sine, amplitude 0.50
    2.500 – 3.000 s  : digital silence
    3.000 – 4.000 s  : 1000 Hz sine, amplitude 0.50
    4.000 – 5.000 s  : digital silence

The waveform is generated with explicit per-sample sin() values so that
identical input always produces identical output across platforms and
NumPy versions.

Output: 48 kHz / stereo / float32 LE / format-tag-3 WAVE file.
"""
from __future__ import annotations

import math
import struct
import sys


SR = 48000
CHANNELS = 2
DURATION_S = 5.0
TOTAL_FRAMES = SR * int(DURATION_S)  # 240000 frames

# (start_seconds, end_seconds, frequency_hz)
SEGMENTS = [
    (0.0, 1.0, 440.0),
    (1.5, 2.5, 880.0),
    (3.0, 4.0, 1000.0),
]
AMPLITUDE = 0.50


def build_pcm() -> bytes:
    out = bytearray()
    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / SR
        sample = 0.0
        for start, end, freq in SEGMENTS:
            if start <= t < end:
                # Use a phase reference local to the segment so any
                # boundary discontinuity is visible.
                local_t = t - start
                sample = AMPLITUDE * math.sin(2.0 * math.pi * freq * local_t)
                break
        # Stereo: identical left and right.
        out += struct.pack("<ff", sample, sample)
    return bytes(out)


def write_wav_f32(path: str, pcm: bytes, rate: int = SR, channels: int = CHANNELS) -> None:
    data_size = len(pcm)
    byte_rate = rate * channels * 4
    block_align = channels * 4
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # fmt chunk size
        f.write(struct.pack("<H", 3))   # IEEE float
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 32))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/deterministic-source.wav"
    pcm = build_pcm()
    write_wav_f32(out_path, pcm)
    print(f"wrote {out_path}")
    print(f"  frames={TOTAL_FRAMES}  bytes={len(pcm)}  duration={DURATION_S}s")
    print(f"  segments: {SEGMENTS}")
    print(f"  amplitude: {AMPLITUDE}")
    print(f"  sample_rate={SR}  channels={CHANNELS}  format=IEEE float LE")
    # Sanity: print first 8 samples of each non-silent region.
    import numpy as np
    arr = np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2)
    for label, (start_s, _end_s, freq) in zip(
        ["440Hz", "880Hz", "1000Hz"], SEGMENTS
    ):
        i = int(start_s * SR)
        window = arr[i:i + 8, 0]
        print(f"  {label} @ {start_s}s first 8 left samples: {window.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())