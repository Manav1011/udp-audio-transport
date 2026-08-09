"""Post-capture analysis of /tmp/android_raw_udp.pcm.

Reads the raw PCM bytes recorded by diagnostics/raw_android_udp_receiver.py
and reports objective properties: byte/frame counts, RMS, peak, mean,
min/max, NaN/Inf counts, zero-sample percentage, and a frequency spectrum
(L/R channels).

PCM format is fixed by the protocol:
  - 48000 Hz
  - stereo
  - float32 little-endian
  - interleaved [L, R, L, R, ...]

This script does NOT modify the file. It mmaps the data for efficient
random access and reads only the bytes it needs for the spectrum.

Usage:
    python -m diagnostics.analyze_raw_pcm
    python -m diagnostics.analyze_raw_pcm --path /tmp/android_raw_udp.pcm
    python -m diagnostics.analyze_raw_pcm --top 20
"""
from __future__ import annotations

import argparse
import math
import mmap
import struct
import sys

# Constants mirror transport/audio_packet.py and the protocol spec.
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 4  # float32
FRAME_BYTES = CHANNELS * SAMPLE_WIDTH  # 8 bytes per stereo frame

DEFAULT_PATH = "/tmp/android_raw_udp.pcm"
NAN = float("nan")
INF = float("inf")


def _format(x: float) -> str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf"
    return f"{x:.6f}"


def _dominant_frequencies(mono: list[float], top: int) -> list[tuple[float, float]]:
    """Compute the top-N frequency components of a mono signal using a
    simple FFT (numpy). Returns [(freq_hz, magnitude), ...] sorted by
    magnitude desc.

    If numpy is not available, falls back to a DFT on the first 4096
    samples (slow but functional).
    """
    n = len(mono)
    window = mono[:4096] if n >= 4096 else mono
    if not window:
        return []
    try:
        import numpy as np  # type: ignore
        arr = np.asarray(window, dtype=np.float32)
        # Hann window for cleaner spectrum.
        window_fn = np.hanning(len(arr))
        windowed = arr * window_fn
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / SAMPLE_RATE)
        # Skip DC (bin 0) when looking for dominant tones.
        order = np.argsort(mag[1:])[::-1][:top] + 1
        return [(float(freqs[i]), float(mag[i])) for i in order]
    except ImportError:
        # Fallback: naive DFT of first 1024 samples. Very slow but works.
        n_dft = min(1024, len(window))
        out: list[tuple[float, float]] = []
        for k in range(1, n_dft // 2):
            re = 0.0
            im = 0.0
            for t in range(n_dft):
                a = -2.0 * math.pi * k * t / n_dft
                re += window[t] * math.cos(a)
                im += window[t] * math.sin(a)
            mag = math.sqrt(re * re + im * im)
            freq = k * SAMPLE_RATE / n_dft
            out.append((freq, mag))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top]


def analyze(path: str, top: int) -> int:
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            # Empty file.
            print(f"file: {path}")
            print("  (empty)")
            return 0
        try:
            size = mm.size()
            total_bytes = size
            total_frames = total_bytes // FRAME_BYTES
            trailing_bytes = total_bytes - total_frames * FRAME_BYTES
            duration_s = total_frames * FRAME_BYTES / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
            print(f"file:                   {path}")
            print(f"sample rate:            {SAMPLE_RATE}")
            print(f"channels:               {CHANNELS}")
            print(f"sample width:           {SAMPLE_WIDTH} bytes (float32)")
            print(f"frame bytes:            {FRAME_BYTES}")
            print(f"total bytes:            {total_bytes}")
            print(f"total frames:           {total_frames}")
            print(f"trailing partial bytes: {trailing_bytes}")
            print(f"duration:               {duration_s:.3f} s")
            if total_frames == 0:
                print("(no frames to analyze)")
                return 0
            # Iterate samples.
            n_samples = total_frames * CHANNELS
            left: list[float] = []
            right: list[float] = []
            n_nan = 0
            n_inf = 0
            n_zero = 0
            l_sum = 0.0
            r_sum = 0.0
            l_sq_sum = 0.0
            r_sq_sum = 0.0
            l_min = +INF
            l_max = -INF
            r_min = +INF
            r_max = -INF
            l_peak = 0.0
            r_peak = 0.0
            for i in range(0, n_samples * SAMPLE_WIDTH, FRAME_BYTES):
                l, r = struct.unpack_from("<ff", mm, i)
                if math.isnan(l):
                    n_nan += 1
                if math.isinf(l):
                    n_inf += 1
                if math.isnan(r):
                    n_nan += 1
                if math.isinf(r):
                    n_inf += 1
                if l == 0.0:
                    n_zero += 1
                if r == 0.0:
                    n_zero += 1
                l_sum += l
                r_sum += r
                l_sq_sum += l * l
                r_sq_sum += r * r
                if l < l_min:
                    l_min = l
                if l > l_max:
                    l_max = l
                if r < r_min:
                    r_min = r
                if r > r_max:
                    r_max = r
                la = -l if l < 0 else l
                ra = -r if r < 0 else r
                if la > l_peak:
                    l_peak = la
                if ra > r_peak:
                    r_peak = ra
                left.append(l)
                right.append(r)
            n = len(left)
            n_right = len(right)
            assert n_right == n
            l_mean = l_sum / n
            r_mean = r_sum / n
            l_rms = math.sqrt(l_sq_sum / n)
            r_rms = math.sqrt(r_sq_sum / n)
            pct_zero = 100.0 * n_zero / (n * 2)
            print()
            print(f"samples (L, R):         {n} , {n_right}")
            print(f"NaN count:              {n_nan}")
            print(f"Inf count:              {n_inf}")
            print(f"zero-sample pct:        {pct_zero:.4f}%")
            print()
            print("Left channel")
            print(f"  peak:  {_format(l_peak)}")
            print(f"  RMS:   {_format(l_rms)}")
            print(f"  mean:  {_format(l_mean)}")
            print(f"  min:   {_format(l_min)}")
            print(f"  max:   {_format(l_max)}")
            print("Right channel")
            print(f"  peak:  {_format(r_peak)}")
            print(f"  RMS:   {_format(r_rms)}")
            print(f"  mean:  {_format(r_mean)}")
            print(f"  min:   {_format(r_min)}")
            print(f"  max:   {_format(r_max)}")
            # Combine L/R for spectrum analysis (mono mix).
            mono = [(l + r) * 0.5 for l, r in zip(left, right)]
            peaks = _dominant_frequencies(mono, top)
            print()
            print(f"Top {min(top, len(peaks))} frequency components (mono mix, "
                  f"first ~4096 frames)")
            if not peaks:
                print("  (no spectrum)")
            for freq, mag in peaks:
                print(f"  {freq:8.2f} Hz  magnitude={mag:.4f}")
            # Control test expectation.
            print()
            print("Control test reference (440 Hz sine, amplitude 0.25, "
                  "stereo L=R):")
            print("  peak:           0.250000")
            print("  RMS:            ~0.176777 (= 0.25 / sqrt(2))")
            print("  dominant freq:  440 Hz")
            print("If your analyzer numbers match exactly, the raw PCM is a "
                  "clean 440 Hz tone.")
            print("If peak is ~0.5 and RMS is ~0.3536, the Android test "
                  "tone is at amplitude 0.5 instead of 0.25.")
        finally:
            mm.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze raw PCM recorded by raw_android_udp_receiver.py"
    )
    parser.add_argument(
        "--path", default=DEFAULT_PATH,
        help=f"PCM file path (default {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many top frequency components to report (default 10)",
    )
    args = parser.parse_args()
    return analyze(args.path, args.top)


if __name__ == "__main__":
    sys.exit(main())
