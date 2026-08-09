"""Sample-level comparison between two PCM streams (float32 LE, stereo).

The two streams can have:
    - different frame counts (capture trailing silence, leading silence)
    - an arbitrary sample offset between them
    - optional gain / DC offset from the capture path

The comparison does, in order:
    1. Find the best sample offset by FFT cross-correlation on mono mixdowns.
       Restrict the search to ±MAX_LAG_S seconds of offset, near the center.
    2. Slice both streams to the aligned common region.
    3. Compute:
         - frame counts (source / output)
         - aligned offset in frames (and ms)
         - mean absolute error (MAE)
         - RMS error (RMSE)
         - max absolute error
         - Pearson correlation (per channel)
         - exact-equality fraction (per-sample ==, with float32 tolerance)
         - whether the two streams are bit-exact in the aligned region
    4. Also report per-channel and per-segment metrics for each known tone.
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np


SR = 48000
CHANNELS = 2
MAX_LAG_S = 2.0  # search ±2 s of offset (capture-path latency)
EPSILON_EXACT = 0.0  # we treat exact equality bit-for-bit for f32


def read_wav_raw_pcm(path: str) -> tuple[dict, bytes]:
    """Return (header_info, raw_pcm_bytes) from a WAVE file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAVE file")
    info: dict[str, Any] = {"path": path}
    i = 12
    while i < len(data) - 8:
        sub_id = data[i:i + 4].decode("ascii", errors="replace")
        sub_size = int.from_bytes(data[i + 4:i + 8], "little")
        if sub_id == "fmt ":
            fmt = data[i + 8:i + 8 + sub_size]
            info.update({
                "audio_format": int.from_bytes(fmt[0:2], "little"),
                "channels": int.from_bytes(fmt[2:4], "little"),
                "sample_rate": int.from_bytes(fmt[4:8], "little"),
                "bits_per_sample": int.from_bytes(fmt[14:16], "little"),
            })
        if sub_id == "data":
            info["data_offset"] = i + 8
            info["data_size"] = sub_size
            info["raw_pcm"] = data[i + 8:i + 8 + sub_size]
            info["frames"] = sub_size // (info["bits_per_sample"] // 8) // info["channels"]
            return info, info["raw_pcm"]
        i += 8 + sub_size
        if sub_size % 2 == 1:
            i += 1
    raise ValueError("no data chunk found")


def to_stereo_frames(pcm: bytes) -> np.ndarray:
    """Reshape raw PCM bytes to (frames, channels) float32."""
    arr = np.frombuffer(pcm, dtype=np.float32)
    if arr.size % CHANNELS != 0:
        arr = arr[: arr.size - (arr.size % CHANNELS)]
    return arr.reshape(-1, CHANNELS)


def find_offset_frames(src: np.ndarray, out: np.ndarray, max_lag_s: float = MAX_LAG_S) -> int:
    """Find best alignment offset using FFT cross-correlation on mono mixdowns.

    Convention: positive offset means `out` is delayed relative to `src`,
    i.e. out[i] should be compared against src[i + offset].
    Returns offset in frames.
    """
    s = src.mean(axis=1).astype(np.float64)
    o = out.mean(axis=1).astype(np.float64)
    s -= s.mean()
    o -= o.mean()
    n = min(len(s), len(o))
    s = s[:n]
    o = o[:n]
    max_lag = min(int(max_lag_s * SR), n // 2)
    # FFT-based cross-correlation: corr[k] = sum_i o[i] * s[i + k]
    # (positive k => out shifted earlier relative to src => src delayed)
    corr = np.fft.ifft(np.fft.fft(o) * np.conj(np.fft.fft(s))).real
    # Try lags in [-max_lag, +max_lag]
    search = corr.copy()
    # wrap-around: negative lags live at the tail
    cand_pos = corr[:max_lag + 1]
    cand_neg = corr[-max_lag:]
    cand = np.concatenate([cand_neg, cand_pos])
    lag_idx = int(np.argmax(np.abs(cand)))
    if lag_idx < max_lag:
        offset = lag_idx - max_lag  # negative lag
    else:
        offset = lag_idx - max_lag   # positive lag
    return offset


def compare_pcm(src_pcm: bytes, out_pcm: bytes, sr: int = SR,
                segments: list[tuple[float, float, float]] | None = None,
                src_path: str = "?", out_path: str = "?") -> dict:
    src = to_stereo_frames(src_pcm)
    out = to_stereo_frames(out_pcm)
    src_frames = src.shape[0]
    out_frames = out.shape[0]
    print(f"src file   : {src_path}")
    print(f"out file   : {out_path}")
    print(f"src frames : {src_frames}  ({src_frames/sr:.3f}s)")
    print(f"out frames : {out_frames}  ({out_frames/sr:.3f}s)")
    print(f"sample rate: {sr}")
    print(f"channels   : {src.shape[1]} (src), {out.shape[1]} (out)")

    offset = find_offset_frames(src, out)
    print(f"\nbest offset (FFT cross-correlation): {offset} frames "
          f"({offset/sr*1000:.2f} ms)")
    print(f"  (positive => out is delayed relative to src)")

    # Aligned common region.
    # out[i] should align with src[i + offset].
    # We want both src and out slices such that out[k] == src[k + offset].
    # Slice out: k in [k0_out, k1_out), src: in [k0_src, k1_src) with
    # k0_src = k0_out + offset, k1_src = k1_out + offset.
    n_common_out = min(out_frames, max(src_frames - offset, 0))
    n_common_src = min(src_frames, max(out_frames + offset, 0))
    # compute slice bounds
    k0_out = 0
    k1_out = min(out_frames, max(src_frames - offset, 0))
    k0_src = k0_out + offset
    k1_src = k1_src = k0_src + (k1_out - k0_out)
    if k0_src < 0 or k1_src > src_frames:
        # clamp
        k0_out = max(k0_out, -offset)
        k0_src = max(k0_src, 0)
        k1_out = min(k1_out, src_frames - offset)
        k1_src = min(k1_src, src_frames)
    common = min(k1_out - k0_out, k1_src - k0_src)
    if common <= 0:
        print("\nERROR: no aligned overlap (offset exceeds length)")
        return {"offset_frames": offset, "common_frames": 0}

    a = src[k0_src:k0_src + common]
    b = out[k0_out:k0_out + common]

    diff = a - b
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_err = float(abs_diff.max())

    # exact equality: per-sample diff == 0 (f32 bit-exact)
    exact_mask = (diff == 0.0)
    exact_frac = float(exact_mask.mean())

    # correlation per channel and overall
    corrs = []
    for ch in range(min(a.shape[1], b.shape[1])):
        ac = a[:, ch].astype(np.float64) - a[:, ch].mean()
        bc = b[:, ch].astype(np.float64) - b[:, ch].mean()
        denom = np.sqrt((ac ** 2).sum() * (bc ** 2).sum())
        c = float((ac * bc).sum() / denom) if denom > 0 else 0.0
        corrs.append(c)
    # overall mono correlation
    am = a.mean(axis=1).astype(np.float64) - a.mean(axis=1).mean()
    bm = b.mean(axis=1).astype(np.float64) - b.mean(axis=1).mean()
    denom_m = np.sqrt((am ** 2).sum() * (bm ** 2).sum())
    corr_mono = float((am * bm).sum() / denom_m) if denom_m > 0 else 0.0

    print(f"\n--- Aligned common region ---")
    print(f"  common frames : {common}  ({common/sr:.3f}s)")
    print(f"  src slice     : frames [{k0_src}, {k1_src})  "
          f"= [{k0_src/sr:.3f}s, {k1_src/sr:.3f}s)")
    print(f"  out slice     : frames [{k0_out}, {k1_out})  "
          f"= [{k0_out/sr:.3f}s, {k1_out/sr:.3f}s)")
    print(f"\n--- Sample-level errors (aligned) ---")
    print(f"  mean |err|    : {mae:.6e}")
    print(f"  RMS  err      : {rmse:.6e}")
    print(f"  max  |err|    : {max_err:.6e}")
    print(f"  exact-equal % : {exact_frac*100:.4f}%")
    print(f"  correlation   : mono={corr_mono:.6f}  per-ch={corrs}")
    print(f"  bit-exact     : {'YES' if exact_frac == 1.0 else 'NO'}")

    # per-segment metrics
    if segments:
        print(f"\n--- Per-segment metrics ---")
        for (start_s, end_s, freq) in segments:
            f0 = int(start_s * sr)
            f1 = int(end_s * sr)
            # Map segment to aligned region: out frame k corresponds to src
            # frame k+offset. Slice out[k0..k1] with k0 = f0-offset, k1 = f1-offset,
            # then src[k0+offset..k1+offset] = src[f0..f1].
            k0 = max(0, f0 - offset)
            k1 = min(common, f1 - offset)
            if k1 <= k0:
                print(f"  {freq:>5} Hz [{start_s:.2f}-{end_s:.2f}s]: "
                      f"outside aligned region")
                continue
            seg_a = a[k0:k1]
            seg_b = b[k0:k1]
            d = seg_a - seg_b
            seg_mae = float(np.abs(d).mean())
            seg_rmse = float(np.sqrt((d ** 2).mean()))
            seg_max = float(np.abs(d).max())
            seg_exact = float((d == 0).mean())
            seg_rms_a = float(np.sqrt((seg_a ** 2).mean()))
            seg_rms_b = float(np.sqrt((seg_b ** 2).mean()))
            seg_peak_a = float(np.abs(seg_a).max())
            seg_peak_b = float(np.abs(seg_b).max())
            print(f"  {freq:>5} Hz [{start_s:.2f}-{end_s:.2f}s]: "
                  f"MAE={seg_mae:.3e} RMSE={seg_rmse:.3e} "
                  f"max={seg_max:.3e} exact={seg_exact*100:.2f}%  "
                  f"src_rms={seg_rms_a:.4f} out_rms={seg_rms_b:.4f}  "
                  f"src_peak={seg_peak_a:.4f} out_peak={seg_peak_b:.4f}")

    return {
        "src_frames": src_frames,
        "out_frames": out_frames,
        "offset_frames": offset,
        "common_frames": common,
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_err,
        "exact_frac": exact_frac,
        "corr_mono": corr_mono,
        "corr_per_channel": corrs,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3 if False else False:  # placeholder
        pass