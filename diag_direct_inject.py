"""Phase 5.1 diagnostic: inject microphone_capture.wav into Phone_Microphone,
record from Phone_Microphone_Input, save to /tmp/direct-injection.wav.

This bypasses UDP entirely. It uses the same pw-cat -p / pw-cat -r mechanism
that Injector / Capture use in production, so the only things exercised are:
  - PCM bytes from a WAV file
  - pw-cat playback into Phone_Microphone
  - PipeWire graph routing
  - pw-cat record from Phone_Microphone_Input

Usage:
    python diag_direct_inject.py <input.wav> [output.wav]

Defaults:
    input  = /home/manav1011/Documents/udp-audio-transport/microphone_capture.wav
    output = /tmp/direct-injection.wav
"""
from __future__ import annotations

import subprocess
import sys
import time
import wave
import os
import threading
import numpy as np

DEFAULT_INPUT = "/home/manav1011/Documents/udp-audio-transport/microphone_capture.wav"
DEFAULT_OUTPUT = "/tmp/direct-injection.wav"


def inspect_wav(path: str) -> dict:
    """Read raw WAVE header (incl. WAVE_FORMAT_IEEE_FLOAT) without using wave."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAVE file")
    i = 12
    while i < len(data) - 8:
        sub_id = data[i:i + 4].decode("ascii", errors="replace")
        sub_size = int.from_bytes(data[i + 4:i + 8], "little")
        if sub_id == "fmt ":
            fmt = data[i + 8:i + 8 + sub_size]
            info = {
                "path": path,
                "audio_format": int.from_bytes(fmt[0:2], "little"),
                "channels": int.from_bytes(fmt[2:4], "little"),
                "sample_rate": int.from_bytes(fmt[4:8], "little"),
                "bits_per_sample": int.from_bytes(fmt[14:16], "little"),
            }
        if sub_id == "data":
            info["data_offset"] = i + 8
            info["data_size"] = sub_size
            info["raw_pcm"] = data[i + 8:i + 8 + sub_size]
            info["frames"] = sub_size // (info["bits_per_sample"] // 8) // info["channels"]
            return info
        i += 8 + sub_size
        if sub_size % 2 == 1:
            i += 1
    raise ValueError("no data chunk found")


def ffmpeg_resample_to_f32(path: str, sr: int = 48000, ch: int = 2) -> bytes:
    """Fallback converter if input isn't already 48k/2/f32. Reads raw float32 LE."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(ch), "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return proc.stdout


def playback_thread(pcm: bytes, target_sink: str, rate: int, channels: int) -> subprocess.Popen:
    """Open pw-cat playback into target sink. Caller feeds stdin."""
    cmd = [
        "pw-cat", "-p",
        "--target", target_sink,
        "--format", "f32",
        "--channels", str(channels),
        "--rate", str(rate),
        "-",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def record_thread(target_source: str, out_wav_path: str, rate: int, channels: int) -> subprocess.Popen:
    """Open pw-cat record from target source, save to a WAV file."""
    cmd = [
        "pw-cat", "-r",
        "--target", target_source,
        "--format", "f32",
        "--channels", str(channels),
        "--rate", str(rate),
        out_wav_path,
    ]
    return subprocess.Popen(cmd, stderr=subprocess.PIPE)


def analyze_rms_peak(pcm: bytes, channels: int = 2) -> dict:
    arr = np.frombuffer(pcm, dtype=np.float32)
    if arr.size == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0.0, "mean": 0.0}
    if arr.size % channels != 0:
        arr = arr[: arr.size - (arr.size % channels)]
    return {
        "samples": int(arr.size),
        "frames": int(arr.size // channels),
        "duration_s": round(arr.size / (channels * 48000), 3),
        "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(arr))),
        "mean": float(np.mean(arr)),
    }


def write_wav_f32(path: str, pcm: bytes, rate: int = 48000, channels: int = 2) -> None:
    """Write raw float32 LE PCM to a WAVE file (format code 3)."""
    import struct
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
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    target_sink = "Phone_Microphone"
    target_source = "Phone_Microphone_Input"

    print("=" * 70)
    print("Phase 5.1 diagnostic — direct injection round-trip")
    print("=" * 70)
    print(f"input WAV     : {in_path}")
    print(f"output WAV    : {out_path}")
    print(f"playback sink : {target_sink}")
    print(f"record source : {target_source}")
    print()

    # Sanity: PipeWire state
    print("--- PipeWire state (sinks) ---")
    print(subprocess.check_output(["pactl", "list", "sinks", "short"]).decode().strip())
    print("--- PipeWire state (sources) ---")
    print(subprocess.check_output(["pactl", "list", "sources", "short"]).decode().strip())
    print()

    if target_sink not in subprocess.check_output(["pactl", "list", "sinks", "short"]).decode():
        print(f"ERROR: {target_sink} not present. Run VirtualAudioManager.start() first.")
        return 1
    if target_source not in subprocess.check_output(["pactl", "list", "sources", "short"]).decode():
        print(f"ERROR: {target_source} not present. Run VirtualAudioManager.start() first.")
        return 1

    # Inspect input WAV
    info = inspect_wav(in_path)
    print(f"--- Input WAV ---")
    print(f"  audio_format   : {info['audio_format']} (3 = IEEE float)")
    print(f"  channels       : {info['channels']}")
    print(f"  sample_rate    : {info['sample_rate']}")
    print(f"  bits_per_sample: {info['bits_per_sample']}")
    print(f"  frames         : {info['frames']}")
    print(f"  duration_s     : {info['frames'] / info['sample_rate']:.3f}")
    src_stats = analyze_rms_peak(info["raw_pcm"], info["channels"])
    print(f"  src_rms        : {src_stats['rms']:.6f}")
    print(f"  src_peak       : {src_stats['peak']:.6f}")
    print(f"  src_mean       : {src_stats['mean']:.6e}")
    print()

    needs_convert = (
        info["audio_format"] != 3
        or info["bits_per_sample"] != 32
        or info["channels"] != 2
        or info["sample_rate"] != 48000
    )
    if needs_convert:
        print("Input is not 48k/2/f32 — converting via ffmpeg...")
        pcm = ffmpeg_resample_to_f32(in_path, sr=48000, ch=2)
        print(f"  converted bytes: {len(pcm)}")
        converted_stats = analyze_rms_peak(pcm, 2)
        print(f"  conv_rms       : {converted_stats['rms']:.6f}")
        print(f"  conv_peak      : {converted_stats['peak']:.6f}")
        print()
    else:
        pcm = info["raw_pcm"]

    # Start recorder first so we don't miss any samples
    print("--- Starting pw-cat recorder from Phone_Microphone_Input ---")
    rec = record_thread(target_source, out_path, 48000, 2)
    time.sleep(0.3)  # let recorder establish

    # Start playback
    print("--- Starting pw-cat playback into Phone_Microphone ---")
    pb = playback_thread(pcm, target_sink, 48000, 2)
    time.sleep(0.2)

    # Feed PCM in chunks
    chunk_size = 48000 * 4 * 2  # 100ms of stereo float32 = 38400 bytes
    bytes_written = 0
    t0 = time.time()
    while bytes_written < len(pcm):
        n = min(chunk_size, len(pcm) - bytes_written)
        try:
            pb.stdin.write(pcm[bytes_written:bytes_written + n])
            pb.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            print(f"playback broken: {e}")
            break
        bytes_written += n
        # Don't outrun the audio too much; aim for real-time-ish
        elapsed = time.time() - t0
        target_time = bytes_written / (48000 * 4 * 2)
        if target_time - elapsed > 0.05:
            time.sleep(target_time - elapsed)

    print(f"--- Played {bytes_written} bytes in {time.time()-t0:.2f}s ---")
    try:
        pb.stdin.close()
        pb.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pb.terminate()
        pb.wait(timeout=2)

    # Give recorder a moment to flush tail
    time.sleep(1.0)
    rec.terminate()
    try:
        rec.wait(timeout=2)
    except subprocess.TimeoutExpired:
        rec.kill()

    # Analyze output
    if not os.path.exists(out_path):
        print("ERROR: output wav was not written")
        return 1

    print()
    print(f"--- Output WAV ({out_path}) ---")
    out_info = inspect_wav(out_path)
    print(f"  size           : {os.path.getsize(out_path)} bytes")
    print(f"  audio_format   : {out_info['audio_format']}")
    print(f"  channels       : {out_info['channels']}")
    print(f"  sample_rate    : {out_info['sample_rate']}")
    print(f"  bits_per_sample: {out_info['bits_per_sample']}")
    print(f"  frames         : {out_info['frames']}")
    print(f"  duration_s     : {out_info['frames'] / out_info['sample_rate']:.3f}")
    out_stats = analyze_rms_peak(out_info["raw_pcm"], out_info["channels"])
    print(f"  out_rms        : {out_stats['rms']:.6f}")
    print(f"  out_peak       : {out_stats['peak']:.6f}")
    print(f"  out_mean       : {out_stats['mean']:.6e}")
    print()

    # Cross-correlation: does the output match the source?
    # The recorder captures some leading silence (a few hundred ms), so the
    # output starts before the source content. We use an FFT-based
    # cross-correlation to find the best alignment, then correlate at that
    # alignment.
    src_arr = np.frombuffer(pcm, dtype=np.float32)
    out_arr = np.frombuffer(out_info["raw_pcm"], dtype=np.float32)
    n = min(len(src_arr), len(out_arr))
    if n > 48000:
        a = src_arr[:n].astype(np.float64) - np.mean(src_arr[:n])
        b = out_arr[:n].astype(np.float64) - np.mean(out_arr[:n])
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

        # Direct correlation at zero lag (no alignment)
        if denom > 0:
            zero_lag_xcorr = float(np.sum(a * b) / denom)
        else:
            zero_lag_xcorr = 0.0

        # FFT-based correlation to find best lag.
        # Cross-correlate as A=output, B=source so positive lag k means
        # out[i] aligns with src[i + k] (i.e., output has leading silence).
        import numpy.fft as fft
        corr_full = fft.ifft(fft.fft(b) * np.conj(fft.fft(a))).real
        # Search positive lags only (output has leading silence, not trailing)
        best_lag = int(np.argmax(np.abs(corr_full[: n // 2])))
        best_xcorr = corr_full[best_lag] / denom if denom > 0 else 0.0

        print(f"--- Correlation ---")
        print(f"  zero-lag xcorr       : {zero_lag_xcorr:.6f}")
        print(f"  best-lag (positive)  : {best_lag} samples = {best_lag/48000*1000:.1f} ms")
        print(f"  best-lag xcorr       : {best_xcorr:.6f}  (1.0 = perfect match)")
        print(f"  alignment            : out[i+{best_lag}] = src[i]")
        print(f"  (output has ~{best_lag/48000*1000:.0f} ms of leading silence before source content)")

    # Match check
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    if out_stats["rms"] > 0 and out_stats["peak"] > 0:
        print(f"output is non-silent: rms={out_stats['rms']:.6f}, peak={out_stats['peak']:.6f}")
    else:
        print("output is silent — injection path broken")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
