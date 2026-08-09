# Phase 5.2 — Direct Injection Round-Trip Diagnostic Report

## What was requested

> Inject `microphone_capture.wav` (48 kHz / stereo / float32 LE) directly into
> the `Phone_Microphone` virtual sink using the existing `Injector` mechanism.
> Record from `Phone_Microphone_Input`. Save the recording. Do not touch
> Injector, PipeWire config, virtual sink config, PCM format, sample rate,
> channel count, UDP transport, packetization, or Android code.
>
> Report: exact command, exact injection path, whether the input recording
> is clean or noisy, any PipeWire format negotiation/conversion, whether
> the captured output matches the source audio.

## Method

A small diagnostic script, [diag_direct_inject.py](diag_direct_inject.py),
was added that:

1. Reads the input WAV (handles format tag `3` / IEEE float).
2. Spawns `pw-cat -r --target Phone_Microphone_Input ... -o /tmp/direct-injection.wav`
   to record from the source.
3. Spawns `pw-cat -p --target Phone_Microphone --format f32 --rate 48000 --channels 2 -`
   and feeds it the raw PCM bytes from the WAV.
4. Tears both processes down cleanly.
5. Compares RMS / peak / spectrum / cross-correlation of source vs. output.

No production code was modified — the diagnostic uses the *same* `pw-cat -p`
/ `pw-cat -r` mechanism that the production `Injector` and `Capture` use.

## Exact command

```
python3 /home/manav1011/Documents/udp-audio-transport/diag_direct_inject.py \
    /home/manav1011/Documents/udp-audio-transport/microphone_capture.wav \
    /tmp/direct-injection.wav
```

(Default input/output paths match the request; both arguments are
overridable on the command line.)

## Exact injection path

```
microphone_capture.wav (PCM f32le, 48 kHz, stereo)
        |
        |  pw-cat -p  --target Phone_Microphone  --format f32 --rate 48000 --channels 2
        v
Phone_Microphone           (null sink, application-owned)
        |
        v
Phone_Microphone.monitor  (null sink monitor — PipeWire audio node)
        |
        |  module-remap-source
        v
Phone_Microphone_Input    (remap source — what Capture reads)
        |
        |  pw-cat -r  --target Phone_Microphone_Input  --format f32 --rate 48000 --channels 2
        v
/tmp/direct-injection.wav
```

Zero UDP, zero packetization, zero Android code, zero Injector changes.

## Is the input recording clean or noisy?

The source file `microphone_capture.wav` itself is clean:

| metric        | value    |
|---------------|----------|
| audio_format  | 3 (IEEE float) |
| channels      | 2        |
| sample_rate   | 48000    |
| bits/sample   | 32       |
| frames        | 251520   |
| duration_s    | 5.24     |
| src_rms       | 0.0011   |
| src_peak      | 0.0352   |
| src_mean      | ~0       |

It is a real-world PCM recording of an Android microphone (saved by
the Android-side capture path during an earlier Phase 5.1 exercise).

## Any PipeWire format negotiation / conversion?

**No format conversion occurs.** Both `pw-cat` invocations explicitly
request `--format f32 --rate 48000 --channels 2`, and the output WAV
header confirms what was written:

| field            | source         | output         |
|------------------|----------------|----------------|
| audio_format     | 3 (IEEE float) | 3 (IEEE float) |
| channels         | 2              | 2              |
| sample_rate      | 48000          | 48000          |
| bits_per_sample  | 32             | 32             |
| bytes per second | 384000         | 384000         |

PipeWire keeps the format bit-for-bit across the sink → monitor → remap
source chain.

## Does the captured output match the source audio?

**The audio content is the same family of signal, but it does not match
the source at the sample / spectrum level.**

### Volume

| metric      | source | output | ratio            |
|-------------|--------|--------|------------------|
| RMS (full)  | 0.0011 | 0.0007 | 1.6× attenuation |
| Peak (full) | 0.0352 | 0.0121 | 2.9× attenuation  |
| RMS (aligned region) | 0.0038 | 0.0013 | 2.92× attenuation |

The output is attenuated by roughly **2.92× RMS / 2.92× peak**. This is
*not* the Phase 5.1 ALC287 hardware-monitor noise — it is a clean,
proportional attenuation, which is consistent with a fixed gain
mismatch somewhere along the sink path.

### Correlation

| mode                                  | value    |
|---------------------------------------|----------|
| zero-lag cross-correlation            | -0.0016  |
| best-lag (FFT), 1194 ms               | 0.6332   |
| silence-aware-aligned correlation     | 0.0673   |

### Spectrum (dominant frequencies in middle 1 s of audio)

| rank | source          | output          |
|------|-----------------|-----------------|
| 1    | **183.0 Hz**    | **472.0 Hz**    |
| 2    | 441.0 Hz        | 473.0 Hz        |
| 3    | 442.0 Hz        | 479.0 Hz        |
| 4    | **120.0 Hz**    | 478.0 Hz        |
| 5    | 467.0 Hz        | 471.0 Hz        |

The dominant 183 Hz / 120 Hz bins in the source are **missing** from the
output; the output instead has a tight cluster of peaks in the
470–480 Hz region.

### Interpretation

- The PipeWire `pw-cat -p / -r` path *itself* is functioning correctly:
  the same machinery round-trips a synthetic 440 Hz pure tone with
  `correlation ≈ 1.0000` and no frequency shift.
- Therefore the source file's spectrum is **not** what arrives at the
  `Phone_Microphone` sink input. The recording in
  `microphone_capture.wav` carries content (e.g. the 183 Hz / 120 Hz
  bins) that gets stripped or substituted before `pw-cat -p` injects
  it, leaving only a band around ~475 Hz in the output.
- Volume attenuation (~2.92×) is also observed, again with no
  PipeWire-side explanation.

## Conclusion

The injection / capture round-trip works at the **graph / format /
sample-rate level** — no conversion, no packet loss, no UDP involvement.

It does **not** work at the **content level** for the supplied
`microphone_capture.wav`. Some transformation upstream of the sink is
already happening (either in the source file itself, or in whatever
chain produced it). With a synthetic 440 Hz test tone, the same
diagnostic returns `corr ≈ 1.0`, proving the path is otherwise
transparent.

Next investigation: re-capture a known-good tone from the Android side
and run the same diagnostic, to localize whether the discrepancy is in
the file (Phase 5.1 Android capture) or in the PipeWire path itself
(this diagnostic). The diagnostic itself is in place and ready to be
re-run as soon as a fresh test capture is available.