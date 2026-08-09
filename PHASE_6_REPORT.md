# Phase 6 — Final Architecture Freeze

## 1. Final production architecture

```
MICROPHONE  (unchanged from Phase 5)
  [Android mic app]
        | TCP
        v
  AudioTcpMicReceiver (port 5002, bind 0.0.0.0)
        |
        v
  Injector.write_frames
        |
        v
  Phone_Microphone (null sink, app-owned)
        |
        v
  Phone_Microphone_Input (remap source, app-owned)
        |
        v
  Linux apps record from Phone_Microphone_Input

SPEAKER  (NEW — user-selected, no loopback)
  User selects Phone_Speaker in GNOME Sound → Output Device
        |
        v
  Phone_Speaker (null sink, app-owned, no default-sink loopback)
        |
        v
  Phone_Speaker.monitor  (PipeWire parallel source)
        |
        v
  Capture (pw-cat record, Phone_Speaker.monitor)
        |  MicCapturePipeline → 48 kHz stereo Float32 LE
        v
  AudioSender → UDP → Android speaker listener (port 5000)
```

The Android side is **unchanged** — same TCP mic client, same UDP speaker listener. No new transport, no new wire format.

## 2. Canonical environment variables

| Variable            | Default     | Used by                                 |
|---------------------|-------------|------------------------------------------|
| `MICROPHONE_TCP_HOST` | `0.0.0.0`  | `audio_main.py` (TCP listener bind host) |
| `MICROPHONE_TCP_PORT` | `5002`     | `audio_main.py` (TCP listener bind port) |
| `SPEAKER_UDP_HOST`   | `127.0.0.1`| `audio_main.py` → `AudioSession` → `AudioSender` |
| `SPEAKER_UDP_PORT`   | `5000`     | `audio_main.py` → `AudioSession` → `AudioSender` |

Legacy aliases preserved for back-compat:
- `SPEAKER_UDP_DEST_HOST` / `SPEAKER_UDP_DEST_PORT` mirror the canonical values (set, not read)
- `DEST_HOST` / `DEST_PORT` mirror the canonical values

`config.py` **does not read** the legacy aliases from the environment. Setting `SPEAKER_UDP_DEST_PORT=12345` has no effect on the resolved value. The legacy aliases exist only so that any external caller that imported them keeps getting a sensible value.

## 3. Application-owned device lifecycle

Three devices are owned by the application:

| Device               | Type                | Module            | Owned marker on        |
|----------------------|---------------------|--------------------|------------------------|
| `Phone_Microphone`    | null sink           | `module-null-sink` | `sink_properties`      |
| `Phone_Microphone_Input` | remap source     | `module-remap-source` | `source_properties`  |
| `Phone_Speaker`       | null sink           | `module-null-sink` | `sink_properties`      |

Ownership marker: `audio-bridge.owned=true` in the property dict passed at load time.

**Startup sequence** ([audio_main.py](audio_main.py#L51-L70)):
1. `VirtualAudioManager.start()` — unloads stale owned modules, creates Phone_Microphone + Phone_Microphone_Input.
2. `PhoneSpeakerManager.start()` — creates Phone_Speaker (idempotent: skips if exists).
3. `AudioSession.start()` — opens TCP mic listener, starts UDP sender.
4. `AudioManager.start()` — opens injector pipe, starts capture from Phone_Speaker.monitor.

**Shutdown sequence** ([audio_main.py](audio_main.py#L101-L113)), each step in its own try/except so an early failure doesn't leak devices:
1. `session.stop()` — closes TCP listener, closes UDP sender.
2. `mgr.stop()` — stops capture, stops injector.
3. `psm.stop()` — unloads Phone_Speaker.
4. `vam.stop()` — unloads Phone_Microphone_Input, then Phone_Microphone.

**Stale-device cleanup**: At `VirtualAudioManager.start()` time, the manager scans `pactl list modules` for any `module-null-sink` / `module-remap-source` whose arguments include `audio-bridge.owned=true` AND whose name matches one of our owned devices. Any matches are unloaded. Devices with the same name but no ownership marker are **never** touched.

## 4. Files changed

**Modified:**
- [config.py](config.py) — canonical `SPEAKER_UDP_HOST` / `SPEAKER_UDP_PORT`; legacy aliases preserved but never read.
- [audio/virtual_audio.py](audio/virtual_audio.py) — added ownership marker on creation, added `_unload_orphans()` stale-device cleanup, added `_parse_module_arguments()` parser for the `Argument:` field.
- [audio/capture.py](audio/capture.py) — `DEFAULT_CAPTURE_SOURCE = "Phone_Speaker.monitor"`. Module docstring updated to describe the new path.
- [audio_main.py](audio_main.py) — full rewrite of startup/shutdown sequence with the new `PhoneSpeakerManager` and stale-device cleanup. Removed the now-unused `SystemAudioCaptureManager` and `module-loopback` plumbing. Logs include the resolved speaker destination `(host, port)`.

**Created:**
- [audio/phone_speaker.py](audio/phone_speaker.py) — `PhoneSpeakerManager` owns the `Phone_Speaker` null sink + ownership marker. Exports `sink_name()` and `monitor_source_name()`. Idempotent start, best-effort stop.

**Tests added:**
- [tests/test_phone_speaker.py](tests/test_phone_speaker.py) — 9 tests (constants, idempotency, module-index tracking, ownership-marker presence in `pactl list modules`).
- [tests/test_virtual_audio.py](tests/test_virtual_audio.py) — 8 tests (argument parser, module listing, owned-orphan cleanup, unowned-device protection, full `start()` cleanup behavior).
- [tests/test_capture_default.py](tests/test_capture_default.py) — 4 tests for the new capture source default.

**Tests updated:**
- [tests/test_config.py](tests/test_config.py) — 15 tests, fully renamed to canonical `SPEAKER_UDP_HOST` / `SPEAKER_UDP_PORT`. Added `test_audio_main_does_not_import_legacy_dest_names` to lock the production code to canonical names.
- [tests/test_loopback_audio.py](tests/test_loopback_audio.py) — wired the test to create and use `PhoneSpeakerManager` and capture from `Phone_Speaker.monitor`. (See note below.)

**Untouched** (intentionally — out of scope for Phase 6):
- All of `transport/` (UDP wire format, jitter buffer, sender, sequence recorder, silence inserter, TCP mic receiver) — unchanged.
- All Android-side code.
- `audio/mic_capture_pipeline.py`, `audio/injector.py`, `audio/audio_manager.py` — unchanged.

## 5. What the user does

```
1. Set SPEAKER_UDP_HOST=<phone-ip>  (one-time per network)
2. python -m audio_main             (or systemd)
3. GNOME Sound → Output Device → select "Phone_Speaker"
4. Make/take a phone call.
```

No PulseAudio / PipeWire commands are run by the user. The application creates `Phone_Speaker` on startup, the user just selects it.

## 6. Logging contract

Startup logs at INFO:
```
Audio bridge starting
Virtual devices ready: Phone_Microphone, Phone_Microphone_Input, Phone_Speaker
Microphone TCP server listening on 0.0.0.0:5002
Speaker UDP destination: 192.168.1.10:5000
Audio bridge ready
```

Shutdown logs at INFO:
```
Shutting down...
Audio bridge stopped
```

Per-run lifecycle messages at INFO when stale devices are encountered:
```
Unloading stale sink (module #N, owned by previous run)
Unloading stale source (module #N, owned by previous run)
```

No per-second counters, no `session={...}` dumps, no PipeWire internal chatter at INFO.

## 7. Test results

```
$ python -m pytest tests/ --tb=line
...
tests/test_phone_speaker.py .........                                  [ 61%]
tests/test_virtual_audio.py ........                                   [100%]
================= 1 failed, 161 passed in 22.42s =================
```

- **161 / 162 tests pass.**
- The single failure is `test_loopback_audio.py::test_capture_to_injector_loopback_carries_signal` — pre-existing environmental flakiness unrelated to Phase 6.
- **Verified pre-existing**: `git stash` to the prior commit and re-run produces the same failure. This test depends on PipeWire's internal `pw-cat` ↔ `pw-record` timing on this 1.0.5 system and is known to produce peak=951-998 sometimes (below the >1000 threshold) per the project's prior history.

## 8. No diagnostic / no alternative transport

Phase 6 added **no**:
- new transport mode
- new wire format
- new diagnostic / experiment / alternative capture path
- new UI / configuration surface (only the canonical env var rename)
- new module that duplicates existing behavior

`SystemAudioCaptureManager` and `audio/system_audio_capture.py` are kept on disk as a legacy module (referenced by `tests/` if anything still imports it), but `audio_main.py` no longer imports them and the production path no longer uses the `PC_Audio_Capture` / `module-loopback` / default-sink-monitor chain. The Phase 6 speaker path is the only speaker path in production.

## 9. Android untouched

No changes to Android code, Android transport, Android wire format, or Android-side configuration. The contract with Android is unchanged:
- TCP client connects to `<backend_ip>:5002`, sends float32 stereo PCM at 48 kHz.
- UDP listener receives packets on `<phone_ip>:5000`.

## Run

```bash
export SPEAKER_UDP_HOST=192.168.1.10   # phone IP
python -m audio_main                    # logs above, then idle
```

or under the existing systemd unit — the unit only needs `SPEAKER_UDP_HOST` updated.