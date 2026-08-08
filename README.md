# Audio Transport — Backend (Phase 1)

UDP backend skeleton for a local-network audio bridge. Exchanges
HELLO/HELLO\_ACK, PING/PONG, and MESSAGE packets over UDP.

## Requirements

- Python 3.12
- No external dependencies

## Run

```bash
cd backend
python app.py
```

The server listens on `0.0.0.0:9812` by default. Change `HOST` and `PORT`
in `config.py`.

## Protocol

| Type       | Direction      | Notes                         |
|------------|----------------|-------------------------------|
| HELLO      | Client → Server | Handshake start              |
| HELLO\_ACK | Server → Client | Confirms client address      |
| PING       | Client → Server | Liveness check               |
| PONG       | Server → Client | Echo response                |
| MESSAGE    | Either → Either | Payload is logged            |

Packets are JSON-encoded:

```json
{"type": 3, "payload": {}}
```

`type` uses `PacketType` values:

```
HELLO       = 1
HELLO_ACK   = 2
PING        = 3
PONG        = 4
MESSAGE     = 5
```

## Project layout

```
backend/
├── app.py                  # Entry point, wires the 3 layers
├── config.py               # HOST / PORT
├── transport/
│   └── udp_transport.py    # Raw byte send/recv (no packet knowledge)
├── protocol/
│   ├── packet.py           # Packet class + PacketType enum
│   └── serializer.py       # JSON serialize/deserialize
├── application/
│   └── handler.py          # HELLO/PING/MESSAGE logic
├── audio/
│   ├── audio_manager.py    # Top-level coordinator (Capture + Injector)
│   ├── capture.py          # System audio capture (pw-cat subprocess)
│   └── injector.py         # Microphone injection (pw-cat subprocess)
├── utils/
│   └── logger.py           # Logging setup
├── requirements.txt        # sounddevice, numpy
└── README.md
```

## Audio API

```python
from audio.audio_manager import AudioManager

mgr = AudioManager()

# Set callback for captured PCM bytes
mgr.set_capture_callback(lambda data: send_over_network(data))

# Write PCM bytes to microphone injection
mgr.write_microphone_frames(pcm_bytes)

mgr.start()
# ... later ...
mgr.stop()
```

## Test with a manual client

```bash
echo '{"type":1,"payload":{}}' | nc -u -w1 127.0.0.1 9812
echo '{"type":3,"payload":{}}'   | nc -u -w1 127.0.0.1 9812
```
# udp-audio-transport
