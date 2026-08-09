"""Application configuration — single authoritative source.

The backend reads its entire transport configuration from environment
variables at process start. Every value below is resolved exactly once
here. Nothing else in the production code path should invent defaults
or override these values.

Canonical environment variables and their defaults:

    MICROPHONE_TCP_HOST    default "0.0.0.0"
    MICROPHONE_TCP_PORT    default 5002
    SPEAKER_UDP_HOST       default "127.0.0.1"  (the user MUST set this
                           to the Android device's IP; there is no
                           sensible default for the destination phone)
    SPEAKER_UDP_PORT       default 5000

Contract with the Android app:
    - Android microphone TCP client   ->  <backend_ip>:5002
    - Android speaker   UDP listener  ->  port 5000 on the phone
"""
import os


# --- microphone transport (FINAL: TCP only) ----------------------------------

# The backend listens on this TCP port. The Android mic app is the
# client and connects to <backend_ip>:5002.
MICROPHONE_TCP_HOST = os.environ.get("MICROPHONE_TCP_HOST", "0.0.0.0")
MICROPHONE_TCP_PORT = int(os.environ.get("MICROPHONE_TCP_PORT", "5002"))


# --- speaker transport (UDP) -------------------------------------------------

# The backend's UDP speaker transport sends PC-captured audio (read from
# the application-owned Phone_Speaker sink monitor) to the Android
# speaker app. The Android speaker app is the UDP listener on port 5000.
# SPEAKER_UDP_HOST has no useful default (no Android device on the LAN
# by default); the user must set it to the phone's IP.
SPEAKER_UDP_HOST = os.environ.get("SPEAKER_UDP_HOST", "127.0.0.1")
SPEAKER_UDP_PORT = int(os.environ.get("SPEAKER_UDP_PORT", "5000"))


# --- legacy control plane (kept for app.py, NOT used by audio_main) --------

HOST = os.environ.get("UDP_HOST", "0.0.0.0")
PORT = int(os.environ.get("UDP_PORT", "9812"))


# --- deprecated aliases -----------------------------------------------------
# Kept so existing tests and downstream callers that import these names
# continue to work. They mirror the new canonical values above. New
# production code should use MICROPHONE_TCP_* and SPEAKER_UDP_*.
BIND_HOST = os.environ.get("UDP_BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("UDP_BIND_PORT", "9812"))
DEST_HOST = SPEAKER_UDP_HOST
DEST_PORT = SPEAKER_UDP_PORT
SPEAKER_UDP_DEST_HOST = SPEAKER_UDP_HOST
SPEAKER_UDP_DEST_PORT = SPEAKER_UDP_PORT