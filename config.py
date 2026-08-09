"""Application configuration — single authoritative source.

The backend reads its entire transport configuration from environment
variables at process start. Every value below is resolved exactly once
here. Nothing else in the production code path should invent defaults
or override these values.

Canonical environment variables and their defaults:

    MICROPHONE_TCP_HOST    default "0.0.0.0"
    MICROPHONE_TCP_PORT    default 5002
    SPEAKER_TCP_HOST       default "127.0.0.1"  (the user MUST set this
                           to the Android device's IP; there is no
                           sensible default for the destination phone)
    SPEAKER_TCP_PORT       default 5000

FINAL ARCHITECTURE — both transports are TCP:

    Microphone:  Android -> TCP :5002 -> Backend -> Phone_Microphone
    Speaker:     Backend  -> TCP :5000 -> Android speaker app

Mic and speaker each have a dedicated TCP connection on a separate port;
they are not multiplexed. The previous UDP speaker transport has been
removed from the production path.
"""
import os


# --- microphone transport (TCP) --------------------------------------------

# The backend listens on this TCP port. The Android mic app is the
# client and connects to <backend_ip>:5002.
MICROPHONE_TCP_HOST = os.environ.get("MICROPHONE_TCP_HOST", "0.0.0.0")
MICROPHONE_TCP_PORT = int(os.environ.get("MICROPHONE_TCP_PORT", "5002"))


# --- speaker transport (TCP) -----------------------------------------------

# The backend's TCP speaker transport forwards PC-captured audio (read
# from the application-owned Phone_Speaker sink monitor) to the Android
# speaker app. The Android speaker app is the TCP server on port 5000;
# the backend connects as a TCP client.
#
# SPEAKER_TCP_HOST has no useful default (no Android device on the LAN
# by default); the user must set it to the phone's IP.
SPEAKER_TCP_HOST = os.environ.get("SPEAKER_TCP_HOST", "127.0.0.1")
SPEAKER_TCP_PORT = int(os.environ.get("SPEAKER_TCP_PORT", "5000"))


# --- legacy control plane (kept for app.py, NOT used by audio_main) --------

HOST = os.environ.get("UDP_HOST", "0.0.0.0")
PORT = int(os.environ.get("UDP_PORT", "9812"))


# --- deprecated aliases (kept so old tests still import them) --------------
# These mirror SPEAKER_TCP_HOST/PORT. New production code MUST use the
# canonical SPEAKER_TCP_* names.
BIND_HOST = os.environ.get("UDP_BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("UDP_BIND_PORT", "9812"))
DEST_HOST = SPEAKER_TCP_HOST
DEST_PORT = SPEAKER_TCP_PORT
SPEAKER_UDP_DEST_HOST = SPEAKER_TCP_HOST
SPEAKER_UDP_DEST_PORT = SPEAKER_TCP_PORT
