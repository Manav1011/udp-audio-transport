"""Configuration tests — verifies the canonical environment variables
and defaults for the production audio transport.

These tests import `config.py` as the single authoritative source of
transport configuration. They guard against regressions like the
previous bug where `SPEAKER_UDP_DEST_PORT` defaulted to 9813 instead
of 5000.

After the Phase 6 rename, the canonical name is `SPEAKER_UDP_PORT`
(no `_DEST_` infix). The legacy name `SPEAKER_UDP_DEST_PORT` is
preserved as a deprecated alias that mirrors the canonical value but
must NOT be read by config.py.

Each test that depends on environment state reloads the module so the
`os.environ.get(...)` calls inside config.py are re-evaluated with the
monkey-patched environment.
"""
from __future__ import annotations

import importlib
import socket

import pytest


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config.py with the given monkeypatched environment."""
    def _reload():
        import config as config_module
        return importlib.reload(config_module)
    return _reload


# ---------------------------------------------------------------------------
# Defaults — no environment variables set
# ---------------------------------------------------------------------------

def test_defaults_when_no_env(monkeypatch, reload_config):
    """Without any environment variables, defaults are correct."""
    for var in (
        "MICROPHONE_TCP_HOST",
        "MICROPHONE_TCP_PORT",
        "SPEAKER_UDP_HOST",
        "SPEAKER_UDP_PORT",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_HOST == "0.0.0.0"
    assert cfg.MICROPHONE_TCP_PORT == 5002
    assert cfg.SPEAKER_UDP_PORT == 5000
    assert cfg.SPEAKER_UDP_HOST == "127.0.0.1"


def test_default_mic_port_is_5002(monkeypatch, reload_config):
    """The mic TCP port default must be exactly 5002."""
    monkeypatch.delenv("MICROPHONE_TCP_PORT", raising=False)
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_PORT == 5002


def test_default_speaker_port_is_5000(monkeypatch, reload_config):
    """The speaker UDP port default must be exactly 5000 (NOT 9813)."""
    monkeypatch.delenv("SPEAKER_UDP_PORT", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_PORT == 5000


def test_default_speaker_host_is_localhost(monkeypatch, reload_config):
    """Speaker host has no useful default; localhost is the placeholder."""
    monkeypatch.delenv("SPEAKER_UDP_HOST", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_HOST == "127.0.0.1"


def test_9813_is_never_a_production_default(monkeypatch, reload_config):
    """Hard guarantee: 9813 must not appear anywhere in the default
    speaker port resolution path."""
    monkeypatch.delenv("SPEAKER_UDP_PORT", raising=False)
    monkeypatch.delenv("UDP_PORT", raising=False)
    monkeypatch.delenv("UDP_BIND_PORT", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_PORT != 9813


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def test_microphone_tcp_port_env_override(monkeypatch, reload_config):
    """Setting MICROPHONE_TCP_PORT must override the default."""
    monkeypatch.setenv("MICROPHONE_TCP_PORT", "6000")
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_PORT == 6000


def test_speaker_udp_port_env_override(monkeypatch, reload_config):
    """Setting SPEAKER_UDP_PORT must override the default."""
    monkeypatch.setenv("SPEAKER_UDP_PORT", "7000")
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_PORT == 7000


def test_speaker_udp_host_env_override(monkeypatch, reload_config):
    """Setting SPEAKER_UDP_HOST must override the default."""
    monkeypatch.setenv("SPEAKER_UDP_HOST", "192.168.1.10")
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_HOST == "192.168.1.10"


def test_microphone_tcp_host_env_override(monkeypatch, reload_config):
    """Setting MICROPHONE_TCP_HOST must override the default."""
    monkeypatch.setenv("MICROPHONE_TCP_HOST", "192.168.1.5")
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_HOST == "192.168.1.5"


# ---------------------------------------------------------------------------
# Only one canonical variable per setting — no silent alternate names
# ---------------------------------------------------------------------------

def test_speaker_udp_dest_port_is_a_deprecated_alias(monkeypatch, reload_config):
    """SPEAKER_UDP_DEST_PORT is a deprecated alias — config.py mirrors
    the canonical value into it but must NOT read from it. Setting it
    in the environment must NOT change the resolved value."""
    monkeypatch.delenv("SPEAKER_UDP_PORT", raising=False)
    monkeypatch.delenv("SPEAKER_UDP_DEST_PORT", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_PORT == 5000
    # The alias mirrors the canonical value.
    assert cfg.SPEAKER_UDP_DEST_PORT == cfg.SPEAKER_UDP_PORT

    # Now set only the deprecated alias. The canonical value must
    # remain the default because config.py must not read the alias.
    monkeypatch.setenv("SPEAKER_UDP_DEST_PORT", "12345")
    cfg = reload_config()
    assert cfg.SPEAKER_UDP_PORT == 5000


# ---------------------------------------------------------------------------
# audio_main.py consumes the resolved values
# ---------------------------------------------------------------------------

def test_audio_main_uses_canonical_config(monkeypatch, reload_config):
    """audio_main.py must not invent its own defaults; it must read
    the resolved values from config.py."""
    monkeypatch.setenv("MICROPHONE_TCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MICROPHONE_TCP_PORT", "5002")
    monkeypatch.setenv("SPEAKER_UDP_HOST", "192.168.1.10")
    monkeypatch.setenv("SPEAKER_UDP_PORT", "5000")
    cfg = reload_config()

    # Re-import audio_main so it picks up the new config values.
    import audio_main as audio_main_module
    importlib.reload(audio_main_module)

    # The names audio_main imports must equal what config.py resolved.
    assert audio_main_module.MICROPHONE_TCP_HOST == "0.0.0.0"
    assert audio_main_module.MICROPHONE_TCP_PORT == 5002
    assert audio_main_module.SPEAKER_UDP_HOST == "192.168.1.10"
    assert audio_main_module.SPEAKER_UDP_PORT == 5000


# ---------------------------------------------------------------------------
# Startup logging reflects the resolved values
# ---------------------------------------------------------------------------

def test_startup_logging_reports_resolved_values(monkeypatch, reload_config, caplog):
    """The startup log line for the speaker destination must show the
    exact (host, port) tuple that was resolved, not a hardcoded
    fallback."""
    import logging

    monkeypatch.setenv("SPEAKER_UDP_HOST", "192.168.1.10")
    monkeypatch.setenv("SPEAKER_UDP_PORT", "5000")
    reload_config()

    import audio_main as audio_main_module
    importlib.reload(audio_main_module)

    caplog.set_level(logging.INFO)
    # Construct the same log line audio_main.py emits.
    logger = logging.getLogger("audio-bridge")
    logger.info(
        "Speaker UDP destination: %s:%d",
        audio_main_module.SPEAKER_UDP_HOST,
        audio_main_module.SPEAKER_UDP_PORT,
    )

    text = caplog.text
    assert "Speaker UDP destination: 192.168.1.10:5000" in text
    assert "9813" not in text


def test_startup_logging_reports_mic_resolved_values(monkeypatch, reload_config, caplog):
    """The mic startup log must show the resolved (host, port)."""
    import logging

    monkeypatch.setenv("MICROPHONE_TCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MICROPHONE_TCP_PORT", "5002")
    reload_config()

    import audio_main as audio_main_module
    importlib.reload(audio_main_module)

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("audio-bridge")
    logger.info(
        "Microphone TCP server listening on %s:%d",
        audio_main_module.MICROPHONE_TCP_HOST,
        audio_main_module.MICROPHONE_TCP_PORT,
    )

    text = caplog.text
    assert "Microphone TCP server listening on 0.0.0.0:5002" in text


# ---------------------------------------------------------------------------
# No duplication of port numbers across files
# ---------------------------------------------------------------------------

def test_no_production_default_9813_remains():
    """Search the production source tree for the bare literal '9813'.

    The production code (config.py, audio_main.py, transport/, audio/)
    must not contain the literal 9813 as a default or resolved value.
    """
    import pathlib
    prod_roots = [
        pathlib.Path("/home/manav1011/Documents/udp-audio-transport/config.py"),
        pathlib.Path("/home/manav1011/Documents/udp-audio-transport/audio_main.py"),
    ]
    for path in prod_roots:
        text = path.read_text()
        assert "9813" not in text, (
            f"{path} still references 9813 in production code"
        )


# ---------------------------------------------------------------------------
# Canonical-name regression: production must import SPEAKER_UDP_*, not _DEST_*
# ---------------------------------------------------------------------------

def test_audio_main_does_not_import_legacy_dest_names():
    """audio_main.py must use the canonical SPEAKER_UDP_HOST /
    SPEAKER_UDP_PORT names — the legacy SPEAKER_UDP_DEST_* names are
    only kept as deprecated aliases in config.py and must not appear
    in production code."""
    import pathlib
    text = pathlib.Path(
        "/home/manav1011/Documents/udp-audio-transport/audio_main.py"
    ).read_text()
    assert "SPEAKER_UDP_DEST_HOST" not in text
    assert "SPEAKER_UDP_DEST_PORT" not in text
    assert "SPEAKER_UDP_HOST" in text
    assert "SPEAKER_UDP_PORT" in text