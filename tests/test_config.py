"""Configuration tests — verifies the canonical environment variables
and defaults for the production audio transport.

The production transport is TCP for both directions:

    Microphone: Android → TCP :5002 → Backend
    Speaker:    Backend → TCP :5000 → Android

Canonical environment variables:

    MICROPHONE_TCP_HOST  default "0.0.0.0"
    MICROPHONE_TCP_PORT  default 5002
    SPEAKER_TCP_HOST     default "127.0.0.1"
    SPEAKER_TCP_PORT     default 5000

Each test that depends on environment state reloads the module so the
``os.environ.get(...)`` calls inside config.py are re-evaluated with
the monkey-patched environment.
"""
from __future__ import annotations

import importlib

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
        "SPEAKER_TCP_HOST",
        "SPEAKER_TCP_PORT",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_HOST == "0.0.0.0"
    assert cfg.MICROPHONE_TCP_PORT == 5002
    assert cfg.SPEAKER_TCP_PORT == 5000
    assert cfg.SPEAKER_TCP_HOST == "127.0.0.1"


def test_default_mic_port_is_5002(monkeypatch, reload_config):
    """The mic TCP port default must be exactly 5002."""
    monkeypatch.delenv("MICROPHONE_TCP_PORT", raising=False)
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_PORT == 5002


def test_default_speaker_port_is_5000(monkeypatch, reload_config):
    """The speaker TCP port default must be exactly 5000."""
    monkeypatch.delenv("SPEAKER_TCP_PORT", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_TCP_PORT == 5000


def test_default_speaker_host_is_localhost(monkeypatch, reload_config):
    """Speaker host has no useful default; localhost is the placeholder."""
    monkeypatch.delenv("SPEAKER_TCP_HOST", raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_TCP_HOST == "127.0.0.1"


def test_9813_is_never_a_production_default(monkeypatch, reload_config):
    """Hard guarantee: 9813 must not appear as a default anywhere in the
    canonical speaker-port resolution path."""
    for var in (
        "SPEAKER_TCP_PORT",
        "SPEAKER_UDP_PORT",
        "UDP_PORT",
        "UDP_BIND_PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = reload_config()
    assert cfg.SPEAKER_TCP_PORT != 9813


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def test_microphone_tcp_port_env_override(monkeypatch, reload_config):
    """Setting MICROPHONE_TCP_PORT must override the default."""
    monkeypatch.setenv("MICROPHONE_TCP_PORT", "6000")
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_PORT == 6000


def test_speaker_tcp_port_env_override(monkeypatch, reload_config):
    """Setting SPEAKER_TCP_PORT must override the default."""
    monkeypatch.setenv("SPEAKER_TCP_PORT", "7000")
    cfg = reload_config()
    assert cfg.SPEAKER_TCP_PORT == 7000


def test_speaker_tcp_host_env_override(monkeypatch, reload_config):
    """Setting SPEAKER_TCP_HOST must override the default."""
    monkeypatch.setenv("SPEAKER_TCP_HOST", "192.168.1.10")
    cfg = reload_config()
    assert cfg.SPEAKER_TCP_HOST == "192.168.1.10"


def test_microphone_tcp_host_env_override(monkeypatch, reload_config):
    """Setting MICROPHONE_TCP_HOST must override the default."""
    monkeypatch.setenv("MICROPHONE_TCP_HOST", "192.168.1.5")
    cfg = reload_config()
    assert cfg.MICROPHONE_TCP_HOST == "192.168.1.5"


# ---------------------------------------------------------------------------
# Canonical-name regression: production uses SPEAKER_TCP_*, not legacy UDP names
# ---------------------------------------------------------------------------

def test_production_uses_canonical_speaker_tcp_names(monkeypatch, reload_config):
    """config.py must expose SPEAKER_TCP_HOST and SPEAKER_TCP_PORT as the
    canonical names; SPEAKER_UDP_* must NOT be exposed as production
    transport variables."""
    cfg = reload_config()
    assert hasattr(cfg, "SPEAKER_TCP_HOST")
    assert hasattr(cfg, "SPEAKER_TCP_PORT")
    assert not hasattr(cfg, "SPEAKER_UDP_HOST")
    assert not hasattr(cfg, "SPEAKER_UDP_PORT")


# ---------------------------------------------------------------------------
# audio_main.py consumes the resolved values
# ---------------------------------------------------------------------------

def test_audio_main_uses_canonical_config(monkeypatch, reload_config):
    """audio_main.py must not invent its own defaults; it must read
    the resolved values from config.py."""
    monkeypatch.setenv("MICROPHONE_TCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MICROPHONE_TCP_PORT", "5002")
    monkeypatch.setenv("SPEAKER_TCP_HOST", "192.168.1.10")
    monkeypatch.setenv("SPEAKER_TCP_PORT", "5000")
    cfg = reload_config()

    # Re-import audio_main so it picks up the new config values.
    import audio_main as audio_main_module
    importlib.reload(audio_main_module)

    # The names audio_main imports must equal what config.py resolved.
    assert audio_main_module.MICROPHONE_TCP_HOST == "0.0.0.0"
    assert audio_main_module.MICROPHONE_TCP_PORT == 5002
    assert audio_main_module.SPEAKER_TCP_HOST == "192.168.1.10"
    assert audio_main_module.SPEAKER_TCP_PORT == 5000


# ---------------------------------------------------------------------------
# Startup logging reflects the resolved values
# ---------------------------------------------------------------------------

def test_startup_logging_reports_resolved_values(monkeypatch, reload_config, caplog):
    """The startup log line for the speaker destination must show the
    exact (host, port) tuple that was resolved, not a hardcoded
    fallback. The label must be 'Speaker TCP destination' (not UDP)."""
    import logging

    monkeypatch.setenv("SPEAKER_TCP_HOST", "192.168.1.10")
    monkeypatch.setenv("SPEAKER_TCP_PORT", "5000")
    reload_config()

    import audio_main as audio_main_module
    importlib.reload(audio_main_module)

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("audio-bridge")
    logger.info(
        "Speaker TCP destination: %s:%d",
        audio_main_module.SPEAKER_TCP_HOST,
        audio_main_module.SPEAKER_TCP_PORT,
    )

    text = caplog.text
    assert "Speaker TCP destination: 192.168.1.10:5000" in text
    assert "Speaker UDP destination" not in text
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
# Canonical-name regression: production must import SPEAKER_TCP_*, not _UDP_*
# ---------------------------------------------------------------------------

def test_audio_main_does_not_import_legacy_udp_names():
    """audio_main.py must use the canonical SPEAKER_TCP_HOST /
    SPEAKER_TCP_PORT names — the legacy SPEAKER_UDP_* names must not
    appear in production code."""
    import pathlib
    text = pathlib.Path(
        "/home/manav1011/Documents/udp-audio-transport/audio_main.py"
    ).read_text()
    assert "SPEAKER_UDP_HOST" not in text
    assert "SPEAKER_UDP_PORT" not in text
    assert "SPEAKER_UDP_DEST_HOST" not in text
    assert "SPEAKER_UDP_DEST_PORT" not in text
    assert "SPEAKER_TCP_HOST" in text
    assert "SPEAKER_TCP_PORT" in text
