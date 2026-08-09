"""Tests for PhoneSpeakerManager — Phase 6 speaker sink lifecycle.

PhoneSpeakerManager is responsible for:
    * Creating the Phone_Speaker null sink tagged with
      `audio-bridge.owned=true`.
    * Exposing Phone_Speaker.monitor as the capture source.
    * Idempotent startup (no-op if the sink already exists).
    * Best-effort teardown via the module index returned at load time.

These tests use the real pactl interface but are skipped if pactl is
not available, so they don't break CI on systems without PipeWire.
"""
from __future__ import annotations

import shutil

import pytest

from audio.phone_speaker import (
    OWNERSHIP_PROPERTY,
    OWNERSHIP_VALUE,
    PhoneSpeakerManager,
)


def _pactl_available() -> bool:
    return shutil.which("pactl") is not None


pytestmark = pytest.mark.skipif(
    not _pactl_available(), reason="pactl not available in this environment"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_sink_name_constant():
    """The sink name must be exactly 'Phone_Speaker' — this is the
    public contract that the user selects in GNOME Sound."""
    mgr = PhoneSpeakerManager()
    assert mgr.SINK_NAME == "Phone_Speaker"


def test_sink_name_accessor():
    """sink_name() must return the same value as the constant."""
    mgr = PhoneSpeakerManager()
    assert mgr.sink_name() == "Phone_Speaker"


def test_monitor_source_name():
    """monitor_source_name() must be '<sink_name>.monitor' — the
    source the capture path subscribes to in PipeWire."""
    mgr = PhoneSpeakerManager()
    assert mgr.monitor_source_name() == "Phone_Speaker.monitor"


def test_ownership_property_marker():
    """The ownership marker constant must be the documented key/value.
    Tests that scan for stale devices rely on these strings."""
    assert OWNERSHIP_PROPERTY == "audio-bridge.owned"
    assert OWNERSHIP_VALUE == "true"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_double_start_is_idempotent():
    """Calling start() twice must not raise — the second call sees the
    sink already exists and returns."""
    mgr = PhoneSpeakerManager()
    mgr.start()
    try:
        # Second start() must not raise even if the sink is already
        # present.
        mgr.start()
    finally:
        mgr.stop()


def test_stop_without_start_is_safe():
    """Calling stop() without start() must not raise — there is no
    module index to unload, so it must be a no-op."""
    mgr = PhoneSpeakerManager()
    mgr.stop()


def test_double_stop_is_safe():
    """Calling stop() twice must not raise — the second call has no
    module index to unload."""
    mgr = PhoneSpeakerManager()
    mgr.start()
    mgr.stop()
    # Second stop is a no-op.
    mgr.stop()


# ---------------------------------------------------------------------------
# Captured at-load time: the manager remembers the module index it
# loaded so stop() can unload the exact module it created.
# ---------------------------------------------------------------------------

def test_module_index_set_after_start():
    """After start() creates the sink, the manager records a non-None
    module index. After stop() it's cleared."""
    mgr = PhoneSpeakerManager()
    assert mgr._module is None
    mgr.start()
    try:
        assert mgr._module is not None
        assert mgr._module.strip().isdigit()
    finally:
        mgr.stop()
    assert mgr._module is None


# ---------------------------------------------------------------------------
# Ownership marker is present on the module
# ---------------------------------------------------------------------------

def test_owned_module_arguments_carry_marker():
    """After start(), `pactl list modules` must show our newly-created
    module with `audio-bridge.owned=true` in its Argument: string —
    this is what makes stale-device cleanup safe."""
    import subprocess

    mgr = PhoneSpeakerManager()
    mgr.start()
    try:
        result = subprocess.run(
            ["pactl", "list", "modules"],
            capture_output=True, text=True, check=True,
        )
        # Find the module with our sink_name. The Argument: line must
        # also contain the ownership marker.
        text = result.stdout
        assert "Phone_Speaker" in text
        assert f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}" in text
    finally:
        mgr.stop()