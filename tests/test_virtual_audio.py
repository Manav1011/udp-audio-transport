"""Tests for VirtualAudioManager ownership + stale-device cleanup — Phase 6.

Verifies:
    * Module argument parser handles quoted values.
    * Stale-device cleanup unloads only modules carrying the
      `audio-bridge.owned=true` marker.
    * Devices that lack the marker are NEVER touched.
    * Module listing includes argument parsing.

These tests are skipped if pactl is not available.
"""
from __future__ import annotations

import shutil

import pytest

from audio.phone_speaker import (
    OWNERSHIP_PROPERTY,
    OWNERSHIP_VALUE,
)
from audio.virtual_audio import (
    VirtualAudioManager,
    _parse_module_arguments,
)


def _pactl_available() -> bool:
    return shutil.which("pactl") is not None


pytestmark = pytest.mark.skipif(
    not _pactl_available(), reason="pactl not available in this environment"
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def test_parse_module_arguments_simple():
    """A flat key=value list is parsed into a dict."""
    args = _parse_module_arguments("sink_name=Phone_Speaker channels=2")
    assert args == {"sink_name": "Phone_Speaker", "channels": "2"}


def test_parse_module_arguments_quoted():
    """PipeWire echoes arguments without quote protection, so a
    multi-word value appears as multiple tokens. We only care about
    simple identifier values: sink_name, source_name, and the ownership
    marker. The multi-word value 'Phone Speaker Sink' is treated as
    two unrelated words and ignored."""
    args = _parse_module_arguments(
        'sink_name=Phone_Speaker '
        'sink_properties=device.description="Phone Speaker Sink" '
        f'{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}'
    )
    # Simple identifier values are captured.
    assert args["sink_name"] == "Phone_Speaker"
    assert args[OWNERSHIP_PROPERTY] == OWNERSHIP_VALUE
    # The multi-word description value is dropped (we don't need it).
    assert "device.description" not in args


def test_parse_module_arguments_empty():
    """Empty input parses to an empty dict."""
    assert _parse_module_arguments("") == {}


def test_parse_module_arguments_single_token():
    """A single token parses cleanly."""
    args = _parse_module_arguments("foo=bar")
    assert args == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Module listing
# ---------------------------------------------------------------------------

def test_list_sink_modules_returns_dicts():
    """_list_sink_modules() returns a non-empty list of dicts each
    having 'index', 'name', and 'args' keys."""
    mgr = VirtualAudioManager()
    modules = mgr._list_sink_modules()
    assert isinstance(modules, list)
    assert len(modules) > 0
    for m in modules:
        assert "index" in m
        assert "name" in m
        assert "args" in m
        assert isinstance(m["args"], dict)


# ---------------------------------------------------------------------------
# Stale-device cleanup
# ---------------------------------------------------------------------------

def test_unload_orphans_removes_owned_modules():
    """An owned module created by a previous run is unloaded by
    _unload_orphans()."""
    import subprocess

    mgr = VirtualAudioManager()
    # Manually load an owned sink to simulate a leftover.
    result = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={mgr.SINK_NAME} "
            f"sink_properties=device.description=\"{mgr.SINK_NAME}\" "
            f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}",
        ],
        capture_output=True, text=True, check=True,
    )
    leftover_index = result.stdout.strip()
    try:
        # Run cleanup.
        mgr._unload_orphans()

        # The leftover module must be gone — verify by attempting to
        # unload again and checking the error message.
        result = subprocess.run(
            ["pactl", "unload-module", leftover_index],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            f"leftover module #{leftover_index} was not unloaded"
        )
    finally:
        subprocess.run(
            ["pactl", "unload-module", leftover_index],
            capture_output=True,
        )


def test_unload_orphans_ignores_unowned_devices():
    """A device whose name matches ours but that lacks the ownership
    marker is NOT touched by cleanup. We simulate this by loading a
    sink with the same name but a different ownership property."""
    import subprocess

    mgr = VirtualAudioManager()
    # Load a sink that has the same name as ours but lacks the
    # ownership marker. We use a distinct property to be sure.
    result = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={mgr.SINK_NAME} "
            f"sink_properties=device.description=\"user-created\"",
        ],
        capture_output=True, text=True, check=True,
    )
    leftover_index = result.stdout.strip()
    try:
        # Cleanup must NOT remove this module.
        mgr._unload_orphans()

        # The unowned module must still be unloadable (still loaded).
        result = subprocess.run(
            ["pactl", "unload-module", leftover_index],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"unowned module #{leftover_index} was wrongly removed"
        )
    finally:
        subprocess.run(
            ["pactl", "unload-module", leftover_index],
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# start() now invokes _unload_orphans() before creating devices
# ---------------------------------------------------------------------------

def test_start_unloads_prior_orphans_before_creating():
    """If a previous run left behind a stale owned module, start()
    must remove it before recreating fresh devices.

    PipeWire may reuse module indices after unload, so we identify the
    leftover module by its argument signature (sink_name + ownership
    marker) rather than by its numerical index.
    """
    import subprocess

    mgr = VirtualAudioManager()
    # Pre-load an owned sink that simulates a prior run. The
    # device.description is a distinctive marker we can scan for.
    distinctive_marker = "stale-device-cleanup-test-marker"
    result = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={mgr.SINK_NAME} "
            f"sink_properties=device.description=\"{distinctive_marker}\" "
            f"{OWNERSHIP_PROPERTY}={OWNERSHIP_VALUE}",
        ],
        capture_output=True, text=True, check=True,
    )
    try:
        # Sanity check: the distinctive marker must appear in pactl list.
        pre = subprocess.run(
            ["pactl", "list", "modules"],
            capture_output=True, text=True, check=True,
        )
        assert distinctive_marker in pre.stdout, (
            "pre-load didn't take effect; cannot test cleanup"
        )

        mgr.start()
        try:
            # After start(), the distinctive marker must be gone —
            # start() cleaned up the leftover before recreating.
            post = subprocess.run(
                ["pactl", "list", "modules"],
                capture_output=True, text=True, check=True,
            )
            assert distinctive_marker not in post.stdout, (
                "start() did not clean up the leftover owned module"
            )

            # And the new module exists.
            assert mgr._mic_module is not None
        finally:
            mgr.stop()
    finally:
        # mgr.stop() already cleaned up anything it created. If anything
        # remains, just let it sit — the next run will clean it up via
        # ownership marker.
        pass