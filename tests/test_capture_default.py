"""Smoke tests for the speaker capture module's defaults.

Verifies that:
    * DEFAULT_CAPTURE_SOURCE is the new Phone_Speaker.monitor path
      (NOT the old PC_Audio_Capture_Input).
    * The Capture class accepts a constructor override.
    * set_capture_source() updates the source.

These tests import the module only — they do not start pw-cat, so they
work in any environment.
"""
from __future__ import annotations

import pytest

from audio.capture import (
    DEFAULT_CAPTURE_SOURCE,
    Capture,
)


def test_default_capture_source_is_phone_speaker_monitor():
    """The default capture source must be 'Phone_Speaker.monitor' —
    the user-selected sink's monitor — not the legacy
    'PC_Audio_Capture_Input'."""
    assert DEFAULT_CAPTURE_SOURCE == "Phone_Speaker.monitor"


def test_capture_default_uses_module_default():
    """A Capture instance with no constructor argument uses the
    module default."""
    cap = Capture()
    assert cap._capture_source == DEFAULT_CAPTURE_SOURCE


def test_capture_constructor_override():
    """A Capture instance with an explicit capture_source uses it."""
    cap = Capture(capture_source="My_Other_Sink.monitor")
    assert cap._capture_source == "My_Other_Sink.monitor"


def test_set_capture_source_updates_field():
    """set_capture_source() must change the captured source."""
    cap = Capture()
    cap.set_capture_source("Phone_Speaker.monitor")
    assert cap._capture_source == "Phone_Speaker.monitor"