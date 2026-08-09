"""Tests for PhoneSpeakerManager.monitor_source_index() resolution.

The pure-parser tests (no pactl needed) run in any environment.
The subprocess-mocked tests run in any environment by patching
``subprocess.run`` in ``audio.phone_speaker``. The end-to-end test
is gated on a real ``pactl`` binary being present.

These tests verify:
    * The parser correctly resolves Phone_Speaker.monitor to its
      column-0 index in tab-separated pactl output.
    * The parser does NOT hardcode any specific index value.
    * The resolver fails loudly with the full pactl output when the
      monitor is absent, when pactl errors, or when pactl is missing.
"""
from __future__ import annotations

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from audio.phone_speaker import (
    PhoneSpeakerError,
    PhoneSpeakerManager,
    _find_source_index_by_name,
    _list_sources_short_output,
)


# ---------------------------------------------------------------------------
# Pure parser — runs everywhere, no pactl needed
# ---------------------------------------------------------------------------


def test_parser_finds_index_in_tab_separated_output():
    """A typical pactl list sources short output: column 0 is the
    numeric Pulse source index, column 1 is the source name."""
    output = (
        "2561\tPhone_Microphone.monitor\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n"
        "2568\tPhone_Microphone_Input\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n"
        "2581\tPhone_Speaker.monitor\tPipeWire\tfloat32le 2ch 48000Hz\tSUSPENDED\n"
    )
    assert _find_source_index_by_name("Phone_Speaker.monitor", output) == "2581"


def test_parser_returns_none_when_name_absent():
    """If the requested source name is not in column 1 of any row,
    the parser returns None — it does NOT raise."""
    output = "2581\tPhone_Microphone.monitor\tPipeWire\t...\n"
    assert _find_source_index_by_name("Phone_Speaker.monitor", output) is None


def test_parser_ignores_short_lines():
    """Lines with fewer than 2 tab-separated columns are ignored."""
    output = (
        "garbage\n"
        "single_column\n"
        "2581\tPhone_Speaker.monitor\tPipeWire\t...\n"
    )
    assert _find_source_index_by_name("Phone_Speaker.monitor", output) == "2581"


def test_parser_does_not_match_substring():
    """A name match must be exact on column 1, not a substring match.

    `Phone_Speaker.monitor.extra` must not match `Phone_Speaker.monitor`.
    """
    output = (
        "1\tPhone_Speaker.monitor.extra\tPipeWire\t...\n"
        "2\tPhone_Speaker.monitor\tPipeWire\t...\n"
    )
    assert _find_source_index_by_name("Phone_Speaker.monitor", output) == "2"


def test_parser_does_not_match_other_monitor():
    """Two sink monitors in the output — the parser returns only the
    one whose column 1 is exactly the requested name."""
    output = (
        "100\tPhone_Microphone.monitor\tPipeWire\t...\n"
        "200\tPhone_Speaker.monitor\tPipeWire\t...\n"
    )
    assert _find_source_index_by_name("Phone_Speaker.monitor", output) == "200"
    assert _find_source_index_by_name("Phone_Microphone.monitor", output) == "100"


def test_parser_handles_empty_output():
    """Empty pactl output returns None, not an error."""
    assert _find_source_index_by_name("Phone_Speaker.monitor", "") is None


# ---------------------------------------------------------------------------
# Resolution + raising — uses subprocess.run monkeypatch (no pactl needed)
# ---------------------------------------------------------------------------


def _completed(stdout: str, returncode: int = 0):
    """Build a fake CompletedProcess-like object."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = ""
    cp.returncode = returncode
    return cp


def test_monitor_source_index_returns_resolved_index(monkeypatch):
    """With pactl output stubbed, monitor_source_index() returns the
    column-0 index for the Phone_Speaker.monitor row — and crucially
    does NOT hardcode any specific value (it returns whatever pactl said)."""
    mgr = PhoneSpeakerManager()
    fake_output = (
        "2561\tPhone_Microphone.monitor\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n"
        "2581\tPhone_Speaker.monitor\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING\n"
    )
    monkeypatch.setattr(
        "audio.phone_speaker.subprocess.run",
        lambda *a, **kw: _completed(fake_output),
    )
    assert mgr.monitor_source_index() == "2581"


def test_monitor_source_index_uses_dynamic_resolution(monkeypatch):
    """If pactl gives a different index on a different run, the resolver
    must return the new index — i.e. it does not hardcode 2581."""
    mgr = PhoneSpeakerManager()
    fake_output = "7777\tPhone_Speaker.monitor\tPipeWire\t...\n"
    monkeypatch.setattr(
        "audio.phone_speaker.subprocess.run",
        lambda *a, **kw: _completed(fake_output),
    )
    assert mgr.monitor_source_index() == "7777"


def test_monitor_source_index_fails_loudly_when_missing(monkeypatch):
    """If Phone_Speaker.monitor is not in pactl output,
    PhoneSpeakerError is raised and the exception message contains
    the pactl output so the operator can debug."""
    mgr = PhoneSpeakerManager()
    fake_output = "2568\tPhone_Microphone_Input\tPipeWire\t...\n"
    monkeypatch.setattr(
        "audio.phone_speaker.subprocess.run",
        lambda *a, **kw: _completed(fake_output),
    )
    with pytest.raises(PhoneSpeakerError) as excinfo:
        mgr.monitor_source_index()
    msg = str(excinfo.value)
    assert "Phone_Speaker.monitor" in msg
    assert "Phone_Microphone_Input" in msg  # full pactl output included


def test_monitor_source_index_fails_loudly_on_pactl_error(monkeypatch):
    """If pactl exits non-zero (CalledProcessError), PhoneSpeakerError
    is raised — never silent fallback."""
    mgr = PhoneSpeakerManager()

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    monkeypatch.setattr("audio.phone_speaker.subprocess.run", fake_run)
    with pytest.raises(PhoneSpeakerError):
        mgr.monitor_source_index()


def test_monitor_source_index_fails_loudly_on_missing_pactl(monkeypatch):
    """If pactl binary is missing (FileNotFoundError), PhoneSpeakerError."""
    mgr = PhoneSpeakerManager()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr("audio.phone_speaker.subprocess.run", fake_run)
    with pytest.raises(PhoneSpeakerError):
        mgr.monitor_source_index()


def test_list_sources_short_output_returns_empty_on_pactl_error(monkeypatch):
    """The _list_sources_short_output helper returns '' when pactl
    fails (the resolver then raises PhoneSpeakerError above this)."""
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    monkeypatch.setattr("audio.phone_speaker.subprocess.run", fake_run)
    assert _list_sources_short_output() == ""


# ---------------------------------------------------------------------------
# Belt-and-braces: pactl-locked end-to-end test
# ---------------------------------------------------------------------------


def _pactl_available() -> bool:
    return shutil.which("pactl") is not None


@pytest.mark.skipif(
    not _pactl_available(), reason="pactl not available in this environment"
)
def test_monitor_source_index_end_to_end_with_real_pactl():
    """When pactl is real and Phone_Speaker.monitor exists (via the
    rest of the test fixture), the index returned by
    monitor_source_index() matches the index in a fresh
    `pactl list sources short` invocation.
    """
    mgr = PhoneSpeakerManager()
    mgr.start()
    try:
        index = mgr.monitor_source_index()
        assert index is not None
        assert index.isdigit()
        # Cross-check with a fresh pactl invocation.
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, check=True,
        )
        assert index == _find_source_index_by_name(
            mgr.monitor_source_name(), result.stdout,
        )
    finally:
        mgr.stop()
