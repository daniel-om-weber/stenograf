"""Windows capture provider — spawns the Rust helper and reads its frames.

The helper (``stenocap.exe``, `native/stenocap`) opens WASAPI shared-mode
streams for both channels — mic from the default capture endpoint, system audio
from *loopback* on the default render endpoint — and streams them as framed PCM
on its stdout. The transport is shared with macOS in
:mod:`stenograf.capture.helper`; what lives here is the two things Windows has
that no other platform does.

**Why a helper at all, when WASAPI is reachable from Python.** It was in-process
through ``soundcard`` until 2026-08, and the reason it moved is timestamps.
Frames stamped when they *arrive* carry their channel's transport latency as a
hidden offset, and the loopback tap is the longer path — so the echo canceller,
which pairs the two channels by timestamp, was handed a reference labelled
~60 ms *after* its own echo, which AEC3 cannot use at any delay hint. That
measured 2.6 dB ERLE with far-end speech attributed to the local speaker. A
declared constant (``FAR_END_LAG_S``) held the line at 13.7 dB while it lasted,
but the offset is re-rolled at every meeting start, so no declared value could
ever be right. WASAPI reports a device-side timestamp on every packet
(``pu64QPCPosition``) and the helper puts both taps on it — see
`native/README.md` and `PLAN-CAPTURE-HELPER.md`.

**Windows never prompts for the microphone.** There is no TCC equivalent: a
denied privacy toggle silently makes the stream deliver zeros, so the consent
store is read up front instead (:func:`mic_access_blocked`), before capture and
models start.

Both devices pin to the defaults at meeting start; a mid-meeting default switch
is not followed (unlike ``@DEFAULT_MONITOR@`` on Linux — WASAPI has no
equivalent alias). No code path writes audio to disk.
"""

from __future__ import annotations

import json
import subprocess
import sys

from stenograf.capture.base import (
    CaptureUnavailableError,
    Channel,
)
from stenograf.capture.helper import HelperCaptureProvider, find_helper

_CHANNEL_FLAG = {Channel.MIC: "--mic", Channel.SYSTEM: "--system"}
_CHANNEL_KEY = {"mic": Channel.MIC, "system": Channel.SYSTEM}


class WindowsCaptureProvider(HelperCaptureProvider):
    """Streams frames from the ``stenocap.exe`` subprocess.

    Stopping closes the helper's stdin rather than signalling it: Windows has no
    signal a parent can aim at one child (``CTRL_C_EVENT`` reaches a whole
    process group), while an EOF on the pipe needs no console, no process group
    and no handler — see :attr:`HelperCaptureProvider._stop_signal`.
    """

    _stop_signal = None


def default_devices(channels: set[Channel]) -> dict[Channel, str]:
    """What each channel would record from right now.

    Asks the helper, which resolves the defaults exactly as its pumps will at
    start — so a missing binary, an absent default device, or a denied
    microphone privacy toggle fails *before* capture (and models) start, and so
    the CLI can name what the meeting will record. The
    loopback-of-default-output choice is invisible otherwise.
    """
    for channel in sorted(channels):
        if channel is Channel.MIC and (blocked := mic_access_blocked()):
            raise CaptureUnavailableError(blocked)

    argv = [str(find_helper()), "--devices"]
    argv += [_CHANNEL_FLAG[ch] for ch in sorted(channels)]
    try:
        result = subprocess.run(argv, capture_output=True, timeout=15, check=False)
    except OSError as exc:
        raise CaptureUnavailableError(f"the capture helper could not be run ({exc})") from exc
    if result.returncode != 0:
        raise CaptureUnavailableError(_helper_complaint(result.stderr))
    try:
        named = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise CaptureUnavailableError(
            f"the capture helper did not report its devices ({exc})"
        ) from exc

    devices = {}
    for key, name in named.items():
        channel = _CHANNEL_KEY.get(key)
        if channel is None or channel not in channels:
            continue
        suffix = " (loopback)" if channel is Channel.SYSTEM else ""
        devices[channel] = f"{name}{suffix}"
    return devices


def _helper_complaint(stderr: bytes) -> str:
    """The helper's own reason, or a fallback when it died without giving one.

    Its diagnostics are ``stenocap: FATAL: <reason>`` lines; the prefixes are
    noise in a message the user reads, so everything up to and including the
    marker comes off.
    """
    lines = [ln.strip() for ln in stderr.decode("utf-8", errors="replace").splitlines()]
    fatal = next((ln for ln in reversed(lines) if "FATAL" in ln), None)
    if fatal is None:
        return (
            "the capture helper could not resolve the default devices "
            "— check Windows sound settings"
        )
    return fatal.split("FATAL", 1)[1].lstrip(": ").strip()


_CONSENT_STORE = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
)


def mic_access_blocked() -> str | None:
    """A user-facing reason when Windows privacy settings deny mic capture.

    Windows never *prompts* desktop apps for the microphone (unlike macOS TCC)
    — a denied toggle silently makes the stream deliver zeros — so the consent
    store behind Settings > Privacy & security > Microphone is read up front.
    Three switches can deny: the user's master toggle, the "let desktop apps
    access" toggle (``NonPackaged``), and the machine-wide/policy toggle.
    Returns ``None`` when allowed or undeterminable (missing keys mean the
    default, which is allowed); loopback capture is not privacy-gated.
    """
    denied = None
    if _consent_value("", machine=False) == "deny":
        denied = "microphone access is turned off"
    elif _consent_value("NonPackaged", machine=False) == "deny":
        denied = "microphone access for desktop apps is turned off"
    elif _consent_value("", machine=True) == "deny":
        denied = "microphone access is turned off machine-wide"
    if denied is None:
        return None
    return (
        f"{denied} in Windows privacy settings — enable it under "
        "Settings > Privacy & security > Microphone "
        "(including 'Let desktop apps access your microphone')"
    )


def _consent_value(subkey: str, *, machine: bool) -> str | None:
    """A consent-store key's ``Value`` ("allow"/"deny", lowered), or ``None``."""
    if sys.platform != "win32":  # also lets the type checker use win32 stubs
        return None
    import winreg

    hive = winreg.HKEY_LOCAL_MACHINE if machine else winreg.HKEY_CURRENT_USER
    path = f"{_CONSENT_STORE}\\{subkey}" if subkey else _CONSENT_STORE
    try:
        with winreg.OpenKey(hive, path) as key:
            value, _ = winreg.QueryValueEx(key, "Value")
    except OSError:
        return None
    return value.lower() if isinstance(value, str) else None
