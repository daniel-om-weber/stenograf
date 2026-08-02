"""Windows capture provider — spawns the Rust helper and reads its frames.

The helper (``stenocap.exe``, `native/stenocap`) opens WASAPI shared-mode
streams for both channels — mic from the default capture endpoint, system audio
from *loopback* on the default render endpoint — and streams them as framed PCM
on its stdout. The transport is shared with macOS in
:mod:`stenograf.capture.helper`; what lives here is the two things Windows has
that no other platform does.

**Windows never prompts for the microphone.** There is no TCC equivalent: a
denied privacy toggle silently makes the stream deliver zeros, so the consent
store is read up front instead (:func:`mic_access_blocked`), before capture and
models start.

Both devices pin to the defaults at meeting start; a mid-meeting default switch
is not followed (unlike ``@DEFAULT_MONITOR@`` on Linux — WASAPI has no
equivalent alias). No code path writes audio to disk.
"""

from __future__ import annotations

import sys

from stenograf.capture.base import (
    CaptureUnavailableError,
    Channel,
)
from stenograf.capture.helper import HelperCaptureProvider, query_devices


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

    The shared helper preflight (:func:`stenograf.capture.helper.query_devices`)
    plus the two things only Windows has: the privacy consent store read before
    the helper runs, and a "(loopback)" suffix — the loopback-of-default-output
    choice is invisible in the device's own name.
    """
    for channel in sorted(channels):
        if channel is Channel.MIC and (blocked := mic_access_blocked()):
            raise CaptureUnavailableError(blocked)

    devices = query_devices(channels)
    return {
        ch: f"{name} (loopback)" if ch is Channel.SYSTEM else name
        for ch, name in devices.items()
    }


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
