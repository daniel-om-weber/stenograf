"""Windows capture provider — WASAPI shared-mode streams via ``soundcard``.

Windows exposes system audio as WASAPI *loopback* capture on a render device,
so no native helper is needed: the ``soundcard`` package opens shared-mode
capture streams for both channels —

- mic    → the default input device
- system → loopback capture on the default output device

**Decision D (PLAN.md §5 Phase 6) resolved to ``soundcard``** over
``pyaudiowpatch``, spiked on real hardware (Windows 11 notebook, 2026-07-11):
one API covers mic and loopback, and it initializes WASAPI with
``AUTOCONVERTPCM | SRC_DEFAULT_QUALITY``, so Windows resamples server-side to
our wire rate like parec does on Linux — no Python resampler dependency.
Measured: recorder open 15–80 ms, first frame < 300 ms, exact frame-sized
delivery cadence, a 440 Hz test tone recovered bit-clean through loopback
(the ~1 s startup gap SoundCard showed under PipeWire is a Pulse-backend
artifact; it does not occur on the native Windows backend).

Two WASAPI behaviours the design leans on:

- **Loopback delivers no packets while nothing renders.** ``soundcard``
  papers over this by synthesizing zeros from the measured idle time, so the
  stream stays continuous — but the fill is wall-clock *estimated*, so
  sample-count-derived timestamps can drift from session time across long
  silences. The session clock re-anchors whenever a channel's derived clock
  falls behind the arrival-derived one by more than
  ``_REANCHOR_TOLERANCE_S`` (forward only: per-channel timestamps must stay
  monotonic, and ``SessionStore`` pads the skipped span with silence).
- **COM apartments are per-thread**, so each channel's device is resolved and
  its recorder opened *inside* its own pump thread (spike-verified working).

Queue/pump/clock machinery is shared with the Linux provider
(:mod:`stenograf.capture.streaming`). Both devices pin to the defaults at
meeting start; a mid-meeting default switch is not followed (unlike
``@DEFAULT_MONITOR@`` on Linux — WASAPI has no equivalent alias). No code
path writes audio to disk.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
from collections.abc import Callable

import numpy as np

from stenograf.audio import to_int16
from stenograf.capture.base import (
    DEFAULT_FRAME_MS,
    SAMPLE_RATE,
    CaptureUnavailableError,
    Channel,
)
from stenograf.capture.streaming import QueueStreamingProvider

_REANCHOR_TOLERANCE_S = 0.5
"""How far a channel's sample-derived clock may fall behind its
arrival-derived clock before the session clock re-anchors. Generous enough
that delivery jitter (one frame + WASAPI buffering, ~0.25 s worst measured)
never trips it; tight enough that a mis-estimated silence gap cannot skew the
AEC's far-end alignment or the transcript for the rest of the meeting."""


_SILENT_MIC_WARN_S = 5.0
"""Seconds of *exact-zero* mic PCM before the pump warns once. Real
microphones have a noise floor well above one int16 step, so a run of digital
zeros this long means the stream is dead (hardware mute, a privacy toggle the
consent-store check missed, a broken device) — never a quiet room."""


def _import_soundcard():
    try:
        import soundcard  # pyright: ignore[reportMissingImports] — Windows-only dependency
    except Exception as exc:  # ImportError, or COM/cffi init failures
        raise CaptureUnavailableError(
            f"the soundcard package is unavailable ({exc}) — reinstall stenograf, "
            "or `pip install soundcard`, to capture on Windows"
        ) from exc
    # Loopback silence gaps set WASAPI's discontinuity flag when audio
    # resumes; that is expected and handled (zero-fill + re-anchor), so the
    # per-gap warning would only spam the terminal/TUI.
    if (category := getattr(soundcard, "SoundcardRuntimeWarning", None)) is not None:
        warnings.filterwarnings("ignore", category=category)
    return soundcard


def default_devices(channels: set[Channel]) -> dict[Channel, str]:
    """What each channel would record from right now.

    Resolves the default devices the same way the pumps will at start, so a
    missing package, an absent default device, or a denied microphone privacy
    toggle fails *before* capture (and models) start, and so the CLI can name
    what the meeting will record — the loopback-of-default-output choice is
    invisible otherwise.
    """
    soundcard = _import_soundcard()
    devices = {}
    for channel in sorted(channels):
        if channel is Channel.MIC and (blocked := mic_access_blocked()):
            raise CaptureUnavailableError(blocked)
        device = _default_device(soundcard, channel)
        suffix = " (loopback)" if channel is Channel.SYSTEM else ""
        devices[channel] = f"{device.name}{suffix}"
    return devices


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


def _default_device(soundcard, channel: Channel):
    """The soundcard device a channel records from (mic, or output loopback)."""
    if channel is Channel.MIC:
        try:
            return soundcard.default_microphone()
        except Exception as exc:
            raise CaptureUnavailableError(
                f"no default microphone ({exc}) — check Windows sound settings"
            ) from exc
    try:
        speaker = soundcard.default_speaker()
        return soundcard.get_microphone(speaker.id, include_loopback=True)
    except Exception as exc:
        raise CaptureUnavailableError(
            f"no default output device to loopback-capture ({exc}) — check Windows sound settings"
        ) from exc


class WindowsCaptureProvider(QueueStreamingProvider[None]):
    """Streams frames from one WASAPI capture stream per captured channel.

    ``backend`` overrides the soundcard module (a fake in tests); production
    imports the real one and fails at construction when it is missing,
    mirroring ``find_helper`` on macOS and the parec check on Linux. ``clock``
    overrides the session clock (tests drive the re-anchor logic with it).

    Each channel gets one pump thread that owns its device end to end (COM
    objects are apartment-bound): it opens the recorder, downmixes the
    device's float32 channels to mono int16, and stamps ~200 ms frames onto
    the shared session clock; ``frames()`` drains their queue.
    """

    _thread_prefix = "wasapi"

    def __init__(
        self,
        *,
        backend=None,
        frame_ms: int = DEFAULT_FRAME_MS,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(
            frame_ms=frame_ms, clock=clock, reanchor_tolerance_s=_REANCHOR_TOLERANCE_S
        )
        self._soundcard = backend if backend is not None else _import_soundcard()

    def _open_channel(self, channel: Channel) -> None:
        return None  # COM: the device must be resolved inside the pump thread

    def _pump(self, channel: Channel, transport: None) -> None:
        """Own one channel end to end: device, recorder, framing, timestamps.

        Runs until the stop event is set or the stream dies; the session
        clock stamps each frame (with the forward re-anchor after an
        under-filled loopback silence — module docstring).
        """
        zero_run = 0
        warned_silent = False
        try:
            device = _default_device(self._soundcard, channel)
            with device.recorder(samplerate=SAMPLE_RATE) as recorder:
                while not self._stop_event.is_set():
                    block = recorder.record(self._frame_samples)
                    samples = _to_mono_int16(block)
                    if not len(samples):
                        continue
                    # Silent-mic watchdog (mic only: a quiet *system* channel is
                    # normal). Exact zeros this long are a dead stream, not a
                    # quiet room — see _SILENT_MIC_WARN_S.
                    if channel is Channel.MIC:
                        zero_run = zero_run + len(samples) if not samples.any() else 0
                        if not warned_silent and zero_run >= _SILENT_MIC_WARN_S * SAMPLE_RATE:
                            warned_silent = True
                            print(
                                f"stenograf: the microphone has delivered only silence for "
                                f"{_SILENT_MIC_WARN_S:.0f}s — check the input volume and "
                                "Windows privacy settings "
                                "(Settings > Privacy & security > Microphone)",
                                file=sys.stderr,
                            )
                    self._emit(channel, samples)
        except Exception as exc:
            # The other providers inherit their subprocess's stderr; this is
            # the in-process equivalent so the user sees why a stream died.
            print(f"stenograf: {channel.value} capture stream died: {exc}", file=sys.stderr)

    def _stop_transport(self) -> None:
        # Pumps notice the stop event within one frame read (~frame_ms +
        # WASAPI's silence threshold) and release their devices on the way
        # out. Skip the current thread: stop() also runs *from* a pump on an
        # unexpected stream death.
        current = threading.current_thread()
        for thread in self._threads.values():
            if thread is not current:
                thread.join(timeout=5)


def _to_mono_int16(block: np.ndarray) -> np.ndarray:
    """Downmix a float32 frames×channels block to the wire format."""
    mono = block.mean(axis=1) if block.ndim == 2 else block
    return to_int16(mono)
