"""Environment checks behind ``steno doctor`` — the first-run experience."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from stenograf.asr.registry import BackendSpec

MACOS_MIN_VERSION = (14, 4)  # Core Audio process taps


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    optional: bool = False
    """A not-ok optional check is reported but doesn't fail the doctor exit
    gate — for opt-in features (LLM notes) a machine can healthily lack."""


def run_checks() -> list[Check]:
    checks = [
        Check(
            name="Python",
            ok=sys.version_info >= (3, 12),
            detail=platform.python_version(),
        )
    ]

    if sys.platform == "darwin":
        checks.append(_macos_version_check())
        checks.append(_capture_helper_check())
        checks.append(_diarizer_helper_check())
    elif sys.platform.startswith("linux"):
        checks.append(_linux_capture_check())
        checks.append(_diarizer_helper_check())
    elif sys.platform == "win32":
        checks.append(_windows_capture_check())
        checks.append(_diarizer_helper_check())
    else:
        # Optional: `steno transcribe` is fully supported everywhere (the
        # ONNX backend), so a box missing only live capture is healthy.
        checks.append(
            Check(
                name="Platform",
                ok=False,
                detail=f"{sys.platform}: `steno transcribe` works here; live capture "
                "(`steno start`) is macOS/Linux/Windows-only so far",
                optional=True,
            )
        )

    checks.append(_desktop_app_check())
    checks.append(_asr_check())
    checks.append(_ffmpeg_check())
    checks.append(_models_check())
    checks.append(_settings_check())
    checks.append(_notes_check())
    checks.extend(_preset_checks())
    return checks


def _settings_check() -> Check:
    """Whether settings.toml (if present) parses and validates whole.

    Not optional: every command now resolves its defaults from the file at
    startup, so a broken file blocks ``steno start`` itself.

    It also names the output home, because that one is *resolved* rather than
    fixed — the documents folder is localised on Linux, so a user whose meetings
    are in ``~/Dokumente`` needs somewhere to read that off instead of guessing.
    """
    from stenograf.output import output_home
    from stenograf.settings import SettingsError, load_settings, settings_path

    path = settings_path()
    try:
        settings = load_settings()
    except SettingsError as exc:
        return Check(name="Settings", ok=False, detail=str(exc))
    home = output_home(settings)
    state = "OK" if path.exists() else "not present — all defaults"
    return Check(name="Settings", ok=True, detail=f"{path} {state}; meetings → {home}")


def installed(module: str) -> bool:
    """Whether ``module`` is importable, without importing it.

    Public: the backend loaders gate on a spec's ``requires`` with the same
    check doctor uses, so "is it installed" can never disagree between them."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _configured_mic_device() -> str | None:
    """``[capture] mic_device``, or ``None`` when the file is unusable.

    A broken settings file is the settings check's finding, not the capture
    check's — it reports the default behaviour rather than a second copy of the
    same error."""
    from stenograf.flow import resolve_mic_device
    from stenograf.settings import SettingsError, load_settings

    try:
        return resolve_mic_device(None, load_settings())
    except SettingsError:
        return None


def _pinned_mic_detail(pin: str) -> tuple[bool, str]:
    """Whether the pinned microphone is connected, and how to say so."""
    from stenograf.capture.base import CaptureUnavailableError
    from stenograf.capture.helper import list_input_devices, match_input_device

    try:
        listed = list_input_devices()
    except CaptureUnavailableError as exc:
        return False, f"microphone {pin!r} could not be checked ({exc})"
    found, problem = match_input_device(listed, pin)
    if found is None:
        return False, (
            f"the configured microphone {pin!r} {problem} — meetings will stop instead of "
            "recording another one; run `steno devices`, or remove [capture] mic_device"
        )
    return True, f"microphone: {found.name}"


def _capture_helper_check() -> Check:
    from stenograf.capture.helper import HelperNotFoundError, find_helper

    try:
        path = find_helper()
    except HelperNotFoundError as exc:
        return Check(name="Capture helper", ok=False, detail=str(exc))
    if not path.is_file():
        return Check(name="Capture helper", ok=False, detail=f"{path} is set but missing")
    if not os.access(path, os.X_OK):
        return Check(
            name="Capture helper", ok=False, detail=f"{path} is not executable — chmod +x it"
        )
    signed, why = _codesign_valid(path)
    if not signed:
        return Check(
            name="Capture helper",
            ok=False,
            detail=f"{path} has no valid code signature ({why}) — macOS refuses audio "
            "permissions to unsigned binaries; rebuild with native/stenocap-macos/build.sh",
        )
    detail = f"{path} — signed; grant the mic + system-audio permission once with `steno setup`"
    # Only a configured pin costs a subprocess here: listing devices is not
    # permission-gated (measured 2026-08-12), but a query nobody asked for on
    # every doctor run is still a query nobody asked for.
    pin = _configured_mic_device()
    if pin:
        ok, pinned = _pinned_mic_detail(pin)
        return Check(name="Capture helper", ok=ok, detail=f"{detail}; {pinned}")
    return Check(name="Capture helper", ok=True, detail=detail)


def _linux_capture_check() -> Check:
    """Whether `steno start` can capture here: helper present, server up, defaults set.

    Uses the same resolution the provider uses at meeting start — the helper's
    own ``--devices``, with ``[capture] mic_device`` applied — so an OK here
    means a meeting would actually record, and names the devices it would
    record from (the monitor-of-default-sink choice is invisible otherwise, and
    a configured microphone that is not connected fails here rather than at the
    start of a meeting).
    """
    from stenograf.capture.base import CaptureUnavailableError, Channel
    from stenograf.capture.helper import HelperCaptureProvider, query_devices

    try:
        HelperCaptureProvider()  # fails fast when the helper binary is missing
        devices = query_devices({Channel.MIC, Channel.SYSTEM}, mic_device=_configured_mic_device())
    except CaptureUnavailableError as exc:
        return Check(name="Capture", ok=False, detail=str(exc))
    listing = ", ".join(f"{ch.value} ← {device}" for ch, device in sorted(devices.items()))
    return Check(
        name="Capture", ok=True, detail=f"PipeWire/PulseAudio via stenocap ({listing})"
    )


def _windows_capture_check() -> Check:
    """Whether `steno start` can capture here: helper present, defaults set.

    Uses the same resolution the provider uses at meeting start — the helper's
    own ``--devices``, with ``[capture] mic_device`` applied — so an OK here
    means a meeting would actually record, and names the devices it would
    record from (the loopback-of-default-output choice is invisible otherwise,
    and a configured microphone that is not connected fails here rather than at
    the start of a meeting).
    """
    from stenograf.capture.base import Channel
    from stenograf.capture.windows import (
        CaptureUnavailableError,
        WindowsCaptureProvider,
        default_devices,
    )

    try:
        WindowsCaptureProvider()  # fails fast when the helper binary is missing
        devices = default_devices(
            {Channel.MIC, Channel.SYSTEM}, mic_device=_configured_mic_device()
        )
    except CaptureUnavailableError as exc:
        return Check(name="Capture", ok=False, detail=str(exc))
    listing = ", ".join(f"{ch.value} ← {device}" for ch, device in sorted(devices.items()))
    return Check(name="Capture", ok=True, detail=f"WASAPI via stenocap ({listing})")


def _diarizer_helper_check() -> Check:
    """stenodiar is optional: without it, an *estimated* speaker count falls back
    to sherpa's threshold clustering, which badly over-splits — explicit counts
    are unaffected. Report it missing without failing the doctor run."""
    from stenograf.diarization.speakrs import DiarizerHelperNotFoundError, find_stenodiar

    try:
        path = find_stenodiar()
    except DiarizerHelperNotFoundError as exc:
        return Check(name="Diarization helper (optional)", ok=True, detail=str(exc))
    if not os.access(path, os.X_OK):
        return Check(
            name="Diarization helper (optional)",
            ok=False,
            detail=f"{path} is not executable — chmod +x it",
        )
    return Check(
        name="Diarization helper (optional)",
        ok=True,
        detail=f"{path} — speaker counts are estimated with speakrs (VBx)",
    )


_QML_PROBE = """
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QPluginLoader
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlComponent, QQmlEngine

from stenograf.gui.app import QML_DIR

# Both get a name on purpose: as temporaries Python collects them mid-probe,
# and the component then reports Null with nothing to say about why
# (measured 2026-08-11).
application = QGuiApplication([])
engine = QQmlEngine()
for source in sorted(QML_DIR.glob("*.qml")):
    component = QQmlComponent(engine, str(source))
    if component.status() is not QQmlComponent.Status.Ready:
        sys.exit(component.errorString().strip() or f"{source.name} did not compile")

# Running offscreen means the platform plugin the app will really use never
# loads, and failing to load it is the most common way a Qt app cannot start.
# Loaded directly instead of by starting a second application: it resolves the
# same dependencies without needing a window server, and without the Dock tile
# a real platform would give a diagnostic that is supposed to be invisible.
# macOS and Windows only — which plugin a Linux session picks depends on the
# session, and a headless box legitimately has none.
platform_plugin = {"darwin": "libqcocoa.dylib", "win32": "qwindows.dll"}.get(sys.platform)
if platform_plugin:
    path = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    loader = QPluginLoader(str(path / "platforms" / platform_plugin))
    if not loader.load():
        sys.exit(f"{platform_plugin}: {loader.errorString()} ({path / 'platforms'})")
"""
"""What the app does when it opens, minus the window: every QML file compiled
(which is what resolves Qt's imports and loads the style plugin) and the
platform plugin loaded. Those are the steps that fail when a Qt library is
present but not loadable.

Compiled, not instantiated: instantiating needs the live shell, and the shell
builds the screen that runs these very checks."""


def _desktop_app_check() -> Check:
    """Whether the desktop app — the default UI — can actually open.

    Optional because the failure is survivable: every CLI subcommand works
    without a window, and this must not turn a headless machine's `steno
    doctor` into an exit 1.

    In a subprocess, because there is nothing safe to do in this one: a
    ``QGuiApplication`` may already exist (the app's own Doctor screen calls
    this), and creating one where none does aborts the process outright on a
    machine with no display. The child is pinned to the offscreen platform, so
    a missing screen is never what it reports.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _QML_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
    except subprocess.TimeoutExpired:
        # Not the exception's own message: it quotes the command, and the
        # command is the whole probe source.
        return Check(
            name="Desktop app",
            ok=False,
            detail="the Qt probe did not finish in 180 s",
            optional=True,
        )
    except OSError as exc:
        return Check(
            name="Desktop app", ok=False, detail=f"the Qt probe did not run: {exc}", optional=True
        )
    if probe.returncode == 0:
        return Check(
            name="Desktop app",
            ok=True,
            detail="Qt loads and the interface compiles",
            optional=True,
        )
    return Check(
        name="Desktop app",
        ok=False,
        detail=f"{_probe_reason(probe.stderr)} — the CLI (`steno start`, "
        "`steno transcribe`) works without it; to repair the window, reinstall with "
        "`uv tool install --force stenograf`",
        optional=True,
    )


def _probe_reason(stderr: str) -> str:
    """The one sentence worth printing out of a failed probe's stderr.

    Where it sits depends on how Qt failed. A short block — an aborted Qt, a
    QML error — is worth quoting whole, since its last line is often only a
    list of alternatives. A traceback is not, and its final line is not
    reliably the message either: a `dlopen` failure ends in indented
    continuation lines listing every path tried. The last unindented line is
    the exception in both cases."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return "no reason given"
    if len(lines) <= 5:
        return "; ".join(line.strip() for line in lines)
    return next((line.strip() for line in reversed(lines) if not line[:1].isspace()), lines[-1])


def _codesign_valid(path: Path) -> tuple[bool, str]:
    """Whether ``codesign --verify`` accepts the binary (ad-hoc signatures pass)."""
    try:
        proc = subprocess.run(
            ["codesign", "--verify", str(path)], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"codesign unavailable: {exc}"
    if proc.returncode == 0:
        return True, ""
    lines = proc.stderr.strip().splitlines()
    return False, lines[-1] if lines else "invalid signature"


def _asr_check() -> Check:
    from stenograf.asr import backend_model_id, get_spec

    spec = get_spec()  # the default backend (STENOGRAF_ASR_BACKEND override applies)
    missing = [module for module in spec.requires if not installed(module)]
    if not missing:
        model = backend_model_id(spec)
        detail = f"{spec.label} ready"
        if model:
            detail += f" ({model}; weights from HuggingFace on first use)"
        detail += _asr_ep_note(spec)
        return Check(name="ASR backend", ok=True, detail=detail)
    return Check(
        name="ASR backend",
        ok=False,
        detail=f"{spec.label} not installed (missing: {', '.join(missing)}) — reinstall "
        "stenograf, or select another backend via [asr] backend in settings.toml "
        "or STENOGRAF_ASR_BACKEND",
    )


def _asr_ep_note(spec: BackendSpec) -> str:
    """Provider status for the ORT-backed backend: what's configured, or what
    acceleration the installed onnxruntime flavor offers but isn't being used.
    A broken settings file is _settings_check's finding, not this one's."""
    if spec.name != "parakeet-onnx":
        return ""
    from stenograf.asr.ep import (
        EP_LABELS,
        available_accelerators,
        default_ep,
        validate_ep,
    )
    from stenograf.settings import SettingsError, load_settings

    try:
        configured = load_settings().asr.ep
    except SettingsError:
        configured = None
    try:
        ep = validate_ep(default_ep(configured))
        accelerated = available_accelerators()
    except (ValueError, ImportError) as exc:
        return f"; provider: {exc}"
    if ep != "cpu":
        label = EP_LABELS.get(ep, "best available")
        return f"; provider {ep} ({label}, CPU fallback)"
    if accelerated:
        return (
            f"; CPU — {EP_LABELS[accelerated[0]]} available: set [asr] "
            f'provider = "{accelerated[0]}" to accelerate'
        )
    return ""


def _ffmpeg_check() -> Check:
    """The decoder ships in the wheel (imageio-ffmpeg); failure here means a
    broken install (or an unsupported platform), not a missing system tool."""
    from stenograf.audio import ffmpeg_exe

    try:
        path = ffmpeg_exe()
    except Exception as exc:
        return Check(
            name="ffmpeg",
            ok=False,
            detail=f"bundled ffmpeg unavailable ({exc}) — reinstall stenograf, or "
            "point IMAGEIO_FFMPEG_EXE at an ffmpeg binary",
        )
    return Check(name="ffmpeg", ok=True, detail=path)


def _models_check() -> Check:
    from stenograf import assets
    from stenograf.paths import cache_dir

    wanted = (assets.SILERO_VAD, assets.PYANNOTE_SEGMENTATION, assets.SPEAKER_EMBEDDING)
    missing = [asset.name for asset in wanted if assets.cached_path(asset) is None]
    if missing:
        detail = (
            f"{len(missing)}/{len(wanted)} pending — `steno setup` downloads them "
            "(or they download on first use): " + ", ".join(missing)
        )
    else:
        detail = f"VAD + diarization cached in {cache_dir()} (ASR weights via HuggingFace)"
    return Check(name="Models", ok=not missing, detail=detail)


def _preset_checks() -> list[Check]:
    """One check per meeting preset that changes the notes setup.

    ``_notes_check`` greens the *standing* backend — a preset selecting another
    one (say ``command`` while settings.toml runs mlx) would otherwise get a
    green doctor and a failed notes run after a real meeting. Each backend's
    ``health()`` owns the diagnosis (a command backend resolves ``argv[0]``
    under the *effective* PATH); the app-launch hint matters most from the app
    bundle: launchd's PATH is widened but no shell rc runs, so a binary (or an
    exported credential) visible in a terminal may not exist there —
    ``STENOGRAF_APP_BUNDLE`` marks that context."""
    from stenograf.notes import NotesBackendError, create_backend
    from stenograf.settings import SettingsError, apply_meeting_preset, load_settings

    try:
        settings = load_settings()
    except SettingsError:
        return []  # _settings_check already reports the broken file
    checks = []
    for name in sorted(settings.meetings):
        preset = settings.meetings[name]
        if preset.notes == type(preset.notes)():
            continue  # no notes overlay — the standing check covers it
        check_name = f"Preset '{name}' notes (optional)"
        try:
            overlaid, _preset = apply_meeting_preset(settings, name)
            backend = create_backend(preset.notes.backend, overlaid.notes)
        except (SettingsError, NotesBackendError, ValueError) as exc:
            checks.append(Check(name=check_name, ok=False, detail=str(exc), optional=True))
            continue
        ok, detail = backend.health()
        if not ok and _app_launch():
            detail += " (app launch: shell PATH/env not sourced)"
        checks.append(Check(name=check_name, ok=ok, detail=detail, optional=True))
    return checks


def _app_launch() -> bool:
    """Whether this process was started from the app bundle (no shell rc ran)."""
    import os

    return bool(os.environ.get("STENOGRAF_APP_BUNDLE"))


def _notes_check() -> Check:
    """Whether the *configured* notes backend could run (`steno notes`, `--notes`).

    Notes are opt-in, so this check is ``optional``: an absent Ollama or an
    unconfigured command never fails the overall doctor gate — it only tells
    the user what `--notes` would need."""
    from stenograf.notes import NotesBackendError, create_backend
    from stenograf.settings import SettingsError, load_settings

    name = "Notes backend (optional)"
    try:
        settings = load_settings().notes
        backend = create_backend(None, settings)
    except (SettingsError, NotesBackendError, ValueError) as exc:
        return Check(name=name, ok=False, detail=str(exc), optional=True)
    ok, detail = backend.health()
    return Check(name=name, ok=ok, detail=detail, optional=True)


def _macos_version_check() -> Check:
    release = platform.mac_ver()[0]
    try:
        version = tuple(int(part) for part in release.split("."))
    except ValueError:
        return Check(name="macOS", ok=False, detail=f"unrecognized version {release!r}")
    ok = version >= MACOS_MIN_VERSION
    min_str = ".".join(map(str, MACOS_MIN_VERSION))
    detail = release if ok else f"{release} — {min_str}+ required for system-audio capture"
    return Check(name="macOS", ok=ok, detail=detail)
