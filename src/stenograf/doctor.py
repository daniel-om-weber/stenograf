"""Environment checks behind ``steno doctor`` — the first-run experience."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass

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
        # Optional: `steno transcribe` is fully supported everywhere (Phase 5's
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

    checks.append(_asr_check())
    checks.append(_ffmpeg_check())
    checks.append(_models_check())
    checks.append(_settings_check())
    checks.append(_notes_check())
    return checks


def _settings_check() -> Check:
    """Whether settings.toml (if present) parses and validates whole.

    Not optional: every command now resolves its defaults from the file at
    startup, so a broken file blocks ``steno start`` itself."""
    from stenograf.settings import SettingsError, load_settings, settings_path

    path = settings_path()
    try:
        load_settings()
    except SettingsError as exc:
        return Check(name="Settings", ok=False, detail=str(exc))
    if not path.exists():
        return Check(name="Settings", ok=True, detail=f"{path} not present — all defaults")
    return Check(name="Settings", ok=True, detail=f"{path} OK")


def _installed(module: str) -> bool:
    """Whether ``module`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _capture_helper_check() -> Check:
    from stenograf.capture.macos import HelperNotFoundError, find_helper

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
            "permissions to unsigned binaries; rebuild with native/helper/build.sh",
        )
    return Check(
        name="Capture helper",
        ok=True,
        detail=f"{path} — signed; grant the mic + system-audio permission once with `steno setup`",
    )


def _linux_capture_check() -> Check:
    """Whether `steno start` can capture here: parec present, server up, defaults set.

    Uses the same resolution the provider uses at meeting start, so an OK here
    means a meeting would actually record — and names the devices it would
    record from (the monitor-of-default-sink choice is invisible otherwise).
    """
    from stenograf.capture.base import Channel
    from stenograf.capture.linux import (
        CaptureUnavailableError,
        LinuxCaptureProvider,
        default_devices,
    )

    try:
        LinuxCaptureProvider()  # fails fast when parec is missing
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
    except CaptureUnavailableError as exc:
        return Check(name="Capture", ok=False, detail=str(exc))
    listing = ", ".join(f"{ch.value} ← {device}" for ch, device in sorted(devices.items()))
    return Check(name="Capture", ok=True, detail=f"parec via PipeWire/PulseAudio ({listing})")


def _windows_capture_check() -> Check:
    """Whether `steno start` can capture here: soundcard importable, defaults set.

    Uses the same resolution the provider uses at meeting start, so an OK here
    means a meeting would actually record — and names the devices it would
    record from (the loopback-of-default-output choice is invisible otherwise).
    """
    from stenograf.capture.base import Channel
    from stenograf.capture.windows import (
        CaptureUnavailableError,
        WindowsCaptureProvider,
        default_devices,
    )

    try:
        WindowsCaptureProvider()  # fails fast when soundcard is missing
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
    except CaptureUnavailableError as exc:
        return Check(name="Capture", ok=False, detail=str(exc))
    listing = ", ".join(f"{ch.value} ← {device}" for ch, device in sorted(devices.items()))
    return Check(name="Capture", ok=True, detail=f"WASAPI via soundcard ({listing})")


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


def _codesign_valid(path) -> tuple[bool, str]:
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
    missing = [module for module in spec.requires if not _installed(module)]
    if not missing:
        model = backend_model_id(spec)
        detail = f"{spec.label} ready"
        if model:
            detail += f" ({model}; weights from HuggingFace on first use)"
        detail += _asr_provider_note(spec)
        return Check(name="ASR backend", ok=True, detail=detail)
    return Check(
        name="ASR backend",
        ok=False,
        detail=f"{spec.label} not installed (missing: {', '.join(missing)}) — reinstall "
        "stenograf, or select another backend via [asr] backend in settings.toml "
        "or STENOGRAF_ASR_BACKEND",
    )


def _asr_provider_note(spec) -> str:
    """Provider status for the ORT-backed backend: what's configured, or what
    acceleration the installed onnxruntime flavor offers but isn't being used.
    A broken settings file is _settings_check's finding, not this one's."""
    if spec.name != "parakeet-onnx":
        return ""
    from stenograf.asr.providers import (
        PROVIDER_LABELS,
        available_accelerators,
        default_provider_name,
        validate_provider_name,
    )
    from stenograf.settings import SettingsError, load_settings

    try:
        configured = load_settings().asr.provider
    except SettingsError:
        configured = None
    try:
        provider = validate_provider_name(default_provider_name(configured))
        accelerated = available_accelerators()
    except (ValueError, ImportError) as exc:
        return f"; provider: {exc}"
    if provider != "cpu":
        label = PROVIDER_LABELS.get(provider, "best available")
        return f"; provider {provider} ({label}, CPU fallback)"
    if accelerated:
        return (
            f"; CPU — {PROVIDER_LABELS[accelerated[0]]} available: set [asr] "
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
    from stenograf import models

    assets = (models.SILERO_VAD, models.PYANNOTE_SEGMENTATION, models.SPEAKER_EMBEDDING)
    missing = [asset.name for asset in assets if models.cached_path(asset) is None]
    if missing:
        detail = (
            f"{len(missing)}/{len(assets)} pending — `steno setup` downloads them "
            "(or they download on first use): " + ", ".join(missing)
        )
    else:
        detail = f"VAD + diarization cached in {models.cache_dir()} (ASR weights via HuggingFace)"
    return Check(name="Models", ok=not missing, detail=detail)


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

    from stenograf.notes.mlx import MlxBackend
    from stenograf.notes.ollama import OllamaBackend

    if isinstance(backend, MlxBackend):
        if not backend.is_available():
            return Check(
                name=name,
                ok=False,
                detail="mlx-lm is not installed here — reinstall stenograf, or configure "
                "another backend under [notes] in settings.toml",
                optional=True,
            )
        hint = "cached" if backend.weights_cached() else "downloads on first notes run"
        return Check(name=name, ok=True, detail=f"MLX in-process, model {backend.model} ({hint})")
    if isinstance(backend, OllamaBackend):
        if not backend.is_available():
            return Check(
                name=name,
                ok=False,
                detail=f"Ollama not reachable at {backend.url} — start `ollama serve`, or "
                "configure another backend under [notes] in settings.toml",
                optional=True,
            )
        try:
            installed = backend.installed_models()
        except NotesBackendError as exc:
            return Check(name=name, ok=False, detail=str(exc), optional=True)
        names = set(installed) | {m.split(":", 1)[0] for m in installed}
        if backend.model not in names:
            return Check(
                name=name,
                ok=False,
                detail=f"Ollama up, but model {backend.model!r} is not pulled "
                f"(`ollama pull {backend.model}`)",
                optional=True,
            )
        return Check(name=name, ok=True, detail=f"Ollama at {backend.url}, model {backend.model}")
    if not backend.is_available():
        argv0 = getattr(backend, "argv", ("?",))[0]
        return Check(
            name=name,
            ok=False,
            detail=f"notes command {argv0!r} is not on PATH",
            optional=True,
        )
    label = " ".join(getattr(backend, "argv", (backend.name,)))
    return Check(name=name, ok=True, detail=f"command backend: {label}")


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
