"""Environment checks behind ``steno doctor`` — the first-run experience."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass

MACOS_MIN_VERSION = (14, 4)  # Core Audio process taps


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


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
        checks.append(
            Check(
                name="Capture helper",
                ok=False,
                detail="native Swift helper not built yet (planned: Phase 1)",
            )
        )
    else:
        checks.append(
            Check(
                name="Platform",
                ok=False,
                detail=f"{sys.platform}: only macOS is supported so far (Linux planned)",
            )
        )

    checks.append(_ffmpeg_check())
    checks.append(_models_check())
    return checks


def _ffmpeg_check() -> Check:
    path = shutil.which("ffmpeg")
    return Check(
        name="ffmpeg",
        ok=path is not None,
        detail=path or "not on PATH — needed to read anything but 16 kHz WAV "
        "(brew install ffmpeg)",
    )


def _models_check() -> Check:
    from stenograf import models

    assets = (models.SILERO_VAD, models.PYANNOTE_SEGMENTATION, models.SPEAKER_EMBEDDING)
    missing = [asset.name for asset in assets if models.cached_path(asset) is None]
    if missing:
        detail = f"{len(missing)}/{len(assets)} pending — downloaded on first use: " + ", ".join(
            missing
        )
    else:
        detail = f"VAD + diarization cached in {models.cache_dir()} (ASR weights via HuggingFace)"
    return Check(name="Models", ok=not missing, detail=detail)


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
