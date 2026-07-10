"""Download and cache for the small ONNX model assets (VAD, diarization).

The ASR backends manage their own weights (parakeet-mlx pulls from
HuggingFace); this module covers the sherpa-onnx assets, which have no
built-in downloader. Files land in one cache directory:

- ``$STENOGRAF_CACHE`` if set,
- ``~/Library/Caches/stenograf`` on macOS,
- ``$XDG_CACHE_HOME/stenograf`` (default ``~/.cache/stenograf``) elsewhere.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

ProgressHook = Callable[[str, int, int], None]
"""(asset name, bytes done, bytes total) — total is 0 when unknown."""


@dataclass(frozen=True)
class ModelAsset:
    name: str
    """Filename inside the cache directory."""
    url: str
    archive_member: str | None = None
    """For tar archives: the member to extract as ``name``."""


SILERO_VAD = ModelAsset(
    name="silero_vad_v5.onnx",
    url=f"{_RELEASES}/asr-models/silero_vad_v5.onnx",
)

PYANNOTE_SEGMENTATION = ModelAsset(
    name="pyannote-segmentation-3-0.onnx",
    url=f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    archive_member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
)

SPEAKER_EMBEDDING = ModelAsset(
    name="eres2net-voxceleb-16k.onnx",
    # Chosen empirically (July 2026, de-1 eval audio): the CAM++ VoxCeleb
    # export flips cluster identity between segmentation windows, shredding
    # one speaker into many; eres2net and titanet-small agree with each
    # other and match the audio. eres2net is the smaller of the two.
    url=f"{_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
)


def cache_dir() -> Path:
    if override := os.environ.get("STENOGRAF_CACHE"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "stenograf"
    xdg = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    return Path(xdg).expanduser() / "stenograf"


def cached_path(asset: ModelAsset) -> Path | None:
    """Path of an already-downloaded asset, or ``None``."""
    path = cache_dir() / asset.name
    return path if path.exists() else None


def fetch(asset: ModelAsset, progress: ProgressHook | None = None) -> Path:
    """Return the local path of ``asset``, downloading it on first use."""
    target = cache_dir() / asset.name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    def hook(blocks: int, block_size: int, total: int) -> None:
        if progress is not None:
            progress(asset.name, min(blocks * block_size, max(total, 0)), max(total, 0))

    # Download to a temp file in the same directory so a crash never leaves a
    # half-written model behind the "already cached" check.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(asset.url, tmp_path, reporthook=hook)
        if asset.archive_member is not None:
            _extract_member(tmp_path, asset.archive_member, target)
        else:
            tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _extract_member(archive: Path, member: str, target: Path) -> None:
    # Extract to a sibling temp file and rename, mirroring the download path:
    # an interrupted extraction must never leave a truncated model where
    # `cached_path` would report it as already downloaded.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(archive) as tar:
            try:
                src = tar.extractfile(member)  # KeyError if the name is absent
            except KeyError:
                src = None
            if src is None:  # missing, or the name is a directory/link
                raise RuntimeError(f"{archive.name}: no member {member!r}")
            with src, open(tmp_path, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)
