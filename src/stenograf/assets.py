"""Download and cache for the small ONNX model assets (VAD, diarization).

The ASR backends manage their own weights (parakeet-mlx pulls from
HuggingFace); this module covers the plain-file assets with no downloader of
their own — the sherpa-onnx VAD, the voice embedder and the Nemotron
diarization export, each pinned by URL and digest. Files land in one cache
(:func:`stenograf.paths.cache_dir`; ``$STENOGRAF_CACHE`` overrides it).
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from stenograf.paths import cache_dir

_DOWNLOAD_TIMEOUT_S = 30.0
"""Socket timeout per connect/read, not for the whole download — a stalled
server raises instead of hanging ``fetch`` forever."""

_CHUNK_BYTES = 1 << 20

ProgressHook = Callable[[str, int, int], None]
"""(asset name, bytes done, bytes total) — total is 0 when unknown."""


@dataclass(frozen=True)
class ModelAsset:
    name: str
    """Filename inside the cache directory."""
    url: str
    sha256: str
    """Digest of the downloaded bytes, verified before the file can reach the
    cache: without it, any complete-looking download — a CDN error page saved
    as ``.onnx``, a corrupted transfer — would pass the ``exists()`` check
    forever."""


SILERO_VAD = ModelAsset(
    name="silero_vad_v6.onnx",
    # Silero VAD v6 (MIT), the file shipped in the snakers4/silero-vad
    # package at tag v6.2.3. v5 missed most of a speakerphone voice on the
    # mic channel (eval/README.md, "Speech detector").
    url=(
        "https://github.com/snakers4/silero-vad/raw/"
        "5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx"
    ),
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)

_MODELS = "https://github.com/daniel-om-weber/stenograf/releases/download/models-1"
"""Files this project converted itself, attached to a release whose tag is
not a version: the publish workflow runs on ``v*`` tags only."""

SPEAKER_EMBEDDING = ModelAsset(
    name="redimnet2-b6-vb2-lm.onnx",
    # ReDimNet2-B6 (github.com/PalabraAI/redimnet2, checkpoint
    # b6-vb2+vox2_v0-lm, CC BY-NC-SA 4.0), exported with its front end by
    # eval/embedders/export_redimnet.py. Most accurate of twelve models at
    # naming short clusters (2026-11-18, eval/README.md). The name is the
    # profile store's model key: changing it retires every saved profile.
    url=f"{_MODELS}/redimnet2-b6-vb2-lm.onnx",
    sha256="ac1655c217d86b68e93c8a03ca8bbc16b4373016ae177765059dac38e77a2327",
)


_NEMOTRON = (
    "https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/resolve/"
    "353b6f8ad2cac3580e982d7fbdf0a010786b0406"
)
_NEMOTRON_DIR = "nemotron-3-diarization-onnx"
"""One directory for the three files: onnxruntime resolves the graph's
external weights by their stored filename, next to the graph."""

NEMOTRON_GRAPH = ModelAsset(
    name=f"{_NEMOTRON_DIR}/model.onnx",
    url=f"{_NEMOTRON}/onnx/model.onnx",
    sha256="12a7c98fc1ee6ec98ea728c7ffd9fcf6b245f3c9c1ea9f7c2a7cc47c023e3f9d",
)

NEMOTRON_WEIGHTS = ModelAsset(
    name=f"{_NEMOTRON_DIR}/model.onnx_data",
    url=f"{_NEMOTRON}/onnx/model.onnx_data",
    sha256="c293d9b5930eb9f6172f095ced052c0d1bbdbeb2594a115497583ca209b1dbd6",
)

NEMOTRON_CONFIG = ModelAsset(
    name=f"{_NEMOTRON_DIR}/config.json",
    url=f"{_NEMOTRON}/config.json",
    sha256="a758bf59867e27a7d92cf7a9149cae6dfc212b408b2ba74bfb87994708a803c5",
)

NEMOTRON = (NEMOTRON_GRAPH, NEMOTRON_WEIGHTS, NEMOTRON_CONFIG)
"""NVIDIA Nemotron 3 Diarization, fp32 ONNX export (OpenMDW-1.1; downloaded,
never redistributed). fp32, not the half-size fp16 export (398 vs 199 MB):
from onnxruntime 1.29 the CPU runs fp16 matmuls natively, and on an M4 Max
that doubles the time (fp16 33–35 vs fp32 15 s per audio-hour on 1.32, where
1.27–1.28 converted to fp32 and ran fp16 at 16–17); attribution on the
held-out AMI set is the same within 0.1 pt (2026-11-15, ``eval/README.md``)."""


def cached_path(asset: ModelAsset) -> Path | None:
    """Path of an already-downloaded asset, or ``None``."""
    path = cache_dir() / asset.name
    return path if path.exists() else None


@contextmanager
def _staged(target: Path) -> Iterator[Path]:
    """A sibling ``.part`` temp file, unlinked on exit unless renamed away.

    Work stages into the yielded path and ``replace``s it onto ``target``, so
    a crash never leaves a half-written file behind the "already cached" check."""
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch(asset: ModelAsset, progress: ProgressHook | None = None) -> Path:
    """Return the local path of ``asset``, downloading it on first use."""
    target = cache_dir() / asset.name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    with _staged(target) as tmp_path:
        _download(asset, tmp_path, progress)
        tmp_path.replace(target)
    return target


def _download(asset: ModelAsset, dest: Path, progress: ProgressHook | None) -> None:
    """Stream ``asset.url`` to ``dest``, verifying the digest as it arrives."""
    digest = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(asset.url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with open(dest, "wb") as out:
            while chunk := response.read(_CHUNK_BYTES):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(asset.name, done, total)
    if digest.hexdigest() != asset.sha256:
        raise RuntimeError(
            f"{asset.name}: download from {asset.url} failed its integrity check "
            f"(sha256 {digest.hexdigest()}, expected {asset.sha256}). The server "
            "may have sent an error page or the transfer was corrupted; re-run to retry."
        )
