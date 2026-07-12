import dataclasses
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from stenograf import models


class _FakeResponse(io.BytesIO):
    """Stands in for urlopen's response: context manager, headers, chunked read."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def serve(monkeypatch, payload: bytes) -> None:
    """Point models' urlopen at a canned payload."""

    def fake_urlopen(url, timeout=None):
        assert timeout is not None  # a hung server must not stall fetch forever
        return _FakeResponse(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)


def asset_for(base: models.ModelAsset, payload: bytes) -> models.ModelAsset:
    """The asset with its digest matching the canned payload."""
    return dataclasses.replace(base, sha256=hashlib.sha256(payload).hexdigest())


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    assert models.cache_dir() == tmp_path


def test_cache_dir_macos_default(monkeypatch):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(models.sys, "platform", "darwin")
    assert models.cache_dir() == Path.home() / "Library" / "Caches" / "stenograf"


def test_cache_dir_xdg_default(monkeypatch, tmp_path):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(models.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert models.cache_dir() == tmp_path / "stenograf"


def test_cache_dir_windows_default(monkeypatch, tmp_path):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(models.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert models.cache_dir() == tmp_path / "stenograf" / "cache"


def test_cached_path_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    assert models.cached_path(models.SILERO_VAD) is None


def test_fetch_downloads_plain_file_once(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    payload = b"onnx-model-bytes"
    asset = asset_for(models.SILERO_VAD, payload)
    serve(monkeypatch, payload)
    seen: list[tuple[str, int, int]] = []

    def record(name, done, total):
        seen.append((name, done, total))

    path = models.fetch(asset, record)

    assert path == tmp_path / asset.name
    assert path.read_bytes() == payload
    assert seen and seen[-1] == (asset.name, len(payload), len(payload))  # progress fired
    assert models.cached_path(asset) == path

    # A second fetch is served from cache and must not re-download.
    def explode(*args, **kwargs):
        raise AssertionError("re-downloaded an already-cached asset")

    monkeypatch.setattr(models.urllib.request, "urlopen", explode)
    assert models.fetch(asset) == path


def test_fetch_rejects_a_corrupted_download(monkeypatch, tmp_path):
    # A complete-looking but wrong payload (CDN error page, corrupted transfer)
    # must never reach the cache, or it would pass the exists() check forever.
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    serve(monkeypatch, b"<html>503 Service Unavailable</html>")
    with pytest.raises(RuntimeError, match="integrity"):
        models.fetch(models.SILERO_VAD)
    assert models.cached_path(models.SILERO_VAD) is None
    assert not list(tmp_path.glob("*.part"))


def test_fetch_extracts_archive_member(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    inner = b"segmentation-onnx"

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo(models.PYANNOTE_SEGMENTATION.archive_member)
        info.size = len(inner)
        tar.addfile(info, io.BytesIO(inner))
    archive = buffer.getvalue()
    asset = asset_for(models.PYANNOTE_SEGMENTATION, archive)  # digest covers the archive

    serve(monkeypatch, archive)
    path = models.fetch(asset)
    assert path == tmp_path / asset.name
    assert path.read_bytes() == inner  # the member, not the archive


def test_fetch_missing_member_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo("unexpected/other.onnx")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    archive = buffer.getvalue()
    asset = asset_for(models.PYANNOTE_SEGMENTATION, archive)

    serve(monkeypatch, archive)
    with pytest.raises(RuntimeError):
        models.fetch(asset)
    # The failed download left nothing behind the "already cached" check.
    assert models.cached_path(asset) is None
    assert not list(tmp_path.glob("*.part"))  # no extraction temp left behind


def test_interrupted_extraction_leaves_no_truncated_model(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo(models.PYANNOTE_SEGMENTATION.archive_member)
        payload = b"x" * (1 << 16)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    # Cut the archive mid-stream: decompression dies while the member is being
    # written out, exactly the crash window the atomic rename protects. The
    # digest matches the truncated bytes so the failure is extraction's, not
    # the integrity check's.
    truncated = buffer.getvalue()[: buffer.tell() // 2]
    asset = asset_for(models.PYANNOTE_SEGMENTATION, truncated)

    serve(monkeypatch, truncated)
    with pytest.raises(Exception):  # noqa: B017 - bz2 raises OSError/EOFError variants
        models.fetch(asset)
    assert models.cached_path(asset) is None  # nothing behind the "already cached" check
    assert not list(tmp_path.glob("*.part"))
