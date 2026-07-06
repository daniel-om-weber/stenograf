import io
import tarfile
from pathlib import Path

import pytest

from stenograf import models


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


def test_cached_path_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    assert models.cached_path(models.SILERO_VAD) is None


def test_fetch_downloads_plain_file_once(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    payload = b"onnx-model-bytes"

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(payload)
        if reporthook is not None:
            reporthook(1, len(payload), len(payload))
        return filename, None

    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake_urlretrieve)
    seen: list[tuple[str, int, int]] = []

    def record(name, done, total):
        seen.append((name, done, total))

    path = models.fetch(models.SILERO_VAD, record)

    assert path == tmp_path / models.SILERO_VAD.name
    assert path.read_bytes() == payload
    assert seen and seen[-1][0] == models.SILERO_VAD.name  # progress hook fired
    assert models.cached_path(models.SILERO_VAD) == path

    # A second fetch is served from cache and must not re-download.
    def explode(*args, **kwargs):
        raise AssertionError("re-downloaded an already-cached asset")

    monkeypatch.setattr(models.urllib.request, "urlretrieve", explode)
    assert models.fetch(models.SILERO_VAD) == path


def test_fetch_extracts_archive_member(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    asset = models.PYANNOTE_SEGMENTATION
    inner = b"segmentation-onnx"

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo(asset.archive_member)
        info.size = len(inner)
        tar.addfile(info, io.BytesIO(inner))
    archive = buffer.getvalue()

    def fake_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(archive)
        return filename, None

    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake_urlretrieve)
    path = models.fetch(asset)
    assert path == tmp_path / asset.name
    assert path.read_bytes() == inner  # the member, not the archive


def test_fetch_missing_member_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    asset = models.PYANNOTE_SEGMENTATION

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo("unexpected/other.onnx")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    archive = buffer.getvalue()

    monkeypatch.setattr(
        models.urllib.request,
        "urlretrieve",
        lambda url, filename, reporthook=None: (Path(filename).write_bytes(archive), None)[1],
    )
    with pytest.raises(RuntimeError):
        models.fetch(asset)
    # The failed download left nothing behind the "already cached" check.
    assert models.cached_path(asset) is None
