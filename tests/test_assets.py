import dataclasses
import hashlib
import io
from pathlib import Path

import pytest

from stenograf import assets, paths


class _FakeResponse(io.BytesIO):
    """Stands in for urlopen's response: context manager, headers, chunked read."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def serve(monkeypatch, payload: bytes) -> None:
    """Point assets' urlopen at a canned payload."""

    def fake_urlopen(url, timeout=None):
        assert timeout is not None  # a hung server must not stall fetch forever
        return _FakeResponse(payload)

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)


def asset_for(base: assets.ModelAsset, payload: bytes) -> assets.ModelAsset:
    """The asset with its digest matching the canned payload."""
    return dataclasses.replace(base, sha256=hashlib.sha256(payload).hexdigest())


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    assert paths.cache_dir() == tmp_path


def test_cache_dir_macos_default(monkeypatch):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.cache_dir() == Path.home() / "Library" / "Caches" / "stenograf"


def test_cache_dir_xdg_default(monkeypatch, tmp_path):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert paths.cache_dir() == tmp_path / "stenograf"


def test_cache_dir_windows_default(monkeypatch, tmp_path):
    monkeypatch.delenv("STENOGRAF_CACHE", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.cache_dir() == tmp_path / "stenograf" / "cache"


def test_cached_path_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    assert assets.cached_path(assets.SILERO_VAD) is None


def test_fetch_downloads_plain_file_once(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    payload = b"onnx-model-bytes"
    asset = asset_for(assets.SILERO_VAD, payload)
    serve(monkeypatch, payload)
    seen: list[tuple[str, int, int]] = []

    def record(name, done, total):
        seen.append((name, done, total))

    path = assets.fetch(asset, record)

    assert path == tmp_path / asset.name
    assert path.read_bytes() == payload
    assert seen and seen[-1] == (asset.name, len(payload), len(payload))  # progress fired
    assert assets.cached_path(asset) == path

    # A second fetch is served from cache and must not re-download.
    def explode(*args, **kwargs):
        raise AssertionError("re-downloaded an already-cached asset")

    monkeypatch.setattr(assets.urllib.request, "urlopen", explode)
    assert assets.fetch(asset) == path


def test_fetch_rejects_a_corrupted_download(monkeypatch, tmp_path):
    # A complete-looking but wrong payload (CDN error page, corrupted transfer)
    # must never reach the cache, or it would pass the exists() check forever.
    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    serve(monkeypatch, b"<html>503 Service Unavailable</html>")
    with pytest.raises(RuntimeError, match="integrity"):
        assets.fetch(assets.SILERO_VAD)
    assert assets.cached_path(assets.SILERO_VAD) is None
    assert not list(tmp_path.glob("*.part"))
