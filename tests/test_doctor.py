import platform
from pathlib import Path

from stenograf import doctor


def test_run_checks_includes_python_and_asr():
    names = {c.name for c in doctor.run_checks()}
    assert "Python" in names
    assert "ASR backend" in names
    python = next(c for c in doctor.run_checks() if c.name == "Python")
    assert python.ok  # we require >=3.12 and run under it


def test_capture_helper_check_reports_found(monkeypatch, tmp_path):
    from stenograf.capture import macos

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    monkeypatch.setattr(macos, "find_helper", lambda: helper)
    check = doctor._capture_helper_check()
    assert check.ok
    assert str(helper) in check.detail
    assert "permission" in check.detail  # first-run permission guidance surfaced


def test_capture_helper_check_reports_missing(monkeypatch):
    from stenograf.capture import macos

    def boom():
        raise macos.HelperNotFoundError("helper not found: build it")

    monkeypatch.setattr(macos, "find_helper", boom)
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "build it" in check.detail


def test_asr_check_present_and_absent(monkeypatch):
    monkeypatch.setattr(doctor, "_installed", lambda module: True)
    assert doctor._asr_check().ok
    monkeypatch.setattr(doctor, "_installed", lambda module: False)
    absent = doctor._asr_check()
    assert not absent.ok
    assert "parakeet-mlx" in absent.detail


def test_macos_version_check_parses_and_compares(monkeypatch):
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.4", ("", "", ""), ""))
    assert doctor._macos_version_check().ok
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.3", ("", "", ""), ""))
    assert not doctor._macos_version_check().ok
    monkeypatch.setattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert not doctor._macos_version_check().ok  # unparseable → not ok


def test_ffmpeg_check_follows_path(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert doctor._ffmpeg_check().ok
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    assert not doctor._ffmpeg_check().ok


def test_models_check_reflects_cache(monkeypatch):
    from stenograf import models

    monkeypatch.setattr(models, "cached_path", lambda asset: Path("/cached"))
    assert doctor._models_check().ok
    monkeypatch.setattr(models, "cached_path", lambda asset: None)
    missing = doctor._models_check()
    assert not missing.ok
    assert "pending" in missing.detail
