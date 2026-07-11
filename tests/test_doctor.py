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
    helper.chmod(0o755)
    monkeypatch.setattr(macos, "find_helper", lambda: helper)
    monkeypatch.setattr(doctor, "_codesign_valid", lambda path: (True, ""))
    check = doctor._capture_helper_check()
    assert check.ok
    assert str(helper) in check.detail
    assert "permission" in check.detail  # first-run permission guidance surfaced


def test_capture_helper_check_rejects_non_executable(monkeypatch, tmp_path):
    from stenograf.capture import macos

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    helper.chmod(0o644)
    monkeypatch.setattr(macos, "find_helper", lambda: helper)
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "executable" in check.detail


def test_capture_helper_check_rejects_bad_signature(monkeypatch, tmp_path):
    from stenograf.capture import macos

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    helper.chmod(0o755)
    monkeypatch.setattr(macos, "find_helper", lambda: helper)
    monkeypatch.setattr(doctor, "_codesign_valid", lambda path: (False, "code has no signature"))
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "signature" in check.detail
    assert "build.sh" in check.detail  # points at the fix


def test_codesign_valid_against_real_codesign(tmp_path):
    # An arbitrary file is not validly signed; the real codesign must say so.
    unsigned = tmp_path / "not-a-binary"
    unsigned.write_bytes(b"\x00" * 16)
    ok, why = doctor._codesign_valid(unsigned)
    assert not ok
    assert why  # carries codesign's reason (or its absence off-macOS)


def test_capture_helper_check_reports_missing(monkeypatch):
    from stenograf.capture import macos

    def boom():
        raise macos.HelperNotFoundError("helper not found: build it")

    monkeypatch.setattr(macos, "find_helper", boom)
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "build it" in check.detail


def test_asr_check_present_and_absent(monkeypatch):
    # Pin the backend under test: the built-in default is capability-based,
    # so the bare default names a different backend per machine.
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    monkeypatch.setattr(doctor, "_installed", lambda module: True)
    assert doctor._asr_check().ok
    monkeypatch.setattr(doctor, "_installed", lambda module: False)
    absent = doctor._asr_check()
    assert not absent.ok
    assert "parakeet-mlx" in absent.detail
    assert "parakeet_mlx" in absent.detail  # the missing modules are named


def test_non_darwin_platform_check_is_optional(monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    platform_check = next(c for c in doctor.run_checks() if c.name == "Platform")
    assert not platform_check.ok
    assert platform_check.optional  # transcribe is supported; only live capture is missing
    assert "transcribe" in platform_check.detail


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


def test_notes_check_ollama_down_is_optional_not_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "ollama")  # force the branch under test
    from stenograf.notes.ollama import OllamaBackend

    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: False)
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional  # an absent backend must not fail the doctor gate
    assert "ollama serve" in check.detail


def test_notes_check_ollama_up_but_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "ollama")  # force the branch under test
    from stenograf.notes.ollama import OllamaBackend

    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: True)
    monkeypatch.setattr(OllamaBackend, "installed_models", lambda self: ["llama3:8b"])
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "ollama pull" in check.detail


def test_notes_check_mlx_reports_cache_state(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "mlx")
    from stenograf.notes.mlx import MlxBackend

    monkeypatch.setattr(MlxBackend, "is_available", lambda self: True)
    monkeypatch.setattr(MlxBackend, "weights_cached", lambda self: False)
    check = doctor._notes_check()
    assert check.ok
    assert "downloads on first" in check.detail

    monkeypatch.setattr(MlxBackend, "is_available", lambda self: False)
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "mlx-lm" in check.detail


def test_notes_check_command_backend_reports_path_presence(monkeypatch, tmp_path):
    import sys

    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'[notes]\nbackend = "command"\ncommand = ["{sys.executable}", "-c", "pass"]\n',
        encoding="utf-8",
    )
    check = doctor._notes_check()
    assert check.ok, check.detail

    settings.write_text(
        '[notes]\nbackend = "command"\ncommand = ["no-such-notes-binary"]\n', encoding="utf-8"
    )
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "PATH" in check.detail


def test_notes_check_unconfigured_command_backend_is_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)
    (tmp_path / "settings.toml").write_text('[notes]\nbackend = "command"\n', encoding="utf-8")
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "settings.toml" in check.detail


def test_doctor_exit_gate_ignores_optional_failures(monkeypatch):
    from click.testing import CliRunner

    from stenograf import cli

    ok = doctor.Check(name="A", ok=True, detail="fine")
    opt = doctor.Check(name="B", ok=False, detail="absent", optional=True)
    hard = doctor.Check(name="C", ok=False, detail="broken")

    monkeypatch.setattr(cli, "run_checks", lambda: [ok, opt])
    assert CliRunner().invoke(cli.main, ["doctor"]).exit_code == 0

    monkeypatch.setattr(cli, "run_checks", lambda: [ok, opt, hard])
    assert CliRunner().invoke(cli.main, ["doctor"]).exit_code == 1
