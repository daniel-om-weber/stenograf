"""`steno setup` and model prefetch: permissions, the privacy toggle,
and the loaders' refusal paths."""

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from stenograf import cli, loaders


def _helper_wrapper(tmp_path, *forced_args):
    """An executable stand-in for stenocap; forced_args replace the real argv."""
    fake = Path(__file__).parent / "fake_stenocap.py"
    args = " ".join(forced_args) if forced_args else '"$@"'
    script = tmp_path / "stenocap"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" {args}\n')
    script.chmod(0o755)
    return script


@pytest.mark.skipif(sys.platform != "darwin", reason="steno setup is macOS-only")
def test_setup_grants_permissions_then_prefetches(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # the launcher lands in $HOME/Applications
    monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(_helper_wrapper(tmp_path)))
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code == 0, result.output
    assert "granted" in result.output
    assert fetched  # downloads run after the permission step
    assert "launcher installed" in result.output
    assert (tmp_path / "Applications" / "Stenograf.app").is_dir()
    # The grant just taken belongs to this terminal; the app is its own TCC
    # client and asks once more. Saying so is the whole point of the line.
    assert "asks for microphone access once" in result.output
    assert "setup complete" in result.output


@pytest.mark.skipif(sys.platform != "darwin", reason="steno setup is macOS-only")
def test_setup_fails_when_helper_dies_without_mic_frames(tmp_path, monkeypatch):
    # A denied permission means the helper exits before its first mic frame;
    # emitting only system frames then exiting reproduces that shape.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(_helper_wrapper(tmp_path, "--system")))
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code != 0
    assert "denied" in result.output
    assert not fetched  # no downloads on a failed permission grant


@pytest.mark.skipif(sys.platform != "win32", reason="the mic privacy toggle is Windows-only")
def test_setup_windows_checks_the_privacy_toggle_then_prefetches(tmp_path, monkeypatch):
    from stenograf import shortcut
    from stenograf.capture import windows as capture_windows

    monkeypatch.setattr(capture_windows, "mic_access_blocked", lambda: None)
    # Both shell folders, or setup writes into the developer's own Start menu.
    monkeypatch.setattr(shortcut, "_windows_desktop", lambda: tmp_path / "Desktop")
    monkeypatch.setattr(shortcut, "_windows_programs", lambda: tmp_path / "Programs")
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code == 0, result.output
    assert "microphone access is allowed" in result.output
    assert "shows no permission prompt" in result.output  # no prompt is ever coming
    assert "launcher installed" in result.output
    # A Start-menu entry to pin, plus the Desktop copy.
    assert (tmp_path / "Programs" / "Stenograf.lnk").exists()
    assert (tmp_path / "Desktop" / "Stenograf.lnk").exists()
    assert "Start menu" in result.output
    assert fetched


@pytest.mark.skipif(sys.platform != "win32", reason="the mic privacy toggle is Windows-only")
def test_setup_windows_fails_when_the_privacy_toggle_denies(monkeypatch):
    # Mirrors the macOS denied-permission shape: fail before any download,
    # naming the settings page (the fake reason stands in for the real one).
    from stenograf.capture import windows as capture_windows

    monkeypatch.setattr(
        capture_windows,
        "mic_access_blocked",
        lambda: "microphone access is turned off in Windows privacy settings",
    )
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup"])
    assert result.exit_code != 0
    assert "privacy settings" in result.output
    assert "re-run `steno setup`" in result.output
    assert not fetched  # no downloads when the toggle denies capture


def test_setup_models_only_skips_the_permission_step(monkeypatch):
    # No STENOGRAF_CAPTURE_HELPER and no fake helper: reaching the permission
    # code would fail loudly, so success proves it was skipped. Runs on any OS.
    monkeypatch.delenv("STENOGRAF_CAPTURE_HELPER", raising=False)
    fetched = []
    monkeypatch.setattr(loaders, "prefetch_models", lambda: fetched.append(True))
    result = CliRunner().invoke(cli.main, ["setup", "--models-only"])
    assert result.exit_code == 0, result.output
    assert fetched
    assert "granted" not in result.output
    assert "launcher" not in result.output  # headless machines get no shortcut


def test_prefetch_models_downloads_missing_and_loads_asr(monkeypatch, tmp_path):
    from stenograf import models
    from stenograf.asr.base import ASRBackend

    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    # One asset pre-cached, the rest missing: only the missing ones are fetched.
    (tmp_path / models.SILERO_VAD.name).write_bytes(b"\x00")
    fetched = []
    monkeypatch.setattr(models, "fetch", lambda asset, progress=None: fetched.append(asset.name))

    class PrefetchASR(ASRBackend):
        name = "fake"
        model_id = "fake/model"
        calls = []

        def load(self):
            self.calls.append("load")

        def transcribe(self, samples, language):
            return []

        def unload(self):
            self.calls.append("unload")

    import stenograf.asr as asr
    from stenograf import doctor

    monkeypatch.setattr(doctor, "installed", lambda module: True)  # deps "present" (any OS)
    monkeypatch.setattr(asr, "create_backend", lambda name=None, **kw: PrefetchASR())
    loaders.prefetch_models()
    assert set(fetched) == {models.PYANNOTE_SEGMENTATION.name, models.SPEAKER_EMBEDDING.name}
    assert PrefetchASR.calls == ["load", "unload"]  # weights pulled and released


def test_prefetch_models_skips_asr_when_backend_deps_absent(monkeypatch, tmp_path, capsys):
    from stenograf import doctor, models

    monkeypatch.setenv("STENOGRAF_CACHE", str(tmp_path))
    monkeypatch.setattr(models, "fetch", lambda asset, progress=None: None)
    monkeypatch.setattr(doctor, "installed", lambda module: False)  # the Linux shape
    import stenograf.asr as asr

    def boom(name=None, **kw):
        raise AssertionError("create_backend must not run without its deps")

    monkeypatch.setattr(asr, "create_backend", boom)
    loaders.prefetch_models()  # must not raise
    assert "skipping its weights" in capsys.readouterr().out


def test_load_backends_refuses_uninstalled_backend(monkeypatch):
    """A selected backend whose runtime is absent must be a CLI error, not an
    import traceback from deep inside ``asr.load()``."""
    import click

    from stenograf import doctor

    monkeypatch.setattr(doctor, "installed", lambda module: False)
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    with pytest.raises(click.ClickException, match="parakeet-mlx is not installed"):
        loaders.load_backends(need_diarizer=False)


def test_load_backends_refuses_unknown_backend(monkeypatch):
    import click

    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "no-such-backend")
    with pytest.raises(click.ClickException, match="unknown ASR backend"):
        loaders.load_backends(need_diarizer=False)
