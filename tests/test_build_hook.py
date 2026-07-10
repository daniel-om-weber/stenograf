"""Unit tests for the wheel build hook (hatch_build.py, PLAN.md Stage E1).

The hook lives at the repo root (it is the hatchling entry point, not package
code), so it is loaded by path rather than imported.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).resolve().parents[1] / "hatch_build.py"


@pytest.fixture(scope="module")
def hook_module():
    spec = importlib.util.spec_from_file_location("hatch_build", _HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_hook(hook_module, root: Path, target_name: str = "wheel"):
    return hook_module.CustomBuildHook(str(root), {}, None, None, str(root / "dist"), target_name)


def build_data() -> dict:
    return {"force_include": {}, "pure_python": True}


def test_noop_off_macos_arm64(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)
    assert data == build_data()  # untouched → pure py3-none-any wheel


@pytest.mark.parametrize("version,target", [("editable", "wheel"), ("standard", "sdist")])
def test_noop_for_editable_and_sdist(hook_module, tmp_path, monkeypatch, version, target):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    data = build_data()
    make_hook(hook_module, tmp_path, target_name=target).initialize(version, data)
    assert data == build_data()


def _fake_build(tmp_path):
    """A subprocess.run stand-in that 'builds' whichever helper's build.sh ran."""

    def fake(cmd, check):
        assert check
        script = Path(cmd[-1])
        assert script.name == "build.sh"
        binary = script.parent / ("stenocap" if script.parent.name == "helper" else "stenodiar")
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"\x00")
        binary.chmod(0o644)  # compilers emit 0o755; prove the hook re-asserts it
        return subprocess.CompletedProcess(cmd, 0)

    return fake


def test_bundles_both_helpers_on_macos_arm64(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    monkeypatch.setattr(hook_module, "_cargo_available", lambda: True)
    monkeypatch.setattr(hook_module.subprocess, "run", _fake_build(tmp_path))
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)

    assert data["pure_python"] is False
    assert data["tag"] == hook_module.WHEEL_TAG
    helper = tmp_path / "native" / "helper" / "stenocap"
    stenodiar = tmp_path / "native" / "stenodiar" / "stenodiar"
    assert data["force_include"][str(helper)] == "stenograf/bin/stenocap"
    assert data["force_include"][str(stenodiar)] == "stenograf/bin/stenodiar"
    for binary in (helper, stenodiar):
        assert binary.stat().st_mode & 0o111 == 0o111


def test_missing_cargo_skips_stenodiar_but_still_ships(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    monkeypatch.setattr(hook_module, "_cargo_available", lambda: False)
    monkeypatch.setattr(hook_module.subprocess, "run", _fake_build(tmp_path))
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)

    assert data["pure_python"] is False  # the wheel is still platform-tagged
    assert list(data["force_include"].values()) == ["stenograf/bin/stenocap"]


def test_failed_build_raises_not_silently_pure(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)

    def fail(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hook_module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Xcode Command Line Tools"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())


def test_failed_stenodiar_build_raises_when_cargo_present(hook_module, tmp_path, monkeypatch):
    # A present-but-broken Rust toolchain is an environment error, not an
    # optional to skip: stenocap builds, stenodiar's build.sh fails.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    monkeypatch.setattr(hook_module, "_cargo_available", lambda: True)
    ok = _fake_build(tmp_path)

    def fake(cmd, check):
        if Path(cmd[-1]).parent.name == "stenodiar":
            raise subprocess.CalledProcessError(1, cmd)
        return ok(cmd, check)

    monkeypatch.setattr(hook_module.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="Rust toolchain"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())


def test_build_without_binary_raises(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    monkeypatch.setattr(
        hook_module.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 0)
    )
    with pytest.raises(RuntimeError, match="not produced"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())
