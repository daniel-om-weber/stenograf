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


def test_bundles_helper_on_macos_arm64(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    helper = tmp_path / "native" / "helper" / "stenocap"

    def fake_build(cmd, check):
        assert check
        assert cmd[-1].endswith("build.sh")
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_bytes(b"\x00")
        helper.chmod(0o644)  # swiftc always emits 0o755; prove the hook re-asserts it
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(hook_module.subprocess, "run", fake_build)
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)

    assert data["pure_python"] is False
    assert data["tag"] == hook_module.WHEEL_TAG
    assert data["force_include"][str(helper)] == "stenograf/bin/stenocap"
    assert helper.stat().st_mode & 0o111 == 0o111


def test_failed_build_raises_not_silently_pure(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)

    def fail(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hook_module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Xcode Command Line Tools"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())


def test_build_without_binary_raises(hook_module, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    monkeypatch.setattr(
        hook_module.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 0)
    )
    with pytest.raises(RuntimeError, match="not produced"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())
