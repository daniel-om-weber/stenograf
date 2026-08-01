"""Unit tests for the wheel build hook (hatch_build.py).

The hook lives at the repo root (it is the hatchling entry point, not package
code), so it is loaded by path rather than imported.
"""

import importlib.util
import os
import subprocess
import sys
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


def _stenodiar(root: Path) -> Path:
    """Where the hook expects cargo's output — `.exe` when the suite runs on Windows."""
    name = "stenodiar.exe" if sys.platform == "win32" else "stenodiar"
    return root / "native" / "stenodiar" / name


def test_noop_off_macos_without_the_release_flag(hook_module, tmp_path, monkeypatch):
    # The common case: `pip install` from the sdist on Linux/Windows. Nothing
    # is built, nothing is bundled, and no cargo is required.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    monkeypatch.delenv(hook_module.BUNDLE_ENV, raising=False)
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)
    assert data == build_data()  # untouched → pure py3-none-any wheel


@pytest.mark.parametrize("version,target", [("editable", "wheel"), ("standard", "sdist")])
def test_noop_for_editable_and_sdist(hook_module, tmp_path, monkeypatch, version, target):
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: True)
    data = build_data()
    make_hook(hook_module, tmp_path, target_name=target).initialize(version, data)
    assert data == build_data()


@pytest.mark.parametrize(
    "platform_name,machine,tag_attr",
    [
        ("linux", "x86_64", "LINUX_TAG"),
        ("win32", "AMD64", "WINDOWS_TAG"),
    ],
)
def test_release_flag_selects_the_platform_tag(
    hook_module, monkeypatch, platform_name, machine, tag_attr
):
    monkeypatch.setenv(hook_module.BUNDLE_ENV, "1")
    monkeypatch.setattr(hook_module.sys, "platform", platform_name)
    monkeypatch.setattr(hook_module.platform, "machine", lambda: machine)
    assert hook_module._off_macos_wheel_tag() == getattr(hook_module, tag_attr)


def test_release_flag_on_an_unsupported_platform_raises(hook_module, monkeypatch):
    # Better a failed build than a wheel tagged manylinux_2_28_x86_64 with an
    # aarch64 binary in it — or with nothing in it at all.
    monkeypatch.setenv(hook_module.BUNDLE_ENV, "1")
    monkeypatch.setattr(hook_module.sys, "platform", "linux")
    monkeypatch.setattr(hook_module.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="no wheel target"):
        hook_module._off_macos_wheel_tag()


def _fake_build(tmp_path):
    """A subprocess.run stand-in that 'builds' whichever helper's script ran."""

    def fake(cmd, check):
        assert check
        script = Path(cmd[-1])
        assert script.stem == "build"  # build.sh, or build.ps1 on Windows
        name = {"stenocap-macos": "stenocap", "stenocap": "stenocap.exe"}.get(
            script.parent.name, "stenodiar"
        )
        if name == "stenodiar" and sys.platform == "win32":
            name += ".exe"  # cargo's suffix, mirroring _stenodiar_path
        binary = script.parent / name
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
    assert data["tag"] == hook_module.MACOS_TAG
    helper = tmp_path / "native" / "stenocap-macos" / "stenocap"
    stenodiar = _stenodiar(tmp_path)
    assert data["force_include"][str(helper)] == "stenograf/bin/stenocap"
    assert data["force_include"][str(stenodiar)] == f"stenograf/bin/{stenodiar.name}"
    if os.name == "posix":  # exec bits don't exist on Windows
        for binary in (helper, stenodiar):
            assert binary.stat().st_mode & 0o111 == 0o111


def test_bundles_only_stenodiar_on_linux(hook_module, tmp_path, monkeypatch):
    # Linux still captures through parec, so its release wheel carries the
    # diarization helper and nothing else.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    monkeypatch.setattr(hook_module.sys, "platform", "linux")
    monkeypatch.setattr(hook_module, "_off_macos_wheel_tag", lambda: hook_module.LINUX_TAG)
    monkeypatch.setattr(hook_module.subprocess, "run", _fake_build(tmp_path))
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)

    assert data["pure_python"] is False
    assert data["tag"] == hook_module.LINUX_TAG
    stenodiar = _stenodiar(tmp_path)
    assert data["force_include"] == {str(stenodiar): f"stenograf/bin/{stenodiar.name}"}


def test_windows_release_wheel_carries_the_capture_helper(hook_module, tmp_path, monkeypatch):
    # The one that must not regress: without stenocap.exe a Windows wheel
    # installs fine and cannot record a meeting.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    monkeypatch.setattr(hook_module.sys, "platform", "win32")
    monkeypatch.setattr(hook_module, "_off_macos_wheel_tag", lambda: hook_module.WINDOWS_TAG)
    monkeypatch.setattr(hook_module.subprocess, "run", _fake_build(tmp_path))
    data = build_data()
    make_hook(hook_module, tmp_path).initialize("standard", data)

    assert data["tag"] == hook_module.WINDOWS_TAG
    assert sorted(data["force_include"].values()) == [
        "stenograf/bin/stenocap.exe",
        "stenograf/bin/stenodiar.exe",
    ]


def test_a_windows_wheel_fails_rather_than_ship_without_capture(hook_module, tmp_path, monkeypatch):
    # stenocap is built first and its failure is fatal, so there is no path to
    # a platform-tagged Windows wheel carrying only the optional helper.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    monkeypatch.setattr(hook_module.sys, "platform", "win32")
    monkeypatch.setattr(hook_module, "_off_macos_wheel_tag", lambda: hook_module.WINDOWS_TAG)

    def fail(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hook_module.subprocess, "run", fail)
    data = build_data()
    with pytest.raises(RuntimeError, match="stenocap.exe"):
        make_hook(hook_module, tmp_path).initialize("standard", data)
    assert data["force_include"] == {}


def test_missing_toolchain_off_macos_fails_the_wheel(hook_module, tmp_path, monkeypatch):
    # Off macOS the flag *is* the request for a bundling wheel, so a failing
    # build must not degrade to a platform-tagged wheel with nothing in it —
    # unlike macOS, where a missing cargo is the ordinary source-install case.
    monkeypatch.setattr(hook_module, "_macos_arm64", lambda: False)
    monkeypatch.setattr(hook_module, "_off_macos_wheel_tag", lambda: hook_module.LINUX_TAG)
    monkeypatch.setattr(hook_module, "_cargo_available", lambda: False)

    def fail(cmd, check):
        raise subprocess.CalledProcessError(127, cmd)

    monkeypatch.setattr(hook_module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="install a Rust toolchain"):
        make_hook(hook_module, tmp_path).initialize("standard", build_data())


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
