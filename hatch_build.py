"""Wheel build hook: bundle the compiled capture helper into platform wheels.

The helper is a gitignored build artifact (native/stenocap-macos/stenocap,
native/stenocap/stenocap{,.exe}), so a plain wheel would ship without it.
Three wheels carry it and are tagged for their platform; every other platform
still gets the pure `py3-none-any` wheel, and pip prefers the most specific
compatible tag wherever one exists:

    macosx_14_0_arm64      stenocap
    win_amd64              stenocap
    manylinux_2_28_x86_64  stenocap

The Linux floor is stenocap's own: it links only libpulse and builds in the
oldest maintained manylinux image, so Ubuntu 22.04 / Debian 12 / RHEL 9 keep
live capture instead of silently losing it to the pure wheel.

macOS builds its helper on any wheel build — stenocap is mandatory there and
compiled per machine anyway. Off macOS the bundling happens **only** when
STENOGRAF_BUNDLE_HELPERS is set (release.yml sets it): a `pip install` from
the sdist must not start shelling out to cargo, and a wheel tagged
manylinux_2_28 that was actually linked against the local glibc would be a
lie. So off-mac the flag means "this is the release build" — and there a
missing toolchain is a hard error, because the artifact it would otherwise
produce is a platform-tagged wheel with nothing platform-specific in it.
Without stenocap `steno start` cannot capture at all, so its build failing
fails the wheel on every platform.

**A developer's checkout is not covered by any of this**, and does not need to
be: `uv sync` installs the project editable, which this hook skips outright.
The capture helper is found next to its sources by the provider's dev
fallback, so a checkout needs `native/stenocap/build.ps1` (or its macOS twin)
run once — and a checkout without cargo simply has no live capture on Windows.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# macOS 14.4 is the runtime floor (Core Audio process taps); platform tags
# only carry major.minor, and 14.0 is the closest tag not above the floor.
MACOS_TAG = "py3-none-macosx_14_0_arm64"
# stenocap links no ONNX, so it builds in the oldest manylinux image still
# maintained (AlmaLinux 8, glibc 2.28). release.yml builds it in that
# container and asserts the symbol floor the tag promises.
LINUX_TAG = "py3-none-manylinux_2_28_x86_64"
WINDOWS_TAG = "py3-none-win_amd64"
# There is deliberately no win_arm64 tag. Moving capture into a native helper
# looked like it would strip capture from Windows-on-ARM, which installs the
# pure wheel — but that machine cannot install stenograf at all, and not
# because of us: sherpa-onnx, onnxruntime-directml, livekit and imageio-ffmpeg
# publish no win_arm64 wheels (checked 2026-08-02). A tagged wheel there would
# advertise support that resolves to nothing.

BUNDLE_ENV = "STENOGRAF_BUNDLE_HELPERS"


def _macos_arm64() -> bool:
    return sys.platform == "darwin" and os.uname().machine == "arm64"


def _off_macos_wheel_tag() -> str | None:
    """The platform tag for an off-mac helper wheel, if one was asked for."""
    if not os.environ.get(BUNDLE_ENV):
        return None
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return LINUX_TAG
    if sys.platform == "win32" and machine in {"x86_64", "amd64"}:
        return WINDOWS_TAG
    raise RuntimeError(
        f"{BUNDLE_ENV} is set, but there is no wheel target for "
        f"{sys.platform}/{platform.machine()} — only linux-x86_64 and win-amd64 "
        "(macOS bundles its helper into its own wheel unconditionally)."
    )


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version != "standard":
            return  # sdists and editable installs never carry the binaries

        if _macos_arm64():
            self._bundle_macos(build_data)
            return

        tag = _off_macos_wheel_tag()
        if tag is None:
            return  # pure py3-none-any wheel — source installs need no Rust

        native = Path(self.root) / "native" / "stenocap"
        helper = native / ("stenocap.exe" if tag == WINDOWS_TAG else "stenocap")
        self._build(
            helper,
            hint=f"install a Rust toolchain (rustup); {BUNDLE_ENV} asks for a wheel "
            "that bundles the capture helper",
        )
        build_data["force_include"][str(helper)] = f"stenograf/bin/{helper.name}"
        build_data["pure_python"] = False
        build_data["tag"] = tag

    def _bundle_macos(self, build_data: dict) -> None:
        helper = Path(self.root) / "native" / "stenocap-macos" / "stenocap"
        self._build(helper, hint="install the Xcode Command Line Tools (xcode-select --install)")
        build_data["force_include"][str(helper)] = "stenograf/bin/stenocap"
        build_data["pure_python"] = False
        build_data["tag"] = MACOS_TAG

    def _build(self, binary: Path, *, hint: str) -> None:
        # Windows has no shell to run build.sh; its twin is a PowerShell script.
        if sys.platform == "win32":
            script = binary.parent / "build.ps1"
            command = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ]
        else:
            script = binary.parent / "build.sh"
            command = ["/bin/sh", str(script)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            # Never fall back to a degraded wheel on a build failure: a
            # platform-tagged wheel without stenocap installs fine and cannot
            # record a meeting.
            raise RuntimeError(f"building {binary.name} failed; {hint} and retry.") from exc
        if not binary.is_file():
            raise RuntimeError(f"{script} succeeded but {binary} was not produced")
        # hatchling propagates the on-disk mode into the wheel; make sure the
        # zip entry (and therefore the installed file) is executable.
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
