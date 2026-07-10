"""Wheel build hook: bundle the compiled native helpers on macOS-arm64.

The helpers are gitignored build artifacts (native/helper/stenocap,
native/stenodiar/stenodiar), so a plain wheel would ship without them. On
macOS-arm64 this hook compiles and packages them under stenograf/bin/ and tags
the wheel platform-specific; anywhere else it does nothing and the build stays
a pure `py3-none-any` wheel (PLAN.md Phase 4, Stage E1).

The two helpers fail differently by design. Without stenocap `steno start`
fails on every machine but a repo checkout, so its build failing must fail the
wheel. stenodiar only upgrades *estimated* speaker counts (stenograf falls
back to sherpa without it), and building it needs a Rust toolchain — so a
machine without cargo gets a loud warning and a stenodiar-less wheel, while
the release workflow refuses to publish one (release.yml verifies both
binaries), keeping PyPI wheels complete without making every source install
require Rust.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# macOS 14.4 is the runtime floor (Core Audio process taps); platform tags
# only carry major.minor, and 14.0 is the closest tag not above the floor.
WHEEL_TAG = "py3-none-macosx_14_0_arm64"


def _macos_arm64() -> bool:
    return sys.platform == "darwin" and os.uname().machine == "arm64"


def _cargo_available() -> bool:
    return (
        shutil.which("cargo") is not None or (Path.home() / ".cargo" / "bin" / "cargo").is_file()
    )


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version != "standard":
            return  # sdists and editable installs never carry the binaries
        if not _macos_arm64():
            return  # pure py3-none-any wheel everywhere else

        helper = Path(self.root) / "native" / "helper" / "stenocap"
        self._build(helper, required=True)
        build_data["force_include"][str(helper)] = "stenograf/bin/stenocap"

        stenodiar = Path(self.root) / "native" / "stenodiar" / "stenodiar"
        if _cargo_available():
            self._build(stenodiar, required=False)
            build_data["force_include"][str(stenodiar)] = "stenograf/bin/stenodiar"
        else:
            print(
                "hatch_build: no Rust toolchain — wheel will lack the stenodiar "
                "diarization helper (estimated speaker counts fall back to sherpa). "
                "Install rust (brew install rust) to bundle it.",
                file=sys.stderr,
            )

        build_data["pure_python"] = False
        build_data["tag"] = WHEEL_TAG

    def _build(self, binary: Path, *, required: bool) -> None:
        script = binary.parent / "build.sh"
        try:
            subprocess.run(["/bin/sh", str(script)], check=True)
        except subprocess.CalledProcessError as exc:
            # Never fall back to a degraded wheel on a build *failure*: for
            # stenocap a silently-pure wheel is exactly the broken artifact E1
            # exists to prevent, and for stenodiar a present-but-failing
            # toolchain is an environment error to fix, not an optional to
            # skip. (Source installs without any Rust toolchain skip stenodiar
            # before reaching here.)
            hint = (
                "install the Xcode Command Line Tools (xcode-select --install)"
                if required
                else "fix the Rust toolchain (cargo build failed) or uninstall it to skip"
            )
            raise RuntimeError(f"building {binary.name} failed; {hint} and retry.") from exc
        if not binary.is_file():
            raise RuntimeError(f"{script} succeeded but {binary} was not produced")
        # hatchling propagates the on-disk mode into the wheel; make sure the
        # zip entry (and therefore the installed file) is executable.
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
