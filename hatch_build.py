"""Wheel build hook: bundle the compiled `stenocap` capture helper on macOS-arm64.

The helper is a gitignored build artifact (native/helper/stenocap), so a plain
wheel would ship without it and `steno start` would fail on every machine but a
repo checkout. On macOS-arm64 this hook compiles it (swiftc + ad-hoc codesign,
via native/helper/build.sh), packages it at stenograf/bin/stenocap, and tags
the wheel platform-specific. Anywhere else it does nothing and the build stays
a pure `py3-none-any` wheel (PLAN.md Phase 4, Stage E1).
"""

from __future__ import annotations

import os
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


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version != "standard":
            return  # sdists and editable installs never carry the binary
        if not _macos_arm64():
            return  # pure py3-none-any wheel everywhere else

        helper = Path(self.root) / "native" / "helper" / "stenocap"
        self._build_helper(helper)

        build_data["pure_python"] = False
        build_data["tag"] = WHEEL_TAG
        build_data["force_include"][str(helper)] = "stenograf/bin/stenocap"

    def _build_helper(self, helper: Path) -> None:
        script = helper.parent / "build.sh"
        try:
            subprocess.run(["/bin/sh", str(script)], check=True)
        except subprocess.CalledProcessError as exc:
            # Never fall back to a helperless wheel here: on this platform a
            # silently-pure wheel is exactly the broken artifact E1 exists to
            # prevent. Building from source (uv tool install git+…) needs the
            # Swift toolchain; PyPI wheels are prebuilt so end users don't.
            raise RuntimeError(
                "building the stenocap capture helper failed. On macOS the "
                "wheel must bundle it; install the Xcode Command Line Tools "
                "(xcode-select --install) and retry."
            ) from exc
        if not helper.is_file():
            raise RuntimeError(f"{script} succeeded but {helper} was not produced")
        # hatchling propagates the on-disk mode into the wheel; make sure the
        # zip entry (and therefore the installed file) is executable.
        helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
