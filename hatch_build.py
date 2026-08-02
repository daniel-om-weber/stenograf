"""Wheel build hook: bundle the compiled native helpers into platform wheels.

The helpers are gitignored build artifacts (native/stenocap-macos/stenocap,
native/stenocap/stenocap{,.exe}, native/stenodiar/stenodiar), so a plain wheel
would ship without them. Four wheels carry one or both and are tagged for their
platform; every other platform still gets the pure `py3-none-any` wheel, and
pip prefers the most specific compatible tag wherever one exists:

    macosx_14_0_arm64      stenocap (required) + stenodiar
    win_amd64              stenocap (required) + stenodiar
    manylinux_2_39_x86_64  stenocap (required) + stenodiar
    manylinux_2_28_x86_64  stenocap (required) only — the low-floor capture wheel

The two Linux wheels exist because their floors have different owners: 2.39 is
*stenodiar's* (the onnxruntime static lib needs glibc 2.38 symbols to link),
while stenocap links only libpulse and builds on the oldest maintained
manylinux image. A machine with glibc ≥ 2.39 resolves the full wheel (pip
ranks newer manylinux tags first); Ubuntu 22.04 / Debian 12 / RHEL 9 land on
the capture wheel and keep live capture, falling back to sherpa-only
diarization — instead of silently losing capture to the pure wheel.

macOS builds its helpers on any wheel build — stenocap is mandatory there and
compiled per machine anyway. Off macOS the bundling happens **only** when
STENOGRAF_BUNDLE_HELPERS is set (release.yml sets it: `1` for a platform's
full wheel, `capture` for the low-floor Linux wheel): a `pip install` from the
sdist must not start shelling out to cargo, and a wheel tagged
manylinux_2_28 that was actually linked against the local glibc would be a
lie. So off-mac the flag means "this is the release build" — and there a
missing toolchain is a hard error, because the artifact it would otherwise
produce is a platform-tagged wheel with nothing platform-specific in it.

The two helpers fail differently by design. Without stenocap `steno start`
cannot capture at all, so its build failing must fail the wheel — on every
platform that has one. stenodiar only upgrades *estimated* speaker counts
(stenograf falls back to sherpa without it), and building it needs a Rust
toolchain — so on macOS a machine without cargo gets a loud warning and a
stenodiar-less wheel, while the release workflow refuses to publish one
(release.yml verifies every wheel's payload), keeping PyPI wheels complete
without making every source install require Rust.

**A developer's checkout is not covered by any of this**, and does not need to
be: `uv sync` installs the project editable, which this hook skips outright.
Both capture helpers are found next to their sources by the provider's dev
fallback, so a checkout needs `native/stenocap/build.ps1` (or its macOS twin)
run once — and a checkout without cargo simply has no live capture on Windows,
the way it already has no diarization helper.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# macOS 14.4 is the runtime floor (Core Audio process taps); platform tags
# only carry major.minor, and 14.0 is the closest tag not above the floor.
MACOS_TAG = "py3-none-macosx_14_0_arm64"
# glibc 2.39 = the ubuntu-24.04 runner the release builds stenodiar on, and
# not a free choice: the onnxruntime static library ort-sys downloads needs
# glibc 2.38 symbols, so nothing older can even link it. The tag is a promise
# about the binary inside, so release.yml pins that runner and asserts the
# binary's own symbol floor against it.
LINUX_TAG = "py3-none-manylinux_2_39_x86_64"
# The low-floor capture wheel: stenocap links no ONNX, so it builds in the
# oldest manylinux image still maintained (AlmaLinux 8, glibc 2.28), which
# covers every distro stenodiar's floor shuts out. release.yml builds it in
# that container and asserts the symbol floor the tag promises.
LINUX_CAPTURE_TAG = "py3-none-manylinux_2_28_x86_64"
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


def _cargo_available() -> bool:
    return (
        shutil.which("cargo") is not None or (Path.home() / ".cargo" / "bin" / "cargo").is_file()
    )


def _off_macos_wheel_tag() -> str | None:
    """The platform tag for an off-mac helper wheel, if one was asked for."""
    mode = os.environ.get(BUNDLE_ENV)
    if not mode:
        return None
    machine = platform.machine().lower()
    linux_x86_64 = sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}
    if mode == "capture":
        if linux_x86_64:
            return LINUX_CAPTURE_TAG
        raise RuntimeError(
            f"{BUNDLE_ENV}=capture builds the low-floor Linux capture wheel and "
            f"exists only for linux-x86_64, not {sys.platform}/{platform.machine()}."
        )
    if linux_x86_64:
        return LINUX_TAG
    if sys.platform == "win32" and machine in {"x86_64", "amd64"}:
        return WINDOWS_TAG
    raise RuntimeError(
        f"{BUNDLE_ENV} is set, but there is no wheel target for "
        f"{sys.platform}/{platform.machine()} — only linux-x86_64 and win-amd64 "
        "(macOS bundles its helpers into its own wheel unconditionally)."
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

        hint = f"install a Rust toolchain (rustup); {BUNDLE_ENV} asks for a wheel "
        for binary in self._helpers_for(tag):
            self._build(binary, hint=hint + "that bundles the native helpers")
            build_data["force_include"][str(binary)] = f"stenograf/bin/{binary.name}"
        build_data["pure_python"] = False
        build_data["tag"] = tag

    def _helpers_for(self, tag: str) -> list[Path]:
        """Every helper the wheel with this tag must carry, stenocap first.

        Keyed off the tag rather than the host platform because the tag *is*
        the promise: the low-floor Linux wheel exists to carry capture onto
        machines stenodiar's glibc floor shuts out, so bundling stenodiar into
        it would drag the floor right back up.
        """
        native = Path(self.root) / "native"
        if tag == WINDOWS_TAG:
            return [native / "stenocap" / "stenocap.exe", native / "stenodiar" / "stenodiar.exe"]
        if tag == LINUX_CAPTURE_TAG:
            return [native / "stenocap" / "stenocap"]
        assert tag == LINUX_TAG, f"no helper list for wheel tag {tag}"
        return [native / "stenocap" / "stenocap", native / "stenodiar" / "stenodiar"]

    def _bundle_macos(self, build_data: dict) -> None:
        helper = Path(self.root) / "native" / "stenocap-macos" / "stenocap"
        self._build(helper, hint="install the Xcode Command Line Tools (xcode-select --install)")
        build_data["force_include"][str(helper)] = "stenograf/bin/stenocap"

        stenodiar = Path(self.root) / "native" / "stenodiar" / "stenodiar"
        if _cargo_available():
            self._build(
                stenodiar,
                hint="fix the Rust toolchain (cargo build failed) or uninstall it to skip",
            )
            build_data["force_include"][str(stenodiar)] = f"stenograf/bin/{stenodiar.name}"
        else:
            print(
                "hatch_build: no Rust toolchain — wheel will lack the stenodiar "
                "diarization helper (estimated speaker counts fall back to sherpa). "
                "Install rust (brew install rust) to bundle it.",
                file=sys.stderr,
            )

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
            # Never fall back to a degraded wheel on a build *failure*: for
            # stenocap a silently-pure wheel is exactly the broken artifact E1
            # exists to prevent, and for stenodiar a present-but-failing
            # toolchain is an environment error to fix, not an optional to
            # skip. (Source installs skip stenodiar before reaching here — on
            # macOS when cargo is absent, off-mac unless the release asked.)
            raise RuntimeError(f"building {binary.name} failed; {hint} and retry.") from exc
        if not binary.is_file():
            raise RuntimeError(f"{script} succeeded but {binary} was not produced")
        # hatchling propagates the on-disk mode into the wheel; make sure the
        # zip entry (and therefore the installed file) is executable.
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
