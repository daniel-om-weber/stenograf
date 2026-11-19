"""A stand-in for ``uv``, for the update tests.

Understands the three invocations the update code makes and nothing else:

- ``uv --version``
- ``uv tool run … python -m stenograf._updater prepare --out DIR`` — the trial
  run of a new release. Leaves the *real* ``_updater.py`` in ``DIR``, as the
  real one does.
- ``uv tool install … "stenograf>=X"`` — builds a pretend environment at
  ``$UV_TOOL_DIR/stenograf``: a receipt, and a ``bin/python`` that answers
  ``-m stenograf --version`` and ``-m stenograf._updater post-install``.

``FAKE_UV_MODE`` picks what goes wrong, one failure the real uv has been
measured to produce each (``eval/update_mechanics.sh``):

- ``ok`` (default)
- ``fail``: exit 1, nothing written — an empty cache, an unresolvable release.
- ``halfway``: a folder with files and **no receipt**, exit 1 — killed mid-install.
- ``silent-halfway``: the same folder, exit 0.
- ``older``: installs ``FAKE_UV_OLD`` instead of what was asked for.
- ``crashing``: installs fine; the pretend app then dies at start.
- ``prepare-fail``: the trial run exits 1 with a sentence for the user.
- ``prepare-slow``: the trial run never finishes (to be cancelled).

Every call is appended to ``$FAKE_UV_LOG`` as one JSON line: argv and the two
uv directories from the environment.
"""

import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

PYTHON = """#!/bin/sh
# The pretend environment's interpreter.
case "$*" in
  "-m stenograf --version") echo "stenograf, version {version}" ;;
  "-m stenograf._updater post-install")
    if [ -n "$FAKE_LAUNCHER" ]; then printf 'written by {version}' > "$FAKE_LAUNCHER"; fi ;;
  *) echo "unexpected: $*" >&2; exit 2 ;;
esac
"""


def build_environment(environment: Path, version: str, *, receipt: bool = True) -> None:
    (environment / "bin").mkdir(parents=True)
    python = environment / "bin" / "python"
    python.write_text(PYTHON.format(version=version), encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    (environment / "version.txt").write_text(version, encoding="utf-8")
    if receipt:
        (environment / "uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "stenograf" }]\n', encoding="utf-8"
        )


def main() -> int:
    argv = sys.argv[1:]
    mode = os.environ.get("FAKE_UV_MODE", "ok")
    log = os.environ.get("FAKE_UV_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as sink:
            record = {
                "argv": argv,
                "tool_dir": os.environ.get("UV_TOOL_DIR"),
                "bin_dir": os.environ.get("UV_TOOL_BIN_DIR"),
            }
            sink.write(json.dumps(record) + "\n")

    if argv == ["--version"]:
        print("uv 0.13.7 (fake)")
        return 0

    if argv[:2] == ["tool", "run"]:
        if mode == "prepare-slow":
            time.sleep(600)
        if mode == "prepare-fail":
            print("Resolved 30 packages")
            print("This release's window does not open here: no Qt")
            return 1
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        real = Path(__file__).parent.parent / "src" / "stenograf" / "_updater.py"
        shutil.copyfile(real, out / "updater.py")
        print("stenograf is ready to install")
        return 0

    if argv[:2] == ["tool", "install"]:
        environment = Path(os.environ["UV_TOOL_DIR"]) / "stenograf"
        wanted = argv[-1].split(">=")[-1]
        if mode == "fail":
            print("error: No solution found when resolving dependencies")
            return 1
        if mode in ("halfway", "silent-halfway"):
            build_environment(environment, wanted, receipt=False)
            return 0 if mode == "silent-halfway" else 1
        version = os.environ["FAKE_UV_OLD"] if mode == "older" else wanted
        build_environment(environment, version)
        if mode == "crashing":
            (environment / "crashes").write_text("", encoding="utf-8")
        print("Installed 2 executables: steno, stenograf")
        return 0

    print(f"fake uv: unexpected arguments {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
