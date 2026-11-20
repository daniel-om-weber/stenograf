"""Every program the library starts opens no window on Windows."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import stenograf
from stenograf import proc

SOURCE = Path(stenograf.__file__).parent

QUIET = {"quiet", "detached", "_child_group"}
"""The updater's helpers that return ``Popen`` arguments keeping a window from
opening; it must stay standard-library-only, so it has its own."""

NEVER_ON_WINDOWS = {
    ("doctor.py", "_codesign_valid"),
    ("notes/warm.py", "_available_bytes"),
    ("playback.py", "_play"),
}
"""Launches that only run on macOS or Linux: codesign, vm_stat, the ffmpeg
player (Windows plays through waveOut)."""


def _launches() -> list[tuple[str, str, ast.Call]]:
    found = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}
                ):
                    found.append((path.relative_to(SOURCE).as_posix(), function.name, node))
    return found


def _quiet(call: ast.Call) -> bool:
    for keyword in call.keywords:
        value = keyword.value
        if keyword.arg is None and isinstance(value, ast.Call):
            name = value.func.attr if isinstance(value.func, ast.Attribute) else None
            name = name or (value.func.id if isinstance(value.func, ast.Name) else None)
            if name in QUIET:
                return True
        if keyword.arg == "creationflags" and "no_window()" in ast.unparse(value):
            return True
    return False


def test_every_launch_keeps_its_window_closed():
    launches = _launches()
    assert len(launches) > 5, "the scan found no launches; has the source moved?"
    loud = [
        f"{file}:{call.lineno} in {function}()"
        for file, function, call in launches
        if (file, function) not in NEVER_ON_WINDOWS and not _quiet(call)
    ]
    assert not loud, "starts a program without creationflags=no_window(): " + ", ".join(loud)


def test_off_windows_nothing_is_added():
    if sys.platform != "win32":
        assert proc.no_window() == 0


def test_a_console_program_runs_with_it():
    done = subprocess.run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
        creationflags=proc.no_window(),
    )
    assert done.stdout.strip() == "ok"
