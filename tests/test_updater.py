"""The updater script: the swap, every way it fails, and the frozen contract.

``apply`` is driven against ``fake_uv.py``, which builds pretend environments
and can fail in each way the real uv has been measured to. What is asserted is
always the same two things: which version is at the environment's path
afterwards, and what the result file tells the user.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import TOOL_TIMEOUT_S
from fake_uv import build_environment

from stenograf import _updater
from stenograf._updater import Request

# The pretend environment's interpreter is a shell script.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")

FAKE_UV = Path(__file__).parent / "fake_uv.py"

APP = """
import fcntl, json, pathlib, sys, time
environment, lock, marker, linger = sys.argv[1:5]
handle = open(lock, "a+b")
fcntl.flock(handle, fcntl.LOCK_SH)
version = (pathlib.Path(environment) / "version.txt").read_text()
with open(pathlib.Path(marker).parent / "starts.txt", "a") as starts:
    starts.write(version + "\\n")
if (pathlib.Path(environment) / "crashes").exists():
    sys.exit(1)
pathlib.Path(marker).write_text(json.dumps({"version": version}))
time.sleep(float(linger))
"""
"""A pretend app: holds the shared lock, says which version it is, and either
reports its window built or — in an environment marked ``crashes`` — dies."""


LAUNCHER = "import subprocess, sys; subprocess.Popen(sys.argv[1:])"
"""What `open` on macOS and a shell link on Windows are to the updater: a
command that starts the app, exits 0 at once, and says nothing about it."""


@pytest.fixture(autouse=True)
def _short_waits(monkeypatch):
    monkeypatch.setattr(_updater, "EXIT_WAIT_S", 1.0)
    monkeypatch.setattr(_updater, "LOCK_WAIT_S", 1.0)
    monkeypatch.setattr(_updater, "RENAME_WAIT_S", 1.0)
    monkeypatch.setattr(_updater, "START_WAIT_S", 15.0)
    monkeypatch.setattr(_updater, "NEVER_STARTED_S", 4.0)
    monkeypatch.setattr(_updater, "QUIET_S", 1.5)


class Machine:
    """A pretend install of 0.7.0 with everything ``apply`` touches around it."""

    def __init__(self, root: Path, monkeypatch) -> None:
        self.root = root
        self.tool_dir = root / "uv" / "tools"
        self.environment = self.tool_dir / "stenograf"
        self.bin_dir = root / "bin"
        self.work = root / "work"
        self.lock = self.work / "in-use.lock"
        self.log = root / "uv-calls.jsonl"
        self.launcher = root / "launcher.txt"
        self.bin_dir.mkdir()
        self.work.mkdir()
        build_environment(self.environment, "0.7.0")
        self.launcher.write_text("written by 0.7.0", encoding="utf-8")
        self.uv = root / "uv-wrapper"
        wrapper = f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_UV}" "$@"\n'
        self.uv.write_text(wrapper, encoding="utf-8")
        self.uv.chmod(0o755)
        self.app = root / "app.py"
        self.app.write_text(APP, encoding="utf-8")
        monkeypatch.setenv("FAKE_UV_LOG", str(self.log))
        monkeypatch.setenv("FAKE_UV_OLD", "0.7.0")
        monkeypatch.setenv("FAKE_LAUNCHER", str(self.launcher))
        self._monkeypatch = monkeypatch

    def mode(self, mode: str) -> None:
        self._monkeypatch.setenv("FAKE_UV_MODE", mode)

    def request(
        self,
        *,
        restart: bool = False,
        linger: float = 2.0,
        marker: Path | None = None,
        through_launcher: bool = False,
        **changes,
    ) -> Path:
        marker = marker or _updater.marker_path(self.work)
        command = (
            (sys.executable, str(self.app), str(self.environment), str(self.lock),
             str(marker), str(linger))
            if restart
            else ()
        )  # fmt: skip
        if through_launcher:
            command = (sys.executable, "-c", LAUNCHER, *command)
        fields = {
            "target_version": "0.8.0",
            "python": "3.13",
            "environment": self.environment,
            "tool_dir": self.tool_dir,
            "bin_dir": self.bin_dir,
            "uv": str(self.uv),
            "work_dir": self.work,
            "lock_file": self.lock,
            "restart": command,
            "launcher_files": (self.launcher,),
            "direct": restart and not through_launcher,
        }
        fields.update(changes)
        path = self.work / "request.json"
        Request(**fields).write(path)
        return path

    @property
    def version(self) -> str:
        return (self.environment / "version.txt").read_text(encoding="utf-8")

    @property
    def result(self) -> dict:
        return json.loads(_updater.result_path(self.work).read_text(encoding="utf-8"))

    @property
    def starts(self) -> list[str]:
        return (self.work / "starts.txt").read_text(encoding="utf-8").split()

    def wait_for_starts(self, count: int) -> list[str]:
        for _ in range(200):
            if (self.work / "starts.txt").exists() and len(self.starts) >= count:
                break
            time.sleep(0.05)
        return self.starts

    def leftovers(self) -> list[str]:
        return sorted(
            path.name
            for path in self.tool_dir.parent.iterdir()
            if path.name != self.tool_dir.name
        )

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def machine(tmp_path, monkeypatch):
    return Machine(tmp_path, monkeypatch)


# -- the swap ---------------------------------------------------------------


def test_a_good_update_replaces_the_environment_at_the_same_path(machine):
    assert _updater.apply(machine.request()) == 0
    assert machine.version == "0.8.0"
    assert machine.result == {
        "format": 1,
        "ok": True,
        "version": "0.8.0",
        "message": "Updated to 0.8.0.",
        "log": str(machine.work / "updater.log"),
        "rolled_back": False,
    }
    # Nothing of the old version, the journal or the launcher copies remains.
    assert machine.leftovers() == []
    assert not _updater.journal_path(machine.work).exists()
    assert not (machine.work / "launchers").exists()


def test_the_install_is_offline_at_least_the_target_and_told_where_it_lives(machine):
    _updater.apply(machine.request())
    (call,) = machine.calls()
    assert call["argv"] == [
        "tool", "install", "--offline", "--force", "--python", "3.13", "stenograf>=0.8.0",
    ]  # fmt: skip
    # uv finds its directories through variables the app may not have had.
    assert call["tool_dir"] == str(machine.tool_dir)
    assert call["bin_dir"] == str(machine.bin_dir)


def test_the_new_release_rewrites_its_launchers(machine):
    _updater.apply(machine.request())
    assert machine.launcher.read_text(encoding="utf-8") == "written by 0.8.0"


@pytest.mark.parametrize("mode", ["fail", "halfway", "silent-halfway", "older"])
def test_a_failed_install_puts_the_old_version_back(machine, mode):
    """Including the folder without a receipt that a killed install leaves:
    it is the updater's own doing, and must not block its own rollback."""
    machine.mode(mode)
    assert _updater.apply(machine.request()) == 1
    assert machine.version == "0.7.0"
    assert _updater.names_stenograf(machine.environment)
    assert machine.result["ok"] is False
    assert machine.result["message"].endswith("The previous version was restored.")
    assert machine.leftovers() == []
    assert machine.launcher.read_text(encoding="utf-8") == "written by 0.7.0"


def test_a_wrong_version_is_named(machine):
    machine.mode("older")
    _updater.apply(machine.request())
    assert "Version 0.7.0 was installed instead of 0.8.0" in machine.result["message"]


def test_a_failed_update_starts_the_old_app_again(machine):
    machine.mode("fail")
    _updater.apply(machine.request(restart=True, linger=0.5))
    assert machine.wait_for_starts(1) == ["0.7.0"]


# -- refusing to start ------------------------------------------------------


def test_a_folder_that_is_not_a_stenograf_install_is_left_alone(machine):
    (machine.environment / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "ruff" }]\n', encoding="utf-8"
    )
    assert _updater.apply(machine.request()) == 1
    assert machine.version == "0.7.0"
    assert "left alone" in machine.result["message"]
    assert not machine.log.exists()  # uv was never run


def test_an_update_does_not_start_while_stenograf_runs_elsewhere(machine):
    holder = subprocess.Popen(
        [sys.executable, str(machine.app), str(machine.environment), str(machine.lock),
         str(machine.root / "other-marker.json"), "20"],
    )  # fmt: skip
    try:
        for _ in range(200):
            if (machine.root / "other-marker.json").exists():
                break
            time.sleep(0.05)
        assert _updater.apply(machine.request()) == 1
    finally:
        holder.kill()
        holder.wait(timeout=TOOL_TIMEOUT_S)
    assert machine.version == "0.7.0"
    assert "in use elsewhere" in machine.result["message"]
    assert not machine.log.exists()


def test_an_update_does_not_start_while_the_app_is_still_alive(machine):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        assert _updater.apply(machine.request(wait_pids=(sleeper.pid,))) == 1
    finally:
        sleeper.kill()
        sleeper.wait(timeout=TOOL_TIMEOUT_S)
    assert machine.version == "0.7.0"
    assert "did not close" in machine.result["message"]


def test_a_second_updater_leaves_the_first_one_to_it(machine):
    with _updater.acquire(_updater.alive_path(machine.work), exclusive=True):
        assert _updater.apply(machine.request()) == 1
    assert machine.version == "0.7.0"
    # No result file: the first updater's is the one the app will read.
    assert not _updater.result_path(machine.work).exists()
    assert not machine.log.exists()


def test_a_crash_in_the_middle_of_the_swap_puts_the_old_version_back(machine, monkeypatch):
    def explode(self):
        raise RuntimeError("out of thin air")

    lock_during_rollback = []
    roll_back = _updater._Swap.roll_back

    def watched(self):
        # Nobody may get in while the environment is being moved back: an app
        # started now would run from a folder about to be renamed away.
        try:
            _updater.acquire(machine.lock, exclusive=False).close()
            lock_during_rollback.append("free")
        except _updater.LockHeld:
            lock_during_rollback.append("held")
        roll_back(self)

    monkeypatch.setattr(_updater._Swap, "refresh_launchers", explode)
    monkeypatch.setattr(_updater._Swap, "roll_back", watched)
    assert _updater.apply(machine.request(restart=True, linger=0.5)) == 1
    assert lock_during_rollback == ["held"]
    assert machine.version == "0.7.0"
    assert machine.leftovers() == []
    assert "stopped unexpectedly" in machine.result["message"]
    assert "previous version was restored" in machine.result["message"]
    assert machine.result["rolled_back"] is True
    assert machine.wait_for_starts(1) == ["0.7.0"]


def test_a_crash_before_anything_moved_still_starts_the_app_again(machine, monkeypatch):
    def explode(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_updater._Swap, "copy_launchers", explode)
    assert _updater.apply(machine.request(restart=True, linger=0.5)) == 1
    assert machine.version == "0.7.0"
    assert machine.result["rolled_back"] is False
    assert machine.wait_for_starts(1) == ["0.7.0"]


def test_a_result_that_cannot_be_written_does_not_cost_the_restart(machine, monkeypatch):
    machine.mode("fail")
    request = machine.request(restart=True, linger=0.5)
    _updater.result_path(machine.work).mkdir()  # a file cannot be written here
    assert _updater.apply(request) == 1
    assert machine.wait_for_starts(1) == ["0.7.0"]


def test_only_a_version_that_was_installed_and_taken_out_counts_as_rolled_back(machine):
    machine.mode("older")
    _updater.apply(machine.request())
    assert machine.result["rolled_back"] is True
    (machine.environment / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "ruff" }]\n', encoding="utf-8"
    )
    _updater.apply(machine.request())
    assert machine.result["rolled_back"] is False


def test_a_delete_that_was_interrupted_can_be_finished(tmp_path, monkeypatch):
    """The receipt is what marks a folder as ours, and a plain recursive delete
    removes it first on some file systems — leaving a folder nobody dares touch,
    which then blocks every later update."""
    ours = tmp_path / "ours"
    build_environment(ours, "0.7.0")
    (ours / "lib").mkdir()
    (ours / "lib" / "big.bin").write_text("x", encoding="utf-8")
    real = shutil.rmtree
    calls = []

    def interrupted(path, **kwargs):
        calls.append(Path(path).name)
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "rmtree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _updater.discard(ours)
    assert calls and "ours" not in calls  # it was inside, not at the folder itself
    monkeypatch.setattr(shutil, "rmtree", real)
    assert ours.is_dir()
    _updater.discard(ours)
    assert not ours.exists()


def test_a_stale_parked_version_is_cleared_first(machine):
    build_environment(_updater.parked_path(machine.tool_dir), "0.6.0")
    assert _updater.apply(machine.request()) == 0
    assert machine.version == "0.8.0"
    assert machine.leftovers() == []


def test_a_foreign_folder_where_the_old_version_would_park_stops_the_update(machine):
    foreign = _updater.parked_path(machine.tool_dir)
    foreign.mkdir()
    (foreign / "thesis.tex").write_text("precious", encoding="utf-8")
    assert _updater.apply(machine.request()) == 1
    assert machine.version == "0.7.0"
    assert (foreign / "thesis.tex").read_text(encoding="utf-8") == "precious"


def test_discard_only_removes_what_is_recognizably_ours(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "file").write_text("x", encoding="utf-8")
    _updater.discard(foreign)
    assert foreign.is_dir()

    ours = tmp_path / "ours"
    build_environment(ours, "0.7.0")
    _updater.discard(ours)
    assert not ours.exists()


# -- after the restart ------------------------------------------------------


def test_the_old_version_goes_once_the_new_app_reports_its_window(machine):
    assert _updater.apply(machine.request(restart=True)) == 0
    assert machine.starts == ["0.8.0"]
    assert machine.version == "0.8.0"
    assert machine.leftovers() == []


def test_a_new_app_that_dies_at_start_is_rolled_back(machine):
    machine.mode("crashing")
    assert _updater.apply(machine.request(restart=True, linger=0.5)) == 1
    assert machine.version == "0.7.0"
    assert "Version 0.8.0 did not start" in machine.result["message"]
    assert machine.launcher.read_text(encoding="utf-8") == "written by 0.7.0"
    # The new one, then the old one again.
    assert machine.wait_for_starts(2) == ["0.8.0", "0.7.0"]


def test_a_terminal_command_during_the_start_is_not_taken_for_the_app(machine, monkeypatch):
    """A `steno` command holds the same lock for a moment. Its exit must not
    read as the app's while the app is still on its way up."""
    monkeypatch.setattr(_updater, "QUIET_S", 4.0)
    # A launcher under which a terminal command comes and goes (a second with
    # the lock) before the app itself, two seconds in, takes the lock.
    slow_start = (
        "import subprocess, sys, time\n"
        "count = int(sys.argv[1])\n"
        "subprocess.Popen(sys.argv[2:2 + count])\n"
        "time.sleep(2)\n"
        "subprocess.Popen(sys.argv[2 + count:])\n"
    )
    terminal = [
        sys.executable, str(machine.app), str(machine.environment), str(machine.lock),
        str(machine.root / "terminal-marker.json"), "1",
    ]  # fmt: skip
    request = machine.request(restart=True, linger=1.0)
    data = json.loads(request.read_text(encoding="utf-8"))
    data["restart"] = [
        sys.executable, "-c", slow_start, str(len(terminal)), *terminal, *data["restart"],
    ]  # fmt: skip
    data["direct"] = False
    request.write_text(json.dumps(data), encoding="utf-8")
    assert _updater.apply(request) == 0
    assert machine.version == "0.8.0"
    assert machine.result["ok"] is True
    assert machine.leftovers() == []


def test_a_launcher_s_exit_code_says_nothing_about_the_app(machine):
    """Windows' explorer.exe returns 1 when all went well."""
    rude = "import subprocess, sys; subprocess.Popen(sys.argv[1:]); sys.exit(1)"
    request = machine.request(restart=True, linger=1.0)
    data = json.loads(request.read_text(encoding="utf-8"))
    data["restart"] = [sys.executable, "-c", rude, *data["restart"]]
    data["direct"] = False
    request.write_text(json.dumps(data), encoding="utf-8")
    assert _updater.apply(request) == 0
    assert machine.version == "0.8.0"


@pytest.mark.parametrize("mode", ["ok", "crashing"])
def test_an_app_started_through_a_launcher_is_judged_by_the_lock(machine, mode):
    machine.mode(mode)
    done = _updater.apply(machine.request(restart=True, linger=1.5, through_launcher=True))
    if mode == "ok":
        assert done == 0 and machine.version == "0.8.0"
        assert machine.leftovers() == []
    else:
        assert done == 1 and machine.version == "0.7.0"
        assert "Version 0.8.0 did not start" in machine.result["message"]
        assert machine.wait_for_starts(2) == ["0.8.0", "0.7.0"]


def test_a_new_app_that_never_reports_keeps_the_old_version_parked(machine, monkeypatch):
    """Running but silent is not dead: nothing is rolled back on a guess, and
    nothing is deleted either. The next start that reaches its window cleans up."""
    monkeypatch.setattr(_updater, "START_WAIT_S", 3.0)
    # The app reports to a file the updater does not read.
    request = machine.request(restart=True, linger=6, marker=machine.work / "elsewhere.json")
    assert _updater.apply(request) == 0
    assert machine.version == "0.8.0"
    assert machine.leftovers() == [_updater.parked_path(machine.tool_dir).name]
    assert _updater.journal_path(machine.work).exists()


# -- the script as the app runs it -------------------------------------------


def test_the_copied_script_runs_on_a_bare_interpreter(machine, tmp_path):
    """``python -I updater.py``: no site-packages, no stenograf to import."""
    script = tmp_path / "prepared" / "updater.py"
    script.parent.mkdir()
    script.write_bytes(Path(_updater.__file__).read_bytes())
    done = subprocess.run(
        [getattr(sys, "_base_executable", sys.executable), "-I", str(script), "apply",
         str(machine.request())],
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_S,
        env={**os.environ},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert machine.version == "0.8.0"


# -- the contract -----------------------------------------------------------


def test_the_contract_with_older_releases_has_not_changed():
    """Every one of these is something a released app already relies on.

    A failure here is the point: changing any of them strands the installs
    that cannot be told about the change."""
    assert Path(_updater.__file__).name == "_updater.py"
    assert _updater.REQUEST_FORMAT == 1
    assert {field.name for field in dataclasses.fields(Request)} == {
        "target_version", "python", "environment", "tool_dir", "bin_dir", "uv", "work_dir",
        "lock_file", "wait_pids", "restart", "launcher_files", "direct", "format",
    }  # fmt: skip
    assert _updater.result_path(Path("w")) == Path("w") / "result.json"


def test_a_request_from_the_first_release_still_reads(tmp_path):
    """The optional fields absent, and a field this release has never heard of."""
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "format": 1,
                "target_version": "0.8.0",
                "python": "3.13",
                "environment": "/t/stenograf",
                "tool_dir": "/t",
                "bin_dir": "/b",
                "uv": "/usr/bin/uv",
                "work_dir": "/w",
                "lock_file": "/w/in-use.lock",
                "invented_later": True,
            }
        ),
        encoding="utf-8",
    )
    request = Request.read(path)
    assert request.environment == Path("/t/stenograf")
    assert request.wait_pids == () and request.restart == () and request.launcher_files == ()


def test_the_two_commands_parse(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(_updater, "prepare", lambda out: seen.setdefault("prepare", out) and 0)
    monkeypatch.setattr(_updater, "apply", lambda request: seen.setdefault("apply", request) and 0)
    monkeypatch.chdir(tmp_path)
    _updater.main(["prepare", "--out", "there"])
    _updater.main(["apply", str(tmp_path / "request.json")])
    assert seen == {"prepare": Path("there"), "apply": tmp_path / "request.json"}
