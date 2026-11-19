"""The update library: which release is offered, to which install, and how.

No test here reaches the network (``conftest`` makes ``update.fetch`` raise)
and none touches a real install — the swap itself is ``test_updater.py``.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import TOOL_TIMEOUT_S, write_settings

from stenograf import _updater, cli, update
from stenograf.cli import main

FAKE_UV = Path(__file__).parent / "fake_uv.py"
REAL_ROOT = update.root
"""Taken at import: every test runs with ``root`` pointed into its tmp_path."""
NOW = 1_800_000_000.0
LONG_AGO = "2026-01-01T00:00:00.000000Z"


def release(*, yanked=False, uploaded=LONG_AGO, requires=">=3.12,<3.14"):
    return [
        {
            "filename": name,
            "yanked": yanked,
            "upload_time_iso_8601": uploaded,
            "requires_python": requires,
        }
        for name in ("any.whl", "macosx.whl")
    ]


def write_receipt(prefix: Path, tool: str) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "uv-receipt.toml").write_text(tool, encoding="utf-8")
    return prefix


ENTRYPOINTS = 'entrypoints = [{ name = "steno", install-path = "/home/u/.local/bin/steno" }]\n'


@pytest.fixture
def standard(monkeypatch, tmp_path):
    """A standard install of 0.7.0 on a system where the swap is measured."""
    prefix = write_receipt(
        tmp_path / "tools" / "stenograf",
        '[tool]\nrequirements = [{ name = "stenograf" }]\n' + ENTRYPOINTS,
    )
    monkeypatch.setattr(update, "__version__", "0.7.0")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(update, "find_uv", lambda: "/usr/bin/uv")
    return prefix


def found(version="0.8.0", **state):
    update.save_state({"checked_at": time.time(), "latest": version, **state})


# -- versions ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.7.1", (0, 7, 1)),
        ("0.7.0", (0, 7)),  # the same release as "0.7"
        ("1.10", (1, 10)),
        ("1.0.0", (1,)),
        ("0.8.0rc1", None),
        ("0.8.0.dev3", None),
        ("", None),
    ],
)
def test_only_final_releases_parse(text, expected):
    assert update.parse_version(text) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (">=3.12,<3.14", True),
        ("<3.14,>=3.12", True),
        (">=3.14", False),
        ("<3.13", False),
        (">=3.12", True),
        (None, True),
        ("", True),
        ("==3.13.*", True),
        ("!=3.13.*", False),
        ("~=3.12", True),
        ("~=3.13.0", True),
        (">=3.12; python_version < '4'", None),  # not a form this project writes
    ],
)
def test_requires_python_against_313(spec, expected):
    assert update.python_allowed(spec, (3, 13, 5)) is expected


# -- which release is offered ------------------------------------------------


def test_the_newest_final_release_is_offered():
    payload = {"releases": {"0.7.0": release(), "0.8.0": release(), "0.7.5": release()}}
    assert update.newest_release(payload, installed=(0, 7, 0), now=NOW) == update.Release(
        "0.8.0", ">=3.12,<3.14"
    )


def test_nothing_is_offered_when_this_is_the_newest():
    payload = {"releases": {"0.6.0": release(), "0.7.0": release(), "0.7": release()}}
    installed = update.parse_version("0.7.0")
    assert installed is not None
    assert update.newest_release(payload, installed=installed, now=NOW) is None


def test_prereleases_and_withdrawn_releases_are_passed_over():
    payload = {
        "releases": {
            "0.9.0rc1": release(),
            "0.8.1": release(yanked=True),
            "0.8.0": release(),
            "0.9.0": [],
        }
    }
    found_release = update.newest_release(payload, installed=(0, 7, 0), now=NOW)
    assert found_release is not None and found_release.version == "0.8.0"


def test_a_release_still_uploading_is_not_offered_yet():
    """Its generic package installs everywhere and cannot record a meeting."""
    just_now = "2027-01-15T07:59:00.000000Z"  # NOW is 08:00 that day
    fresh = release()
    fresh[1]["upload_time_iso_8601"] = just_now
    payload = {"releases": {"0.8.0": fresh}}
    assert update.newest_release(payload, installed=(0, 7, 0), now=NOW) is None
    assert update.newest_release(payload, installed=(0, 7, 0), now=NOW + 3600) is not None


def test_a_malformed_answer_offers_nothing():
    assert update.newest_release({"releases": "nope"}, installed=(0, 7, 0), now=NOW) is None
    assert update.newest_release({}, installed=(0, 7, 0), now=NOW) is None


# -- which install may take it -----------------------------------------------


@pytest.mark.parametrize(
    ("tool", "kind"),
    [
        ('[tool]\nrequirements = [{ name = "stenograf" }]\n', "standard"),
        ('[tool]\nrequirements = [{ name = "stenograf" }]\npython = "3.13"\n', "standard"),
        # What the updater itself leaves behind, and it must stay updatable.
        ('[tool]\nrequirements = [{ name = "stenograf", specifier = ">=0.8.0" }]\n', "standard"),
        ('[tool]\nrequirements = [{ name = "stenograf", extras = ["gui"] }]\n', "standard"),
        ('[tool]\nrequirements = [{ name = "stenograf", specifier = "==0.6.0" }]\n', "pinned"),
        ('[tool]\nrequirements = [{ name = "stenograf", specifier = "<0.8" }]\n', "pinned"),
        ('[tool]\nrequirements = [{ name = "stenograf", extras = ["cuda"] }]\n', "pinned"),
        (
            '[tool]\nrequirements = [{ name = "stenograf" }]\n'
            'constraints = [{ name = "numpy", specifier = "==2.4.6" }]\n',
            "pinned",
        ),
        (
            '[tool]\nrequirements = [{ name = "stenograf" }, { name = "rich" }]\n',
            "pinned",
        ),
        (
            '[tool]\nrequirements = [{ name = "stenograf", directory = "/home/u/stenograf" }]\n',
            "source",
        ),
        ('[tool]\nrequirements = [{ name = "stenograf", git = "https://x/y" }]\n', "source"),
    ],
)
def test_installs_are_told_apart_by_their_receipt(tmp_path, tool, kind):
    """Each receipt shape is one uv was measured to write (eval/update_mechanics.sh)."""
    prefix = write_receipt(tmp_path / "stenograf", tool + ENTRYPOINTS)
    assert update.describe_install(prefix).kind == kind


def test_exclude_newer_in_the_receipt_is_a_pin(tmp_path):
    prefix = write_receipt(
        tmp_path / "stenograf",
        '[tool]\nrequirements = [{ name = "stenograf" }]\n'
        + ENTRYPOINTS
        + '\n[tool.options]\nexclude-newer = "2026-08-10T00:00:00Z"\n',
    )
    assert update.describe_install(prefix).kind == "pinned"


def test_a_standard_install_knows_where_uv_put_its_commands(tmp_path):
    prefix = write_receipt(
        tmp_path / "stenograf", '[tool]\nrequirements = [{ name = "stenograf" }]\n' + ENTRYPOINTS
    )
    install = update.describe_install(prefix)
    assert install.environment == prefix
    assert install.bin_dir == Path("/home/u/.local/bin")


def test_a_receipt_without_entry_points_cannot_update_itself(tmp_path):
    prefix = write_receipt(
        tmp_path / "stenograf", '[tool]\nrequirements = [{ name = "stenograf" }]\n'
    )
    assert update.describe_install(prefix).kind == "pinned"


def test_this_checkout_is_a_source_install():
    assert update.describe_install().kind == "source"
    assert not update.enabled()


# -- the check --------------------------------------------------------------


def test_a_check_stores_what_it_found(standard, monkeypatch):
    monkeypatch.setattr(update, "fetch", lambda: {"releases": {"0.8.0": release()}})
    update.check()
    state = update.load_state()
    assert state["latest"] == "0.8.0"
    assert state["requires_python"] == ">=3.12,<3.14"


def test_a_second_check_the_same_day_asks_nobody(standard, monkeypatch):
    asked = []
    monkeypatch.setattr(update, "fetch", lambda: asked.append(1) or {"releases": {}})
    update.check()
    update.check()
    assert len(asked) == 1
    update.check(force=True)
    assert len(asked) == 2


def test_a_clock_that_was_once_ahead_does_not_silence_the_check(standard, monkeypatch):
    asked = []
    monkeypatch.setattr(update, "fetch", lambda: asked.append(1) or {"releases": {}})
    update.save_state({"checked_at": time.time() + 10 * 86400})
    update.check()
    assert asked == [1]


def test_a_check_that_cannot_reach_pypi_is_silent_and_tries_again(standard):
    update.check()  # conftest's fetch raises
    assert "checked_at" not in update.load_state()


def test_the_switch_stops_the_request_itself(standard, monkeypatch):
    write_settings("[updates]\ncheck = false\n")
    monkeypatch.setattr(update, "fetch", lambda: pytest.fail("asked PyPI with the check off"))
    update.check(force=True)
    found()
    assert update.offer() is None
    assert update.notice() is None


def test_unreadable_settings_do_not_default_to_asking(standard, monkeypatch):
    write_settings("[updates]\ncheck = 3\n")
    monkeypatch.setattr(update, "fetch", lambda: pytest.fail("asked PyPI"))
    update.check(force=True)


# -- the offer --------------------------------------------------------------


def test_a_standard_install_is_offered_the_button(standard):
    found()
    offer = update.offer()
    assert offer == update.Offer("0.8.0", True, update.UPGRADE_COMMAND, "")
    assert offer.url.endswith("/releases/tag/v0.8.0")


def test_nothing_is_offered_without_a_newer_release(standard):
    found("0.7.0")
    assert update.offer() is None


def test_a_version_that_failed_here_is_not_offered_again_unasked(standard):
    found(failed=["0.8.0"])
    assert update.offer() is None
    assert update.offer(include_failed=True) is not None
    found("0.8.1", failed=["0.8.0"])
    assert update.offer() is not None  # the next release is


def test_an_unmeasured_system_gets_the_notice_not_the_button(standard, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    found()
    offer = update.offer()
    assert offer is not None and not offer.installable
    assert offer.command == update.UPGRADE_COMMAND
    monkeypatch.setenv(update.UNMEASURED_OVERRIDE, "1")
    offer = update.offer()
    assert offer is not None and offer.installable


def test_a_release_that_drops_this_python_is_a_notice(standard):
    found(requires_python=">=3.99")
    offer = update.offer()
    assert offer is not None and not offer.installable
    assert "newer Python" in offer.reason
    found(requires_python=">=3.12; python_version < '4'")
    offer = update.offer()
    assert offer is not None and not offer.installable
    assert "could not be read" in offer.reason


def test_without_uv_it_is_a_notice(standard, monkeypatch):
    monkeypatch.setattr(update, "find_uv", lambda: None)
    found()
    offer = update.offer()
    assert offer is not None and not offer.installable


def test_a_pinned_install_is_told_and_given_no_command(standard):
    write_receipt(
        standard,
        '[tool]\nrequirements = [{ name = "stenograf", specifier = "==0.7.0" }]\n' + ENTRYPOINTS,
    )
    found()
    offer = update.offer()
    assert offer is not None and not offer.installable and offer.command is None
    assert "pinned" in offer.reason


# -- the terminal notice ----------------------------------------------------


def test_the_notice_names_the_version_the_command_and_the_caveat(standard):
    found()
    line = update.notice()
    assert line is not None
    assert "stenograf 0.8.0 is available (this is 0.7.0)" in line
    assert "not while a meeting is running" in line
    assert update.UPGRADE_COMMAND in line


def test_the_notice_comes_once_a_day(standard, monkeypatch):
    found()
    assert update.notice() is not None
    # The second time nothing is worked out at all — no `uv --version`.
    with monkeypatch.context() as patched:
        patched.setattr(update, "offer", lambda **kwargs: pytest.fail("worked out an offer"))
        assert update.notice() is None
    state = update.load_state()
    state["noticed_at"] -= update.NOTICE_INTERVAL_S + 1
    update.save_state(state)
    assert update.notice() is not None


@pytest.fixture
def at_a_terminal(monkeypatch):
    monkeypatch.setattr(cli, "_person_at_terminal", lambda: True)


def test_a_command_ends_with_the_notice(standard, at_a_terminal):
    found()
    result = CliRunner().invoke(main, ["presets"])
    assert result.exit_code == 0
    assert "stenograf 0.8.0 is available" in result.output


def test_a_pipe_gets_no_notice(standard):
    found()
    result = CliRunner().invoke(main, ["presets"])
    assert "is available" not in result.output


def test_doctor_reports_updates_as_a_row_instead(standard, at_a_terminal, monkeypatch):
    from stenograf import doctor

    found()
    check = doctor._update_check()
    assert check.optional and not check.ok
    assert "0.8.0 is available (this is 0.7.0)" in check.detail
    assert update.UPGRADE_COMMAND in check.detail
    # The row, not the row and a line under it.
    assert "doctor" in cli._OWN_UPDATE_REPORT


def test_doctor_says_when_the_check_is_off(standard):
    from stenograf import doctor

    write_settings("[updates]\ncheck = false\n")
    assert "switched off" in doctor._update_check().detail


def test_only_long_commands_check_in_the_background(standard, monkeypatch):
    started = []
    monkeypatch.setattr(update, "check_in_background", lambda: started.append(1))
    CliRunner().invoke(main, ["presets"])
    assert started == []
    CliRunner().invoke(main, ["transcribe", "--help"])
    assert started == [1]


# -- the lock ---------------------------------------------------------------

HOLD = """
import sys, time
sys.path.insert(0, {source!r})
from stenograf import _updater
from pathlib import Path
handle = _updater.acquire(Path(sys.argv[1]), exclusive=sys.argv[2] == "exclusive")
print("held", flush=True)
time.sleep(30)
"""


def holder(path: Path, mode: str) -> subprocess.Popen:
    source = str(Path(update.__file__).parent.parent)
    process = subprocess.Popen(
        [sys.executable, "-c", HOLD.format(source=source), str(path), mode],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "held"
    return process


def test_every_process_shares_the_lock_and_the_updater_excludes_them(tmp_path):
    path = tmp_path / "in-use.lock"
    first, second = holder(path, "shared"), holder(path, "shared")
    try:
        with pytest.raises(_updater.LockHeld):
            _updater.acquire(path, exclusive=True)
        first.kill()  # a crash: nothing gets to clean up
        first.wait(timeout=TOOL_TIMEOUT_S)
        with pytest.raises(_updater.LockHeld):
            _updater.acquire(path, exclusive=True)
    finally:
        for process in (first, second):
            process.kill()
            process.wait(timeout=TOOL_TIMEOUT_S)
    # Windows frees a killed process's LockFileEx lock a moment after the
    # process is reaped: up to 2.2 ms, and in 21 of 30 kills not at once
    # (2026-11-19, Windows 11). The updater polls, so only the test must wait.
    _updater.acquire_within(path, exclusive=True, seconds=1.0).close()


def test_a_command_started_during_the_swap_steps_aside():
    updater = holder(update.lock_path(), "exclusive")
    try:
        assert update.hold_lock() == "updating"
        result = CliRunner().invoke(main, ["presets"])
        assert result.exit_code == 1
        assert "Stenograf is being updated" in result.output
    finally:
        updater.kill()
        updater.wait(timeout=TOOL_TIMEOUT_S)
    assert update.hold_lock() == "held"


def test_a_home_that_cannot_lock_only_loses_self_update(standard, monkeypatch):
    def refuse(path, *, exclusive, seconds):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(_updater, "acquire_within", refuse)
    assert update.hold_lock() == "unsupported"
    assert CliRunner().invoke(main, ["presets"]).exit_code == 0
    found()
    offer = update.offer()
    assert offer is not None and not offer.installable


# -- prepare and hand over --------------------------------------------------


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("the stand-in for uv is a shell wrapper")
    wrapper = tmp_path / "uv-wrapper"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_UV}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr(update, "find_uv", lambda: str(wrapper))
    monkeypatch.setenv("FAKE_UV_LOG", str(tmp_path / "uv-calls.jsonl"))
    return tmp_path / "uv-calls.jsonl"


def test_prepare_tries_the_new_release_on_the_running_python(fake_uv):
    folder = update.prepare("0.8.0")
    assert (folder / "updater.py").read_bytes() == Path(_updater.__file__).read_bytes()
    (call,) = [json.loads(line) for line in fake_uv.read_text().splitlines()]
    assert call["argv"][:6] == [
        "tool", "run", "--python", update.running_python(), "--from", "stenograf==0.8.0",
    ]  # fmt: skip
    assert call["argv"][6:11] == ["python", "-m", "stenograf._updater", "prepare", "--out"]


def test_a_failed_prepare_hands_on_the_new_release_s_own_sentence(fake_uv, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "prepare-fail")
    with pytest.raises(update.UpdateError, match="window does not open here: no Qt"):
        update.prepare("0.8.0")


def test_prepare_can_be_cancelled(fake_uv, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "prepare-slow")
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    began = time.monotonic()
    with pytest.raises(update.UpdateCancelled):
        update.prepare("0.8.0", cancelled=stop.is_set)
    assert time.monotonic() - began < 10


def test_hand_over_writes_the_request_and_starts_the_new_release_s_updater(
    standard, monkeypatch, tmp_path
):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: started.append((argv, kwargs)))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    update.hand_over(
        "0.8.0",
        prepared,
        restart=["/usr/bin/open", "/Applications/Stenograf.app"],
        wait_pids=[11, 10],
        launcher_files=[tmp_path / "launch-target"],
    )
    request = _updater.Request.read(prepared / "request.json")
    assert request.target_version == "0.8.0"
    assert request.environment == standard
    assert request.tool_dir == standard.parent
    assert request.bin_dir == Path("/home/u/.local/bin")
    assert request.wait_pids == (11, 10)
    assert request.lock_file == update.lock_path()
    # `open` hands over and exits 0 whatever becomes of the app.
    assert request.direct is False
    update.hand_over(
        "0.8.0", prepared, restart=[sys.executable, "-m", "stenograf", "--gui"],
        wait_pids=[], launcher_files=[],
    )  # fmt: skip
    assert _updater.Request.read(prepared / "request.json").direct is True
    (argv, kwargs), _second = started
    # The interpreter the environment was built from, isolated, on the copy.
    assert argv[1:] == ["-I", str(prepared / "updater.py"), "apply", str(prepared / "request.json")]
    assert Path(argv[0]) != Path(sys.prefix) / "bin" / "python"
    assert Path(kwargs["cwd"]) == update.work_dir()
    assert update.load_state()["pending"] == {"target": "0.8.0", "from": "0.7.0"}


def test_a_non_standard_install_refuses_to_hand_over(tmp_path):
    with pytest.raises(update.UpdateError):
        update.hand_over("0.8.0", tmp_path, restart=[], wait_pids=[], launcher_files=[])


# -- after the restart ------------------------------------------------------


def test_the_new_version_reports_the_update_once(standard, monkeypatch):
    update.save_state({"pending": {"target": "0.8.0", "from": "0.7.0"}})
    monkeypatch.setattr(update, "__version__", "0.8.0")
    update.mark_started()
    marker = json.loads(_updater.marker_path(update.work_dir()).read_text())
    assert marker == {"version": "0.8.0"}
    assert update.settle() == update.Report(True, "0.8.0", "Updated to 0.8.0.")
    assert update.settle() is None


def test_the_old_version_reports_the_failure_and_remembers_it(standard):
    update.save_state({"pending": {"target": "0.8.0", "from": "0.7.0"}, "latest": "0.8.0"})
    _updater.result_path(update.work_dir()).write_text(
        json.dumps(
            {
                "format": 1,
                "ok": False,
                "version": "0.8.0",
                "message": "Version 0.8.0 did not start. The previous version was restored.",
                "log": "/w/updater.log",
                "rolled_back": True,
            }
        )
    )
    report = update.settle()
    assert report == update.Report(
        False, "0.8.0", "Version 0.8.0 did not start. The previous version was restored."
    )
    assert update.load_state()["failed"] == ["0.8.0"]
    assert not _updater.result_path(update.work_dir()).exists()
    update.save_state({**update.load_state(), "checked_at": time.time()})
    assert update.offer() is None


def test_an_update_that_never_began_is_not_held_against_the_version(standard):
    """A meeting in a terminal stopped it; the version itself was never tried."""
    update.save_state({"pending": {"target": "0.8.0", "from": "0.7.0"}, "latest": "0.8.0"})
    _updater.result_path(update.work_dir()).write_text(
        json.dumps(
            {
                "format": 1,
                "ok": False,
                "version": "0.8.0",
                "message": "Stenograf is in use elsewhere. The update was not started.",
                "log": "/w/updater.log",
                "rolled_back": False,
            }
        )
    )
    report = update.settle()
    assert report is not None and "in use elsewhere" in report.message
    assert "failed" not in update.load_state()
    update.save_state({**update.load_state(), "checked_at": time.time()})
    assert update.offer() is not None


def test_an_updater_that_vanished_still_gets_a_sentence(standard):
    update.save_state({"pending": {"target": "0.8.0", "from": "0.7.0"}})
    report = update.settle()
    assert report is not None and not report.updated
    assert report.message == "The update to 0.8.0 did not finish."
    assert "failed" not in update.load_state()


def test_no_marker_without_an_update_in_flight(standard):
    update.mark_started()
    assert not _updater.marker_path(update.work_dir()).exists()


def test_a_later_start_removes_what_an_interrupted_updater_left(standard):
    from fake_uv import build_environment

    if sys.platform == "win32":
        pytest.skip("the pretend environment holds a shell script")
    parked = _updater.parked_path(standard.parent)
    build_environment(parked, "0.6.0")
    update.work_dir().mkdir(parents=True, exist_ok=True)
    _updater.journal_path(update.work_dir()).write_text("{}")
    update.settle()
    for _ in range(100):
        if not parked.exists():
            break
        time.sleep(0.05)
    assert not parked.exists()
    assert not _updater.journal_path(update.work_dir()).exists()


def test_a_start_leaves_the_parked_version_alone_while_the_updater_still_runs(standard):
    """The updater is then waiting to hear whether *this* start works, and the
    parked version is what it would go back to."""
    from fake_uv import build_environment

    if sys.platform == "win32":
        pytest.skip("the pretend environment holds a shell script")
    parked = _updater.parked_path(standard.parent)
    build_environment(parked, "0.6.0")
    update.work_dir().mkdir(parents=True, exist_ok=True)
    _updater.journal_path(update.work_dir()).write_text("{}")
    updater = holder(_updater.alive_path(update.work_dir()), "exclusive")
    try:
        update.settle()
        time.sleep(0.5)
        assert parked.is_dir()
        assert _updater.journal_path(update.work_dir()).exists()
    finally:
        updater.kill()
        updater.wait(timeout=TOOL_TIMEOUT_S)


# -- how the app is started again --------------------------------------------


def test_on_macos_the_app_is_reopened_through_its_bundle(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("STENOGRAF_APP_BUNDLE", "/Users/u/Apps/Stenograf.app")
    assert update.restart_command(tray=False) == ["/usr/bin/open", "/Users/u/Apps/Stenograf.app"]
    assert update.restart_command(tray=True)[-2:] == ["--args", "--tray"]
    # The bundle's launcher is the app's parent, and must be gone before `open`.
    import os

    assert update.launch_chain() == [os.getpid(), os.getppid()]


def test_from_a_terminal_it_is_started_as_it_was(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert update.restart_command(tray=True) == [
        sys.executable, "-m", "stenograf", "--gui", "--tray",
    ]  # fmt: skip
    import os

    assert update.launch_chain() == [os.getpid()]


def test_the_names_two_releases_must_agree_on_have_not_changed(standard, monkeypatch, tmp_path):
    """An old app's request names the work folder and the lock; the new app the
    updater starts works them out again. If a release moved any of these, the
    updater would watch one lock while the app held another — and roll back a
    version that is running."""
    import hashlib
    import os

    from stenograf import paths

    monkeypatch.setattr(update, "root", REAL_ROOT)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "tools" / "stenograf"))
    monkeypatch.setattr(Path, "home", lambda: Path("/home/u"))
    monkeypatch.setenv("LOCALAPPDATA", "/home/u/AppData/Local")
    # Not the data folder, which a variable can move from under a running app.
    monkeypatch.setenv("STENOGRAF_DATA", "/elsewhere")
    expected = {
        "darwin": Path("/home/u/Library/Application Support/stenograf/update"),
        "linux": Path("/home/u/.local/state/stenograf/update"),
        "win32": Path("/home/u/AppData/Local/stenograf/update"),
    }
    for platform, folder in expected.items():
        monkeypatch.setattr(sys, "platform", platform)
        assert update.root() == folder
    assert paths.data_dir() == Path("/elsewhere")
    key = hashlib.sha256(os.path.normcase(os.path.abspath(sys.prefix)).encode()).hexdigest()[:12]
    work = update.work_dir()
    assert work == update.root() / key
    assert update.lock_path() == work / "in-use.lock"
    assert _updater.marker_path(work) == work / "started.json"
    assert _updater.result_path(work) == work / "result.json"
    assert _updater.alive_path(work) == work / "updater.alive"
    assert _updater.journal_path(work) == work / "journal.json"
    tools = Path(sys.prefix).parent
    assert _updater.parked_path(tools) == tools.parent / "tools-stenograf-previous"
    assert _updater.trash_path(tools) == tools.parent / "tools-stenograf-failed"
