"""The app's update line, its Settings row and its menu-bar entry.

The library behind them is ``test_update.py``; what is tested here is *when*
the app acts — never during a meeting, only on a click — and that the QML for
every phase of the line builds without a binding warning.
"""

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtQml import QQmlComponent  # noqa: E402
from test_gui import _Warnings, gui, pump, qt_app  # noqa: E402, F401

from stenograf import update  # noqa: E402
from stenograf.gui.app import QML_DIR  # noqa: E402

OFFER = update.Offer("0.8.0", True, update.UPGRADE_COMMAND, "")
NOTICE = update.Offer("0.8.0", False, update.UPGRADE_COMMAND, "")


@pytest.fixture
def updates(gui, monkeypatch, tmp_path):  # noqa: F811
    """The update object of a built shell, looking at a world with an offer."""
    shell, _engine = gui
    world = {"offer": OFFER, "checks": [], "handed": [], "quit": []}
    monkeypatch.setattr(update, "enabled", lambda: True)
    monkeypatch.setattr(update, "check", lambda *, force=False: world["checks"].append(force))
    monkeypatch.setattr(update, "offer", lambda *, include_failed=False: world["offer"])
    monkeypatch.setattr(update, "prepare", lambda version, *, cancelled: tmp_path / "prepared")
    monkeypatch.setattr(
        update, "hand_over", lambda version, prepared, **how: world["handed"].append((version, how))
    )
    monkeypatch.setattr(shell, "quit_app", lambda: world["quit"].append(True))
    shell.updates.world = world
    return shell.updates


def begun(updates, phase):
    updates.begin(tray=False)
    pump(lambda: updates.state["phase"] == phase)
    return updates


def test_a_found_release_becomes_the_line(updates):
    begun(updates, "available")
    assert updates.state["text"] == "Stenograf 0.8.0 is available"
    assert updates.state["version"] == "0.8.0"
    assert "0.8.0 is available" in updates.state["summary"]
    assert updates.world["checks"] == [False]  # the daily check, not a forced one


def test_an_install_that_cannot_update_itself_shows_the_command(updates):
    updates.world["offer"] = NOTICE
    begun(updates, "notice")
    assert updates.state["detail"] == update.UPGRADE_COMMAND
    updates.install()  # there is no button, and a stray call does nothing
    assert updates.state["phase"] == "notice"


def test_one_click_prepares_hands_over_and_quits(updates):
    begun(updates, "available")
    updates.install()
    pump(lambda: updates.state["phase"] == "restarting")
    ((version, how),) = updates.world["handed"]
    assert version == "0.8.0"
    assert how["restart"] == update.restart_command(tray=False)
    assert how["wait_pids"] == update.launch_chain()
    assert updates.world["quit"] == [True]


def test_nothing_starts_while_a_meeting_runs(updates, gui):  # noqa: F811
    shell, _engine = gui
    begun(updates, "available")
    shell.screen("Meeting").set(active=True)
    assert updates.state["quiet"] is True
    updates.install()
    assert updates.state["phase"] == "available"
    assert updates.world["handed"] == []
    shell.screen("Meeting").set(active=False)
    assert updates.state["quiet"] is False


def test_a_meeting_that_starts_meanwhile_keeps_the_app_open(updates, gui, monkeypatch, tmp_path):  # noqa: F811
    shell, _engine = gui
    release = threading.Event()

    def slow(version, *, cancelled):
        release.wait(5)
        return tmp_path / "prepared"

    monkeypatch.setattr(update, "prepare", slow)
    monkeypatch.setattr(updates, "_meeting_changed", lambda: None)  # prepare is not cancelled
    begun(updates, "available")
    updates.install()
    assert updates.state["phase"] == "preparing"
    shell.screen("Meeting").set(active=True)
    release.set()
    pump(lambda: updates.state["phase"] == "ready")
    assert updates.world["quit"] == []
    updates.restart()  # the tray entry, clicked during the meeting
    assert updates.state["phase"] == "ready" and updates.world["handed"] == []
    shell.screen("Meeting").set(active=False)
    updates.restart()
    assert updates.state["phase"] == "restarting"
    assert updates.world["quit"] == [True]


def test_starting_a_meeting_cancels_a_prepare(updates, gui, monkeypatch):  # noqa: F811
    shell, _engine = gui

    def until_cancelled(version, *, cancelled):
        for _ in range(500):
            if cancelled():
                raise update.UpdateCancelled("cancelled")
            threading.Event().wait(0.01)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(update, "prepare", until_cancelled)
    begun(updates, "available")
    updates.install()
    shell.screen("Meeting").set(active=True)
    pump(lambda: updates.state["phase"] == "available")  # the offer is back, untouched
    assert updates.world["handed"] == []


@pytest.mark.parametrize("page", ["Transcribe", "Notes"])
def test_nothing_starts_while_a_transcription_or_a_notes_run_is_going(updates, gui, page):  # noqa: F811
    shell, _engine = gui
    begun(updates, "available")
    shell.screen(page).set(busy=True)
    updates.install()
    assert updates.state["phase"] == "available"
    assert updates.state["detail"] == "Finish what is running first."
    shell.screen(page).set(busy=False)
    updates.install()
    pump(lambda: updates.state["phase"] == "restarting")


def test_a_cancel_as_the_trial_run_finishes_still_cancels(updates, monkeypatch, tmp_path):
    def finishes_anyway(version, *, cancelled):
        updates.cancel()  # the click arrives while the last line is being read
        return tmp_path / "prepared"

    monkeypatch.setattr(update, "prepare", finishes_anyway)
    begun(updates, "available")
    updates.install()
    pump(lambda: updates.state["phase"] == "available")
    assert updates.world["handed"] == [] and updates.world["quit"] == []


def test_a_second_attempt_is_cancelled_by_a_meeting_like_the_first(
    updates, gui, monkeypatch  # noqa: F811
):
    """What an earlier attempt prepared must not make this one look finished."""
    shell, _engine = gui

    def refuse(version, prepared, **how):
        raise update.UpdateError("uv is gone")

    monkeypatch.setattr(update, "hand_over", refuse)
    begun(updates, "available")
    updates.install()
    pump(lambda: updates.state["phase"] == "failed")
    updates.dismiss()
    updates.check_now()
    pump(lambda: updates.state["phase"] == "available")

    def until_cancelled(version, *, cancelled):
        for _ in range(500):
            if cancelled():
                raise update.UpdateCancelled("cancelled")
            threading.Event().wait(0.01)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(update, "prepare", until_cancelled)
    updates.install()
    shell.screen("Meeting").set(active=True)
    pump(lambda: updates.state["phase"] == "available")


def test_a_prepare_that_fails_says_why(updates, monkeypatch):
    def fail(version, *, cancelled):
        raise update.UpdateError("This release's window does not open here: no Qt")

    monkeypatch.setattr(update, "prepare", fail)
    begun(updates, "available")
    updates.install()
    pump(lambda: updates.state["phase"] == "failed")
    assert updates.state["text"] == "The update could not be prepared."
    assert "no Qt" in updates.state["detail"]
    assert updates.world["quit"] == []


def test_a_hand_over_that_fails_leaves_the_app_running(updates, monkeypatch):
    def refuse(version, prepared, **how):
        raise update.UpdateError("uv is gone")

    monkeypatch.setattr(update, "hand_over", refuse)
    begun(updates, "available")
    updates.install()
    pump(lambda: updates.state["phase"] == "failed")
    assert updates.state["detail"] == "uv is gone"
    assert updates.world["quit"] == []


def test_later_hides_the_offer_until_the_next_release(updates):
    begun(updates, "available")
    updates.dismiss()
    assert updates.state["phase"] == "none"
    updates._look(force=False)
    pump(lambda: len(updates.world["checks"]) == 2)
    pump(lambda: True)
    assert updates.state["phase"] == "none"
    updates.world["offer"] = update.Offer("0.8.1", True, update.UPGRADE_COMMAND, "")
    updates._look(force=False)
    pump(lambda: updates.state["phase"] == "available")
    assert updates.state["version"] == "0.8.1"


def test_check_for_updates_asks_again_and_brings_a_hidden_offer_back(updates):
    begun(updates, "available")
    updates.dismiss()
    updates.check_now()
    pump(lambda: updates.state["phase"] == "available")
    assert updates.world["checks"] == [False, True]


def test_the_switch_is_named_in_settings_and_shows_no_line(updates, monkeypatch):
    monkeypatch.setattr(update, "enabled", lambda: False)
    updates.begin(tray=False)
    pump(lambda: "switched off" in updates.state["summary"])
    assert updates.state["phase"] == "none"
    assert updates.world["checks"] == []


@pytest.mark.parametrize(
    ("report", "phase"),
    [
        (update.Report(True, "0.8.0", "Updated to 0.8.0."), "updated"),
        (update.Report(False, "0.8.0", "Version 0.8.0 did not start."), "failed"),
    ],
)
def test_the_first_start_after_an_update_reports_it(updates, monkeypatch, report, phase):
    updates.world["offer"] = None
    monkeypatch.setattr(update, "settle", lambda: report)
    marked = []
    monkeypatch.setattr(update, "mark_started", lambda: marked.append(True))
    updates.begin(tray=False)
    assert marked == [True]
    assert updates.state["phase"] == phase
    assert updates.state["text"] == report.message
    pump(lambda: "no newer release" in updates.state["summary"])
    assert updates.state["phase"] == phase  # a check does not wipe the report
    updates.dismiss()
    assert updates.state["phase"] == "none"


PHASES = {
    "available": ("Stenograf 0.8.0 is available", ""),
    "notice": ("Stenograf 0.8.0 is available", update.UPGRADE_COMMAND),
    "preparing": ("Preparing the update to 0.8.0…", ""),
    "ready": ("Stenograf 0.8.0 is ready", "Finish what is running first."),
    "restarting": ("Restarting to update…", ""),
    "updated": ("Updated to 0.8.0.", ""),
    "failed": ("The update could not be prepared.", "error: No solution found"),
}


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_the_line_builds_in_every_phase(gui, phase):  # noqa: F811
    shell, engine = gui
    text, detail = PHASES[phase]
    with _Warnings() as log:
        shell.updates.set(phase=phase, version="0.8.0", text=text, detail=detail)
        for name in ("Home", "Settings"):
            component = QQmlComponent(engine, str(QML_DIR / f"{name}.qml"))
            properties = {"app": shell}
            if shell.screen(name) is not None:
                properties["screen"] = shell.screen(name)
            assert component.createWithInitialProperties(properties) is not None, name
        shell.updates.set(quiet=True)
        assert log.seen == []


class TestTrayEntry:
    @pytest.fixture
    def tray(self, gui):  # noqa: F811
        from stenograf.gui.tray import Tray

        shell, _engine = gui
        return Tray(shell)

    def test_the_menu_offers_the_update_only_when_there_is_one(self, tray, updates):
        tray._relabel()
        assert not tray.update.isVisible()
        begun(updates, "available")
        tray._relabel()
        assert tray.update.isVisible()
        assert tray.update.text() == "Update to 0.8.0…"

    def test_not_during_a_meeting(self, tray, updates, gui):  # noqa: F811
        shell, _engine = gui
        begun(updates, "available")
        shell.screen("Meeting").set(active=True, phase="rec")
        tray._relabel()
        assert not tray.update.isVisible()

    def test_a_prepared_update_is_one_entry_away(self, tray, updates, tmp_path):
        begun(updates, "available")
        updates._prepared = tmp_path / "prepared"
        updates.set(phase="ready")
        tray._relabel()
        assert tray.update.text() == "Restart to update"
        tray.update.trigger()
        assert updates.state["phase"] == "restarting"
        assert updates.world["quit"] == [True]
