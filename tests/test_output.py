"""The visible output home: folder allocation and the --last scan (Stage C)."""

from datetime import datetime

from stenograf.output import (
    allocate_meeting_dir,
    created_at_from_dir_name,
    latest_meeting_dir,
)

WHEN = datetime(2026, 7, 10, 9, 15, 0)


def _finished(home, name):
    d = home / name
    d.mkdir(parents=True)
    (d / "transcript.json").write_text("{}", encoding="utf-8")
    return d


def test_allocate_names_the_dir_after_the_start_time(tmp_path):
    d = allocate_meeting_dir(tmp_path, WHEN)
    assert d == tmp_path / "meeting-20260710-091500"
    assert not d.exists()  # nothing is created — an aborted run leaves no trace


def test_allocate_suffixes_on_collision(tmp_path):
    (tmp_path / "meeting-20260710-091500").mkdir()
    assert allocate_meeting_dir(tmp_path, WHEN).name == "meeting-20260710-091500-2"
    # Any on-disk entry collides, not just dirs (a stray file must not be clobbered).
    (tmp_path / "meeting-20260710-091500-2").write_text("")
    assert allocate_meeting_dir(tmp_path, WHEN).name == "meeting-20260710-091500-3"


def test_latest_is_newest_by_name_skipping_unfinished_and_unrelated(tmp_path):
    _finished(tmp_path, "meeting-20260709-100000")
    newest_finished = _finished(tmp_path, "meeting-20260710-090000")
    # Newer by name but no transcript.json (a crashed run) — skipped.
    (tmp_path / "meeting-20260711-080000").mkdir()
    # Non-meeting names never match, whatever they contain.
    _finished(tmp_path, "holiday-photos")

    assert latest_meeting_dir(tmp_path) == newest_finished


def test_latest_none_for_a_missing_or_empty_home(tmp_path):
    assert latest_meeting_dir(tmp_path / "nope") is None
    assert latest_meeting_dir(tmp_path) is None


def test_created_at_round_trips_the_dir_name():
    assert created_at_from_dir_name("meeting-20260710-091500") == WHEN
    assert created_at_from_dir_name("meeting-20260710-091500-2") == WHEN  # suffixed
    assert created_at_from_dir_name("holiday-photos") is None
    assert created_at_from_dir_name("meeting-20261301-000000") is None  # month 13
