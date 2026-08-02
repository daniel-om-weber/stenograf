"""The visible output home: where it is, folder allocation, the --last scan."""

import sys
from datetime import datetime

from stenograf.output import (
    allocate_meeting_dir,
    created_at_from_dir_name,
    default_output_home,
    latest_meeting_dir,
)
from stenograf.paths import documents_dir

WHEN = datetime(2026, 7, 10, 9, 15, 0)


def _linux_home(monkeypatch, tmp_path, user_dirs: str | None) -> None:
    """A Linux session whose documents folder is whatever ``user_dirs`` says."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on a Windows host
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    if user_dirs is not None:
        (tmp_path / ".config").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".config" / "user-dirs.dirs").write_text(user_dirs, encoding="utf-8")


def test_documents_is_the_folder_the_desktop_names(tmp_path, monkeypatch):
    # The localised case: writing to ~/Documents here would create a second
    # documents tree that the file manager's sidebar does not list.
    _linux_home(
        monkeypatch,
        tmp_path,
        "# written by xdg-user-dirs-update\n"
        'XDG_DESKTOP_DIR="$HOME/Desktop"\n'
        'XDG_DOCUMENTS_DIR="$HOME/Dokumente"\n'
        'XDG_MUSIC_DIR="$HOME/Musik"\n',
    )

    assert documents_dir() == tmp_path / "Dokumente"
    assert default_output_home() == tmp_path / "Dokumente" / "Meetings"


def test_documents_falls_back_to_the_english_default(tmp_path, monkeypatch):
    # No user-dirs.dirs at all (a minimal WM, a container): ~/Documents is both
    # xdg-user-dirs' own default and what macOS and Windows use.
    _linux_home(monkeypatch, tmp_path, None)

    assert documents_dir() == tmp_path / "Documents"
    assert default_output_home() == tmp_path / "Documents" / "Meetings"


def test_an_absolute_or_disabled_entry_is_honoured(tmp_path, monkeypatch):
    _linux_home(monkeypatch, tmp_path, f'XDG_DOCUMENTS_DIR="{tmp_path}/elsewhere/docs"\n')
    assert documents_dir() == tmp_path / "elsewhere" / "docs"

    # An entry pointing at $HOME means "disabled" in the XDG spec — the output
    # home must not become the home directory itself.
    _linux_home(monkeypatch, tmp_path, 'XDG_DOCUMENTS_DIR="$HOME/"\n')
    assert documents_dir() == tmp_path / "Documents"


def test_off_linux_the_file_is_not_consulted(tmp_path, monkeypatch):
    # macOS localises the *display* name only: ~/Documents is the real path, and
    # a stray user-dirs.dirs (from a dual-booted home dir) must not win there.
    _linux_home(monkeypatch, tmp_path, 'XDG_DOCUMENTS_DIR="$HOME/Dokumente"\n')
    monkeypatch.setattr(sys, "platform", "darwin")

    assert documents_dir() == tmp_path / "Documents"


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
