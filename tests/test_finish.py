"""The finish card's library side: the titled folder name, the context line
the notes see, retitling a meeting, and the meeting type chosen afterwards."""

from __future__ import annotations

from datetime import datetime

import pytest
from conftest import write_settings

from stenograf import output
from stenograf.config import MeetingProfile
from stenograf.flow import FinishChoices, MeetingRun, resolve_meeting_request
from stenograf.notes.prompt import build_messages
from stenograf.transcript import Transcript, TranscriptEntry

WHEN = datetime(2026, 11, 16, 9, 15)


class TestTitledFolder:
    @pytest.mark.parametrize(
        ("title", "suffix"),
        [
            ("Weekly sync", "-weekly-sync"),
            ("Budget: Q3 / Freigabe!", "-budget-q3-freigabe"),
            ("Müller & Söhne", "-müller-söhne"),
            ("   ", ""),
            (None, ""),
        ],
    )
    def test_the_title_becomes_a_suffix_after_the_date(self, title, suffix):
        assert output.titled_dir_name(WHEN, title) == "meeting-20261116-091500" + suffix

    def test_retitling_replaces_the_suffix_and_never_overwrites(self, tmp_path):
        folder = tmp_path / "meeting-20261116-091500"
        folder.mkdir()
        (folder / "transcript.md").write_text("x", encoding="utf-8")
        first = output.retitle_meeting_dir(folder, "Standup")
        assert first.name == "meeting-20261116-091500-standup"
        assert (first / "transcript.md").exists(), "the files move with the folder"
        second = output.retitle_meeting_dir(first, "Planning")
        assert second.name == "meeting-20261116-091500-planning", "replaced, not stacked"
        (tmp_path / "meeting-20261116-091500-retro").mkdir()  # taken by another meeting
        third = output.retitle_meeting_dir(second, "Retro")
        assert third.name == "meeting-20261116-091500-retro-2"
        assert output.created_at_from_dir_name(third.name) == WHEN
        assert output.latest_meeting_dir(tmp_path) is None  # no transcript.json yet

    def test_a_folder_the_user_named_is_refused(self, tmp_path):
        folder = tmp_path / "my-notes"
        folder.mkdir()
        with pytest.raises(ValueError, match="not a date-named"):
            output.retitle_meeting_dir(folder, "x")


def test_the_context_reaches_the_notes_prompt_and_survives_json():
    transcript = Transcript(
        language=None,
        profile=MeetingProfile(title="Weekly", context="  Budget freigeben "),
        entries=[TranscriptEntry("Local-1", "Hallo.", 0.0, 1.0)],
    )
    assert transcript.profile.context == "Budget freigeben"
    assert Transcript.from_json(transcript.to_json()) == transcript
    system = build_messages(transcript)[0]["content"]
    assert "Budget freigeben" in system


def _meeting(tmp_path, monkeypatch, **request) -> MeetingRun:
    monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings")
    run = MeetingRun(resolve_meeting_request(mic=True, system=True, diarize=False, **request))
    output.write_transcript(
        Transcript(language=None, profile=run.request.profile, entries=[]),
        run.out_dir,
        run.basename,
        ["json"],
    )
    return run


class TestMeetingRun:
    def test_retitle_renames_and_writes_the_title(self, tmp_path, monkeypatch):
        run = _meeting(tmp_path, monkeypatch)
        before = run.out_dir
        after = run.retitle("Weekly sync")
        assert after == run.out_dir and after.name == before.name + "-weekly-sync"
        assert output.load_transcript(after)[0].profile.title == "Weekly sync"
        assert run.audio_path.parent == after

    def test_retitle_waits_for_the_notes(self, tmp_path, monkeypatch):
        run = _meeting(tmp_path, monkeypatch)
        run.notes_running = True
        with pytest.raises(RuntimeError, match="notes"):
            run.retitle("x")

    def test_a_type_chosen_afterwards_swaps_only_the_notes_setup(self, tmp_path, monkeypatch):
        write_settings(
            '[notes]\nbackend = "ollama"\n\n[meetings.board]\n'
            'title = "Board"\n\n[meetings.board.notes]\nbackend = "mlx"\n'
        )
        run = _meeting(tmp_path, monkeypatch)
        transcript = output.load_transcript(run.out_dir)[0]
        run.settle(FinishChoices(notes=True, preset="board", context="Q3"))
        transcript, warmer = run._apply_finish(transcript, None)
        assert run.request.settings.notes.backend == "mlx"
        assert run.request.preset == "board"
        assert transcript.profile.context == "Q3"
        assert transcript.profile.title is None, "the type's title is the card's to set"

    def test_only_the_first_answer_counts(self, tmp_path, monkeypatch):
        run = _meeting(tmp_path, monkeypatch)
        run.settle(FinishChoices(notes=True, preset=None))
        run.settle(FinishChoices(notes=False, preset=None))
        assert run._finish == FinishChoices(notes=True, preset=None)
