"""CLI tests for `steno notes`, the `--notes` flag, and the combined-note export."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import write_wav
from test_cli import fake_load_backends

from stenograf import cli
from stenograf import notes as notes_pkg
from stenograf.config import Language, MeetingProfile
from stenograf.transcript import Transcript, TranscriptEntry

NOTES_JSON = json.dumps(
    {
        "title": "Quartalsplanung",
        "summary": "Es wurde das Quartal geplant.",
        "decisions": ["Juli-Release"],
        "action_items": [{"task": "Budget entwerfen", "owner": "Anna"}],
        "highlights": [],
        "open_questions": [],
    }
)


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    from stenograf import output

    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "steno-data"))
    monkeypatch.setattr(output, "default_output_home", lambda: tmp_path / "meetings-home")
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)


class FakeBackend:
    name = "fake"
    model = "fake-model"
    max_input_chars = 48_000

    def __init__(self, available=True, response=NOTES_JSON):
        self.available = available
        self.response = response

    def is_available(self):
        return self.available

    def complete(self, messages, schema):
        return self.response


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(notes_pkg, "create_backend", lambda name, settings: backend)
    return backend


def write_transcript_json(path: Path, *, title=None) -> Transcript:
    transcript = Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(title=title),
        entries=[TranscriptEntry(speaker="Local-1", text="Hallo.", start=0.0, end=1.0)],
    )
    path.write_text(transcript.to_json(), encoding="utf-8")
    return transcript


def meeting_folder(tmp_path, name="meeting-20260710-143000", *, title=None) -> Path:
    """One finished meeting folder in the (isolated) output home."""
    meeting_dir = tmp_path / "meetings-home" / name
    meeting_dir.mkdir(parents=True)
    write_transcript_json(meeting_dir / "transcript.json", title=title)
    return meeting_dir


# ---- steno notes <path> --------------------------------------------------------


def test_notes_on_a_transcript_path_writes_siblings(tmp_path, fake_backend):
    path = tmp_path / "transcript.json"
    write_transcript_json(path)

    result = CliRunner().invoke(cli.main, ["notes", str(path)])

    assert result.exit_code == 0, result.output
    md = (tmp_path / "transcript.notes.md").read_text(encoding="utf-8")
    assert md.startswith("# Quartalsplanung")
    saved = json.loads((tmp_path / "transcript.notes.json").read_text(encoding="utf-8"))
    assert saved["provenance"]["backend"] == "fake"


def test_notes_on_a_flat_layout_transcript_names_siblings_after_it(tmp_path, fake_backend):
    path = tmp_path / "standup.transcript.json"
    write_transcript_json(path)

    result = CliRunner().invoke(cli.main, ["notes", str(path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "standup.transcript.notes.md").exists()


def test_notes_on_garbage_json_fails_cleanly(tmp_path, fake_backend):
    path = tmp_path / "transcript.json"
    path.write_text("not json", encoding="utf-8")

    result = CliRunner().invoke(cli.main, ["notes", str(path)])

    assert result.exit_code != 0
    assert "not a readable transcript" in result.output


# ---- steno notes <folder> / --last ----------------------------------------------


def test_notes_on_a_meeting_folder_uses_its_transcript(tmp_path, fake_backend):
    meeting_dir = meeting_folder(tmp_path)

    result = CliRunner().invoke(cli.main, ["notes", str(meeting_dir)])

    assert result.exit_code == 0, result.output
    assert (meeting_dir / "transcript.notes.md").exists()


def test_notes_on_a_folder_without_a_transcript_errors(tmp_path, fake_backend):
    empty = tmp_path / "meetings-home" / "meeting-20260710-143000"
    empty.mkdir(parents=True)

    result = CliRunner().invoke(cli.main, ["notes", str(empty)])

    assert result.exit_code != 0
    assert "no transcript.json" in result.output


def test_notes_last_picks_the_newest_meeting_folder(tmp_path, fake_backend):
    meeting_folder(tmp_path, "meeting-20260709-090000")
    newest = meeting_folder(tmp_path, "meeting-20260710-143000")
    # Newer folder name, but no transcript.json (a crashed run) — skipped.
    (tmp_path / "meetings-home" / "meeting-20260711-080000").mkdir()

    result = CliRunner().invoke(cli.main, ["notes", "--last"])

    assert result.exit_code == 0, result.output
    assert str(newest) in result.output  # says which meeting it picked
    assert (newest / "transcript.notes.md").exists()


def test_notes_last_with_an_empty_home_fails_with_guidance(tmp_path, fake_backend):
    result = CliRunner().invoke(cli.main, ["notes", "--last"])
    assert result.exit_code != 0
    assert "no finished meeting" in result.output


def test_notes_requires_a_path_or_last_but_not_both(tmp_path, fake_backend):
    assert CliRunner().invoke(cli.main, ["notes"]).exit_code != 0
    meeting_dir = meeting_folder(tmp_path)
    result = CliRunner().invoke(cli.main, ["notes", str(meeting_dir), "--last"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_notes_backend_down_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        notes_pkg, "create_backend", lambda name, settings: FakeBackend(available=False)
    )
    path = tmp_path / "transcript.json"
    write_transcript_json(path)

    result = CliRunner().invoke(cli.main, ["notes", str(path)])

    assert result.exit_code != 0
    assert not (tmp_path / "transcript.notes.md").exists()
    assert not (tmp_path / "transcript.notes.json").exists()


def test_notes_reports_a_bug_as_a_traceback_not_a_clean_error(tmp_path, monkeypatch):
    # Documented failure modes (backend down, bad settings) become clean CLI
    # errors; a programming bug must keep its traceback, or an AttributeError
    # reads like "Ollama down" and never gets reported.
    def buggy(*args, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'entries'")

    monkeypatch.setattr(cli.notes, "_generate_and_write_notes", buggy)
    path = tmp_path / "transcript.json"
    write_transcript_json(path)

    result = CliRunner().invoke(cli.main, ["notes", str(path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, AttributeError)


# ---- export --------------------------------------------------------------------


def test_notes_export_dir_writes_combined_note(tmp_path, fake_backend):
    # The export date comes from the folder name, which encodes the start time.
    meeting_dir = meeting_folder(tmp_path)
    vault = tmp_path / "vault" / "Meetings"

    result = CliRunner().invoke(cli.main, ["notes", str(meeting_dir), "--export-dir", str(vault)])

    assert result.exit_code == 0, result.output
    exported = vault / "2026-07-10 – Quartalsplanung.md"
    assert exported.exists()
    text = exported.read_text(encoding="utf-8")
    assert text.startswith("---\n")  # frontmatter
    assert "> [!quote]- Transcript" in text


def test_notes_export_defaults_from_settings_and_no_export_disables(tmp_path, fake_backend):
    import os

    vault = tmp_path / "vault"
    settings = Path(os.environ["STENOGRAF_DATA"]) / "settings.toml"
    settings.parent.mkdir(parents=True, exist_ok=True)
    # as_posix(): a raw Windows path in a TOML basic string is invalid (\U…).
    settings.write_text(f'[notes.export]\ndir = "{vault.as_posix()}"\n', encoding="utf-8")
    path = tmp_path / "transcript.json"
    write_transcript_json(path)

    result = CliRunner().invoke(cli.main, ["notes", str(path)])
    assert result.exit_code == 0, result.output
    assert list(vault.glob("*.md")), "settings-configured export dir was not used"

    result = CliRunner().invoke(cli.main, ["notes", str(path), "--no-export"])
    assert result.exit_code == 0, result.output
    assert len(list(vault.glob("*.md"))) == 1  # unchanged — export skipped


# ---- --notes flag on transcribe -------------------------------------------------


def test_transcribe_with_notes_flag_writes_notes(tmp_path, monkeypatch, fake_backend):
    monkeypatch.setattr(cli.loaders, "load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--notes"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.notes.md").exists()
    assert "notes: wrote" in result.output


def test_transcribe_notes_failure_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.loaders, "load_backends", fake_load_backends)
    monkeypatch.setattr(
        notes_pkg, "create_backend", lambda name, settings: FakeBackend(available=False)
    )
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--notes"]
    )

    assert result.exit_code == 0, result.output  # transcript stands; run succeeds
    assert (tmp_path / "transcript.md").exists()
    assert not (tmp_path / "transcript.notes.md").exists()
    assert "notes failed" in result.output
    assert "steno notes" in result.output  # retry guidance


def test_transcribe_without_notes_flag_never_touches_a_backend(tmp_path, monkeypatch):
    def explode(name, settings):
        raise AssertionError("--notes was not given; no backend may be created")

    monkeypatch.setattr(cli.loaders, "load_backends", fake_load_backends)
    monkeypatch.setattr(notes_pkg, "create_backend", explode)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
