"""CLI tests for `steno notes`, the `--notes` flag, and the combined-note export."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_cli import fake_load_backends, write_wav

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
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path / "steno-data"))
    monkeypatch.delenv("STENOGRAF_NOTES_BACKEND", raising=False)


class FakeBackend:
    name = "fake"
    model = "fake-model"

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


def archive_meeting(*, title=None) -> str:
    """Register one meeting in the (isolated) managed archive."""
    from stenograf.archive import MeetingArchive, MeetingRecord

    archive = MeetingArchive.load()
    meeting_id = archive.allocate_id(datetime(2026, 7, 10, 14, 30))
    meeting_dir = archive.meeting_dir(meeting_id)
    meeting_dir.mkdir(parents=True)
    write_transcript_json(meeting_dir / "transcript.json", title=title)
    archive.add(
        MeetingRecord(
            id=meeting_id,
            title=title,
            created_at="2026-07-10T14:30:00",
            duration_s=1.0,
            language=Language.GERMAN,
            formats=("json",),
            dir=meeting_dir,
        )
    )
    return meeting_id


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


# ---- steno notes <archive id> --------------------------------------------------


def test_notes_on_archive_id_writes_into_meeting_dir_and_backfills_title(fake_backend):
    meeting_id = archive_meeting(title=None)

    result = CliRunner().invoke(cli.main, ["notes", meeting_id])

    assert result.exit_code == 0, result.output
    from stenograf.archive import MeetingArchive

    archive = MeetingArchive.load()
    assert (archive.meeting_dir(meeting_id) / "transcript.notes.md").exists()
    assert archive.get(meeting_id).title == "Quartalsplanung"  # back-filled
    assert "title: Quartalsplanung" in result.output
    listing = CliRunner().invoke(cli.main, ["meetings", "list"])
    assert "Quartalsplanung" in listing.output


def test_notes_never_overwrites_a_user_set_title(fake_backend):
    meeting_id = archive_meeting(title="Weekly Sync")

    result = CliRunner().invoke(cli.main, ["notes", meeting_id])

    assert result.exit_code == 0, result.output
    from stenograf.archive import MeetingArchive

    assert MeetingArchive.load().get(meeting_id).title == "Weekly Sync"


def test_notes_unknown_id_fails_with_guidance(fake_backend):
    result = CliRunner().invoke(cli.main, ["notes", "meeting-19700101-000000"])
    assert result.exit_code != 0
    assert "meetings list" in result.output


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


# ---- export --------------------------------------------------------------------


def test_notes_export_dir_writes_combined_note(tmp_path, fake_backend):
    meeting_id = archive_meeting()
    vault = tmp_path / "vault" / "Meetings"

    result = CliRunner().invoke(cli.main, ["notes", meeting_id, "--export-dir", str(vault)])

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
    settings.write_text(f'[notes.export]\ndir = "{vault}"\n', encoding="utf-8")
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
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(
        cli.main, ["transcribe", str(audio), "--out", str(tmp_path), "--notes"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.notes.md").exists()
    assert "notes: wrote" in result.output


def test_transcribe_notes_failure_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
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

    monkeypatch.setattr(cli, "_load_backends", fake_load_backends)
    monkeypatch.setattr(notes_pkg, "create_backend", explode)
    audio = tmp_path / "meeting.wav"
    write_wav(audio)

    result = CliRunner().invoke(cli.main, ["transcribe", str(audio), "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
