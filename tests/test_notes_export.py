from datetime import datetime

from stenograf.config import Language, MeetingProfile
from stenograf.notes import ActionItem, MeetingNotes
from stenograf.notes.export import export_note, render_note
from stenograf.transcript import Transcript, TranscriptEntry

CREATED = datetime(2026, 7, 10, 14, 30)


def transcript() -> Transcript:
    return Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(),
        entries=[
            TranscriptEntry(speaker="Local-1", text="Hallo zusammen.", start=0.0, end=2.0),
            TranscriptEntry(
                speaker="Remote-1", text="Fangen wir an.", start=2.0, end=4.0, provisional=True
            ),
        ],
    )


def notes(title="Quartalsplanung") -> MeetingNotes:
    return MeetingNotes(
        title=title,
        summary="Es wurde geplant.",
        decisions=("Juli-Release",),
        action_items=(
            ActionItem(task="Budget entwerfen", owner="Anna", due="Freitag"),
            ActionItem(task="Raum buchen", owner="Anna"),
            ActionItem(task="Agenda schreiben", owner="Ben"),
            ActionItem(task="Protokoll ablegen"),
        ),
        open_questions=("Q3-Einstellungen?",),
    )


def test_render_note_structure():
    text = render_note(transcript(), notes(), created_at=CREATED)
    assert text.startswith("---\n")
    assert 'title: "Quartalsplanung"' in text
    assert "date: 2026-07-10" in text
    assert "created: 2026-07-10T14:30" in text
    assert "source: stenograf" in text
    assert "language: de" in text
    assert "# Quartalsplanung" in text
    assert "## Decisions" in text
    # Action items grouped per owner, unassigned last.
    anna = text.index("**Anna**")
    ben = text.index("**Ben**")
    unassigned = text.index("**Unassigned**")
    assert anna < ben < unassigned
    assert "- [ ] Budget entwerfen (due Freitag)" in text
    # Collapsible transcript callout with every entry quoted.
    assert "> [!quote]- Transcript" in text
    assert "> **Local-1** [0:00]: Hallo zusammen." in text
    assert "> **Remote-1** [0:02] *(overlap)*: Fangen wir an." in text


def test_render_note_escapes_double_quotes_in_title():
    text = render_note(transcript(), notes(title='Projekt "Alpha" Review'), created_at=CREATED)
    assert "title: \"Projekt 'Alpha' Review\"" in text


def test_export_writes_dated_titled_filename(tmp_path):
    path = export_note(transcript(), notes(), tmp_path, created_at=CREATED)
    assert path.name == "2026-07-10 – Quartalsplanung.md"
    assert path.read_text(encoding="utf-8").startswith("---\n")


def test_export_slug_strips_link_and_path_characters(tmp_path):
    weird = notes(title='Sync: [[Q3]] #plan | a/b\\c *draft?* "x" <y>')
    path = export_note(transcript(), weird, tmp_path, created_at=CREATED)
    assert path.name == "2026-07-10 – Sync Q3 plan a b c draft x y.md"


def test_export_keeps_umlauts_and_emoji(tmp_path):
    path = export_note(transcript(), notes(title="Größenplanung 🚀"), tmp_path, created_at=CREATED)
    assert path.name == "2026-07-10 – Größenplanung 🚀.md"


def test_export_caps_slug_length(tmp_path):
    path = export_note(transcript(), notes(title="x" * 300), tmp_path, created_at=CREATED)
    assert len(path.stem) <= len("2026-07-10 – ") + 80


def test_export_blank_slug_falls_back_to_meeting(tmp_path):
    path = export_note(transcript(), notes(title="###"), tmp_path, created_at=CREATED)
    assert path.name == "2026-07-10 – Meeting.md"


def test_export_collision_gets_numbered_suffix(tmp_path):
    first = export_note(transcript(), notes(), tmp_path, created_at=CREATED)
    second = export_note(transcript(), notes(), tmp_path, created_at=CREATED)
    third = export_note(transcript(), notes(), tmp_path, created_at=CREATED)
    assert first.name == "2026-07-10 – Quartalsplanung.md"
    assert second.name == "2026-07-10 – Quartalsplanung (2).md"
    assert third.name == "2026-07-10 – Quartalsplanung (3).md"


def test_export_into_dir_with_spaces_created_on_demand(tmp_path):
    vault = tmp_path / "iCloud~md~obsidian" / "Documents" / "Arbeit Meetings"
    path = export_note(transcript(), notes(), vault, created_at=CREATED)
    assert path.parent == vault
    assert path.exists()
