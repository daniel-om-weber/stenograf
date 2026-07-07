from datetime import datetime
from pathlib import Path

from stenograf.archive import (
    AUDIO_NAME,
    TRANSCRIPT_STEM,
    MeetingArchive,
    MeetingRecord,
    meetings_dir,
)
from stenograf.config import (
    Language,
    MeetingProfile,
    Provenance,
    ResolvedParameters,
    ResolvedValue,
)
from stenograf.transcript import Transcript, TranscriptEntry


def _record(meeting_id: str, root: Path, **over) -> MeetingRecord:
    defaults = dict(
        id=meeting_id,
        title="Weekly sync",
        created_at="2026-07-07T19:30:00",
        duration_s=12.5,
        language=Language.GERMAN,
        speakers={"mic": 1, "system": 2},
        formats=("md", "json"),
        dir=root / meeting_id,
        audio_path=None,
    )
    defaults.update(over)
    return MeetingRecord(**defaults)


def _transcript(title: str | None = "Weekly sync") -> Transcript:
    return Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(
            language=Language.GERMAN, local_speakers=1, remote_speakers=2, title=title
        ),
        entries=[
            TranscriptEntry(speaker="Local-1", text="hallo", start=0.5, end=2.0),
            TranscriptEntry(speaker="Remote-1", text="moin", start=2.0, end=8.25),
        ],
        parameters=ResolvedParameters(
            language=ResolvedValue(Language.GERMAN, Provenance.EXPLICIT),
            speakers={
                "mic": ResolvedValue(1, Provenance.EXPLICIT),
                "system": ResolvedValue(2, Provenance.DETECTED),
            },
        ),
    )


def _write_meeting_dir(root: Path, meeting_id: str, *, formats=("json", "md"), audio=False):
    """Write a managed meeting dir with a real transcript.json (+ sibling formats)."""
    directory = root / meeting_id
    directory.mkdir(parents=True)
    transcript = _transcript()
    for ext in formats:
        payload = transcript.to_json() if ext == "json" else transcript.to_markdown()
        (directory / f"{TRANSCRIPT_STEM}.{ext}").write_text(payload, encoding="utf-8")
    if audio:
        (directory / AUDIO_NAME).write_bytes(b"RIFF....WAVE")
    return directory


def test_add_list_get_remove_round_trip(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    a = _record("meeting-20260707-193000", tmp_path)
    b = _record("meeting-20260707-194500", tmp_path, title="Standup")
    archive.add(a)
    archive.add(b)

    assert {r.id for r in archive.records()} == {a.id, b.id}
    assert archive.get(a.id) == a
    assert archive.get("nope") is None

    assert archive.remove(a.id) is True
    assert archive.remove(a.id) is False  # already gone
    assert [r.id for r in archive.records()] == [b.id]


def test_index_persists_and_reloads_equal(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    rec = _record(
        "meeting-20260707-193000",
        tmp_path,
        audio_path=tmp_path / "meeting-20260707-193000" / AUDIO_NAME,
    )
    archive.add(rec)

    assert archive.index_path.exists()
    reloaded = MeetingArchive.load(root=tmp_path)
    assert reloaded.get(rec.id) == rec
    # Atomic write leaves no temp turds behind.
    assert not list(tmp_path.glob("*.part"))


def test_missing_index_is_an_empty_archive(tmp_path):
    assert MeetingArchive.load(root=tmp_path / "nothing").records() == []


def test_allocate_id_suffixes_on_collision(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    when = datetime(2026, 7, 7, 19, 30, 0)
    first = archive.allocate_id(when)
    assert first == "meeting-20260707-193000"

    archive.add(_record(first, tmp_path))
    # Same second → suffixed away from the taken id (both in-index and on-disk).
    assert archive.allocate_id(when) == "meeting-20260707-193000-2"

    (tmp_path / "meeting-20260707-193000-2").mkdir()
    assert archive.allocate_id(when) == "meeting-20260707-193000-3"


def test_load_transcript_round_trips_via_a1(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    meeting_id = "meeting-20260707-193000"
    _write_meeting_dir(tmp_path, meeting_id)
    archive.add(_record(meeting_id, tmp_path))

    loaded = archive.load_transcript(meeting_id)
    assert loaded == _transcript()  # faithful reload through Transcript.from_json


def test_reconcile_drops_records_whose_dir_vanished(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    _write_meeting_dir(tmp_path, "meeting-20260707-193000")
    archive.add(_record("meeting-20260707-193000", tmp_path))
    archive.add(_record("meeting-20260707-999999", tmp_path))  # dir never created

    archive.reconcile()
    assert {r.id for r in archive.records()} == {"meeting-20260707-193000"}


def test_reconcile_adopts_orphan_dir_with_correct_metadata(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    _write_meeting_dir(
        tmp_path, "meeting-20260707-193000", formats=("json", "md", "srt"), audio=True
    )

    archive.reconcile()

    rec = archive.get("meeting-20260707-193000")
    assert rec is not None
    assert rec.title == "Weekly sync"
    assert rec.language is Language.GERMAN
    assert rec.speakers == {"mic": 1, "system": 2}
    assert rec.formats == ("md", "json", "srt")  # ordered by TRANSCRIPT_FORMATS
    assert rec.duration_s == 8.25  # last entry end
    assert rec.created_at == "2026-07-07T19:30:00"  # recovered from the id
    assert rec.has_audio()  # audio.wav present


def test_reconcile_skips_dirs_without_a_transcript(tmp_path):
    archive = MeetingArchive(root=tmp_path)
    (tmp_path / "half-written").mkdir()
    (tmp_path / "half-written" / "notes.txt").write_text("wip", encoding="utf-8")

    archive.reconcile()
    assert archive.records() == []


def test_has_audio_requires_the_file_to_exist(tmp_path):
    audio = tmp_path / "meeting-20260707-193000" / AUDIO_NAME
    rec = _record("meeting-20260707-193000", tmp_path, audio_path=audio)
    assert rec.has_audio() is False  # referenced but not written
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF")
    assert rec.has_audio() is True
    assert _record("x", tmp_path, audio_path=None).has_audio() is False


def test_meetings_dir_uses_data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    assert meetings_dir() == tmp_path / "meetings"
