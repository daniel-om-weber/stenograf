"""Meeting archive: a managed library of finalized transcripts.

Phase 4 turns ``steno start`` into a tool with memory. Every run writes its
transcript into a managed directory whose name *is* a stable meeting id
(``meeting-YYYYMMDD-HHMMSS`` + a collision suffix), and a maintained JSON index
records lightweight metadata for each — so the web UI's "meeting archive" view
lists meetings without re-scanning and re-parsing every transcript on disk
(PLAN.md §5 Stage B1).

Design mirrors :class:`~stenograf.profiles.ProfileStore`:

- The archive lives in the platform **data** dir (``data_dir()/meetings/``),
  distinct from the re-downloadable model cache — a transcript library is
  precious user data.
- The index (``meetings/index.json``) is written atomically (temp + replace), so
  a crash mid-save never corrupts the library.
- The index is *maintained* (updated on every add/remove), not derived by
  scanning — but :meth:`MeetingArchive.reconcile` self-heals it against the
  actual directories (dropping records whose dir vanished, adopting orphan
  meeting dirs written while the index was unavailable).

The **in-RAM-only privacy guarantee is preserved**: a record references audio
only when ``--record-audio`` actually wrote a WAV, which :meth:`MeetingRecord.
has_audio` gates (audio playback and archived re-diarize check it — PLAN.md §5
Stage B4). A record with no audio still supports text click-to-jump, because the
word timestamps live in the transcript JSON.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from stenograf.config import Language
from stenograf.profiles import data_dir
from stenograf.transcript import Transcript, UnsupportedTranscriptVersion

_INDEX_VERSION = 1

TRANSCRIPT_STEM = "transcript"
"""Basename (without extension) of the managed transcript files in a meeting dir:
``transcript.json`` / ``transcript.md`` / ``transcript.srt`` / ``transcript.vtt``."""

TRANSCRIPT_FORMATS = ("md", "json", "srt", "vtt")
"""Transcript file extensions the archive recognizes when summarizing a dir."""

AUDIO_NAME = "audio.wav"
"""Managed name of the opt-in ``--record-audio`` WAV inside a meeting dir."""

_ID_TIMESTAMP = re.compile(r"^meeting-(\d{8})-(\d{6})")


@dataclass
class MeetingRecord:
    """Lightweight index metadata for one archived meeting.

    Denormalized into the index so the archive can be listed without opening each
    transcript. ``dir`` is where the transcript files actually live — normally the
    managed ``meetings/<id>/`` but an explicit ``--out`` can point elsewhere while
    still registering here. Plain (unfrozen) dataclass: records are value-compared
    for round-trip tests but deliberately unhashable (the ``speakers`` dict would
    make a frozen hash raise — see :class:`~stenograf.profiles.SpeakerProfile`)."""

    id: str
    title: str | None
    created_at: str
    """ISO-8601 creation timestamp (string, so the index is plain JSON)."""
    duration_s: float
    language: Language | None
    speakers: dict[str, int | None] = field(default_factory=dict)
    """Per-channel resolved speaker count (``mic``/``system``, or ``audio`` for a
    file transcribe); ``None`` for a channel whose count was left at a default."""
    formats: tuple[str, ...] = ()
    dir: Path = field(default=Path())
    audio_path: Path | None = None

    def has_audio(self) -> bool:
        """True only when a recorded WAV actually exists on disk.

        The single predicate gating archived audio *playback* and archived
        *re-diarize* (PLAN.md §5 Stage B4) — both contradict the in-memory-only
        guarantee unless ``--record-audio`` was used, and a referenced file can
        also have been deleted since."""
        return self.audio_path is not None and self.audio_path.exists()

    def _to_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "duration_s": self.duration_s,
            "language": self.language.value if self.language else None,
            "speakers": dict(self.speakers),
            "formats": list(self.formats),
            "dir": str(self.dir),
            "audio_path": str(self.audio_path) if self.audio_path is not None else None,
        }

    @classmethod
    def _from_json(cls, data: dict) -> MeetingRecord:
        audio = data.get("audio_path")
        language = data.get("language")
        return cls(
            id=data["id"],
            title=data.get("title"),
            created_at=data.get("created_at", ""),
            duration_s=data.get("duration_s", 0.0),
            language=Language(language) if language is not None else None,
            speakers=dict(data.get("speakers", {})),
            formats=tuple(data.get("formats", ())),
            dir=Path(data["dir"]),
            audio_path=Path(audio) if audio is not None else None,
        )


class MeetingArchive:
    """A managed, index-backed library of :class:`MeetingRecord` s.

    Load with :meth:`load` (a missing index is an empty archive), mutate with
    :meth:`add`/:meth:`remove` (each persists atomically), read a transcript back
    through the A1 loader with :meth:`load_transcript`, and self-heal the index
    against the directory tree with :meth:`reconcile`.
    """

    def __init__(
        self, root: Path | None = None, records: list[MeetingRecord] | None = None
    ) -> None:
        self.root = Path(root) if root is not None else meetings_dir()
        # Keyed by id (== dir name) so get/remove are O(1) and ids stay unique;
        # dict preserves insertion order for a stable listing.
        self._records: dict[str, MeetingRecord] = {r.id: r for r in (records or [])}

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    # ---- persistence ------------------------------------------------------

    @classmethod
    def load(cls, root: Path | None = None) -> MeetingArchive:
        root = Path(root) if root is not None else meetings_dir()
        index = root / "index.json"
        if not index.exists():
            return cls(root)
        data = json.loads(index.read_text(encoding="utf-8"))
        records = [MeetingRecord._from_json(r) for r in data.get("meetings", [])]
        return cls(root, records)

    def save(self) -> None:
        """Write the index to ``index_path`` atomically (temp file + replace)."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _INDEX_VERSION, "meetings": [r._to_json() for r in self._records.values()]},
            ensure_ascii=False,
            indent=2,
        )
        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, suffix=".part", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.index_path)

    # ---- reads ------------------------------------------------------------

    def records(self) -> list[MeetingRecord]:
        """Every record, in insertion order (callers sort as they like)."""
        return list(self._records.values())

    def get(self, meeting_id: str) -> MeetingRecord | None:
        return self._records.get(meeting_id)

    def meeting_dir(self, meeting_id: str) -> Path:
        """The managed directory for ``meeting_id`` (``root/<id>``); the default
        ``steno start`` output location for a new meeting."""
        return self.root / meeting_id

    def load_transcript(self, meeting_id: str) -> Transcript:
        """Read a meeting's transcript back through :meth:`Transcript.from_json`."""
        record = self._records.get(meeting_id)
        if record is None:
            raise KeyError(meeting_id)
        text = (record.dir / f"{TRANSCRIPT_STEM}.json").read_text(encoding="utf-8")
        return Transcript.from_json(text)

    # ---- writes -----------------------------------------------------------

    def allocate_id(self, created_at: datetime) -> str:
        """Mint a stable, unique meeting id for ``created_at``.

        Base form ``meeting-YYYYMMDD-HHMMSS``; on collision (same second, or a dir
        already present) append ``-2``, ``-3``, … so the id — and therefore the
        managed dir name — is always free."""
        base = f"meeting-{created_at:%Y%m%d-%H%M%S}"
        candidate = base
        suffix = 2
        while candidate in self._records or (self.root / candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def add(self, record: MeetingRecord) -> None:
        """Insert or replace a record by id and persist the index."""
        self._records[record.id] = record
        self.save()

    def remove(self, meeting_id: str) -> bool:
        """Drop a record from the index (leaves its files alone). Persists if changed."""
        if self._records.pop(meeting_id, None) is None:
            return False
        self.save()
        return True

    def reconcile(self) -> None:
        """Self-heal the index against the actual meeting directories.

        Drops records whose ``dir`` has vanished, and adopts orphan managed dirs
        (a ``meetings/<id>/`` holding a ``transcript.json`` that the index doesn't
        know about — e.g. written while the index was unavailable). External
        ``--out`` dirs are never scanned for adoption; only the managed root is."""
        for meeting_id, record in list(self._records.items()):
            if not record.dir.exists():
                del self._records[meeting_id]
        if self.root.exists():
            for child in sorted(self.root.iterdir()):
                if not child.is_dir() or child.name in self._records:
                    continue
                adopted = _record_from_dir(child)
                if adopted is not None:
                    self._records[adopted.id] = adopted
        self.save()


def _record_from_dir(directory: Path) -> MeetingRecord | None:
    """Reconstruct a record from a managed meeting dir, or ``None`` if it holds no
    readable transcript (a half-written or unrelated directory)."""
    transcript_json = directory / f"{TRANSCRIPT_STEM}.json"
    if not transcript_json.exists():
        return None
    try:
        transcript = Transcript.from_json(transcript_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnsupportedTranscriptVersion, KeyError):
        return None
    speakers: dict[str, int | None] = {}
    if transcript.parameters is not None:
        speakers = {ch: rv.value for ch, rv in transcript.parameters.speakers.items()}  # type: ignore[misc]
    audio = directory / AUDIO_NAME
    formats = tuple(
        ext for ext in TRANSCRIPT_FORMATS if (directory / f"{TRANSCRIPT_STEM}.{ext}").exists()
    )
    return MeetingRecord(
        id=directory.name,
        title=transcript.profile.title,
        created_at=_created_at_from_id(directory.name),
        duration_s=max((e.end for e in transcript.entries), default=0.0),
        language=transcript.language,
        speakers=speakers,
        formats=formats,
        dir=directory,
        audio_path=audio if audio.exists() else None,
    )


def _created_at_from_id(meeting_id: str) -> str:
    """Recover an ISO timestamp from a ``meeting-YYYYMMDD-HHMMSS`` id (the id
    encodes it), or ``""`` for a non-standard dir name adopted during reconcile."""
    match = _ID_TIMESTAMP.match(meeting_id)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            pass
    return ""


def meetings_dir() -> Path:
    """Managed archive root: ``data_dir()/meetings`` (honors ``$STENOGRAF_DATA``)."""
    return data_dir() / "meetings"
