"""User settings — ``data_dir()/settings.toml``.

Machine-specific configuration and standing preferences that must NOT live in a
:class:`MeetingProfile` (profiles serialize into every transcript; a local
command line or vault path would leak into shared files). Every key is
optional; a missing file is simply all defaults. The full schema::

    [transcript]
    formats = ["md", "json", "txt"]     # default --format list

    [vocab]                             # standing vocabulary for every run
    glossary_file = "~/steno/glossary.txt"  # one term per line, # comments
    attendees = ["Ada Lovelace"]        # names corrected like glossary terms
    glossary_threshold = 0.82           # similarity 0-1 to correct a term

    [archive]
    enabled = true                      # false = flat files, as --no-archive
    out_dir = "~/Transcripts"           # where flat files go when NOT archiving
                                        # (ignored while the archive is on — a
                                        # managed meeting gets its own dir)

    [speakers]
    reid_threshold = 0.5                # cosine similarity 0-1 to match a
                                        # saved speaker profile
    profile_store = "~/steno/profiles.json"  # re-ID store location override

    [asr]
    backend = "parakeet"                # ASR backend for finalize + live

    [notes]
    backend = "command"                 # "mlx" (default on Apple Silicon),
                                        # "ollama" (default elsewhere), "command"
    model = "claude-opus-4-8"           # HF repo id (mlx) / Ollama tag /
                                        # provenance label (command)
    command = ["claude", "-p", "Summarize the meeting transcript on stdin."]
    timeout_s = 600
    instructions = "~/notes-style.md"   # appended to the built-in system prompt
    thinking = false                    # mlx backend: skip the model's reasoning
                                        # pass — faster, less careful (default true)

    [notes.export]
    dir = "~/Obsidian/Meetings"         # combined-note export target (unset = off)

Precedence everywhere a value is consumed: CLI flag > environment variable
(``STENOGRAF_ASR_BACKEND``, ``STENOGRAF_NOTES_BACKEND``, ``STENOGRAF_NOTES_MODEL``,
``OLLAMA_HOST``) > this file > built-in default. Almost every value is a
*default the flag replaces*; the one exception is ``[vocab]``, whose glossary
terms and attendee names *merge* with per-run ``--glossary``/``--attendee``
values — the configured vocabulary is a standing baseline, not an either/or.

Unknown tables and keys are rejected: a typo in a hand-edited file must fail
loudly (naming the file and key), never silently configure nothing. All
validation happens at load time so ``steno doctor`` — and every command's
startup — vets the whole file before any real work begins.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from stenograf.profiles import data_dir


class SettingsError(Exception):
    """settings.toml exists but cannot be used; the message names the file."""


@dataclass(frozen=True)
class TranscriptSettings:
    formats: tuple[str, ...] = ()
    """Default output formats; empty = the built-in default (md, json, txt)."""


@dataclass(frozen=True)
class VocabSettings:
    glossary_file: Path | None = None
    attendees: tuple[str, ...] = ()
    glossary_threshold: float | None = None


@dataclass(frozen=True)
class ArchiveSettings:
    enabled: bool | None = None
    """``False`` makes ``--no-archive`` the default; ``None`` = archive on."""
    out_dir: Path | None = None
    """Default ``--out`` for the flat (non-archived) layout. Deliberately not
    applied while the archive is on: there ``--out`` names one meeting's own
    dir, so a standing value would pile every meeting into the same files."""


@dataclass(frozen=True)
class SpeakerSettings:
    reid_threshold: float | None = None
    profile_store: Path | None = None


@dataclass(frozen=True)
class AsrSettings:
    backend: str | None = None


@dataclass(frozen=True)
class NotesSettings:
    backend: str | None = None
    model: str | None = None
    command: tuple[str, ...] = ()
    timeout_s: float | None = None
    instructions: Path | None = None
    ollama_url: str | None = None
    export_dir: Path | None = None
    max_input_chars: int | None = None
    """Single-completion transcript budget override; ``None`` = the backend's
    own default (local models get a smaller one than hosted frontier models)."""
    thinking: bool | None = None
    """Reasoning mode for local models that have one (Qwen3 via the mlx
    backend); ``None`` = the backend's default."""


@dataclass(frozen=True)
class Settings:
    transcript: TranscriptSettings = field(default_factory=TranscriptSettings)
    vocab: VocabSettings = field(default_factory=VocabSettings)
    archive: ArchiveSettings = field(default_factory=ArchiveSettings)
    speakers: SpeakerSettings = field(default_factory=SpeakerSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    notes: NotesSettings = field(default_factory=NotesSettings)


def settings_path() -> Path:
    return data_dir() / "settings.toml"


def load_settings(path: Path | None = None) -> Settings:
    """Read ``settings.toml`` (or ``path``); a missing file is all defaults.

    Malformed TOML, a wrong-typed or out-of-range value, an unknown backend or
    format, or an unrecognized table/key raise one :class:`SettingsError`
    naming the file — settings problems must never surface as a traceback deep
    inside a meeting run."""
    path = path or settings_path()
    if not path.exists():
        return Settings()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from exc
    try:
        top = _Table("", data)
        settings = Settings(
            transcript=_transcript_from_table(top.table("transcript")),
            vocab=_vocab_from_table(top.table("vocab")),
            archive=_archive_from_table(top.table("archive")),
            speakers=_speakers_from_table(top.table("speakers")),
            asr=_asr_from_table(top.table("asr")),
            notes=_notes_from_table(top.table("notes")),
        )
        top.reject_unknown()
        return settings
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"invalid settings in {path}: {exc}") from exc


class _Table:
    """Typed reads from one TOML table; every error names ``table.key``.

    Each getter records the key it consumed so :meth:`reject_unknown` can name
    anything left over — the typo guard. ``name=""`` is the document root."""

    def __init__(self, name: str, data: object) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"[{name}] must be a table")
        self._name = name
        self._data: dict = data
        self._read: set[str] = set()

    def _get(self, key: str) -> object:
        self._read.add(key)
        return self._data.get(key)

    def _err(self, key: str, problem: str) -> None:
        label = f"{self._name}.{key}" if self._name else key
        raise ValueError(f"{label} {problem}")

    def str_(self, key: str) -> str | None:
        value = self._get(key)
        if value is not None and not isinstance(value, str):
            self._err(key, "must be a string")
        return value

    def path(self, key: str) -> Path | None:
        value = self.str_(key)
        return Path(value).expanduser() if value is not None else None

    def bool_(self, key: str) -> bool | None:
        value = self._get(key)
        if value is not None and not isinstance(value, bool):
            self._err(key, "must be true or false")
        return value

    def number(self, key: str, lo: float | None = None, hi: float | None = None) -> float | None:
        value = self._get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            self._err(key, "must be a number")
        if lo is not None and not lo <= value <= hi:
            self._err(key, f"must be between {lo:g} and {hi:g}")
        return float(value)

    def pos_int(self, key: str) -> int | None:
        value = self._get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            self._err(key, "must be a positive integer")
        return value

    def str_list(self, key: str) -> tuple[str, ...]:
        value = self._get(key)
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            self._err(key, "must be an array of strings")
        return tuple(value)

    def table(self, key: str) -> dict:
        """The nested table under ``key`` (``{}`` if absent), for a child ``_Table``."""
        value = self._get(key)
        return value if value is not None else {}

    def reject_unknown(self) -> None:
        unknown = sorted(set(self._data) - self._read)
        if unknown:
            where = f" in [{self._name}]" if self._name else ""
            raise ValueError(f"unknown setting(s){where}: {', '.join(unknown)}")


def _transcript_from_table(data: dict) -> TranscriptSettings:
    t = _Table("transcript", data)
    formats = t.str_list("formats")
    from stenograf.transcript import FORMATS

    for name in formats:
        if name not in FORMATS:
            raise ValueError(
                f"transcript.formats: unknown format {name!r} (choose from {', '.join(FORMATS)})"
            )
    t.reject_unknown()
    return TranscriptSettings(formats=formats)


def _vocab_from_table(data: dict) -> VocabSettings:
    t = _Table("vocab", data)
    settings = VocabSettings(
        glossary_file=t.path("glossary_file"),
        attendees=t.str_list("attendees"),
        glossary_threshold=t.number("glossary_threshold", 0, 1),
    )
    t.reject_unknown()
    return settings


def _archive_from_table(data: dict) -> ArchiveSettings:
    t = _Table("archive", data)
    settings = ArchiveSettings(enabled=t.bool_("enabled"), out_dir=t.path("out_dir"))
    t.reject_unknown()
    return settings


def _speakers_from_table(data: dict) -> SpeakerSettings:
    t = _Table("speakers", data)
    settings = SpeakerSettings(
        reid_threshold=t.number("reid_threshold", 0, 1),
        profile_store=t.path("profile_store"),
    )
    t.reject_unknown()
    return settings


def _asr_from_table(data: dict) -> AsrSettings:
    t = _Table("asr", data)
    backend = t.str_("backend")
    if backend is not None:
        from stenograf.asr.registry import available_backends

        if backend not in available_backends():
            raise ValueError(
                f"unknown ASR backend {backend!r} (choose from {', '.join(available_backends())})"
            )
    t.reject_unknown()
    return AsrSettings(backend=backend)


def _notes_from_table(data: dict) -> NotesSettings:
    t = _Table("notes", data)
    backend = t.str_("backend")
    if backend is not None:
        from stenograf.notes.backend import available_backends

        if backend not in available_backends():
            raise ValueError(
                f"unknown notes backend {backend!r} (choose from {', '.join(available_backends())})"
            )
    export = _Table("notes.export", t.table("export"))
    settings = NotesSettings(
        backend=backend,
        model=t.str_("model"),
        command=t.str_list("command"),
        timeout_s=t.number("timeout_s"),
        instructions=t.path("instructions"),
        ollama_url=t.str_("ollama_url"),
        export_dir=export.path("dir"),
        max_input_chars=t.pos_int("max_input_chars"),
        thinking=t.bool_("thinking"),
    )
    export.reject_unknown()
    t.reject_unknown()
    return settings
