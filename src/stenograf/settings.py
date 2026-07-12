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

    [output]
    dir = "~/Documents/Meetings"        # the output home: every run writes its
                                        # own meeting-YYYYMMDD-HHMMSS/ folder
                                        # here (--out bypasses it for one run)

    [speakers]
    diarization = false                 # skip speaker diarization by default:
                                        # each channel is one speaker and the
                                        # diarizer model is never loaded (for
                                        # machines where it takes minutes). A
                                        # per-run --diarization flag or an
                                        # explicit speaker count above 1 still
                                        # turns it on.
    reid_threshold = 0.5                # cosine similarity 0-1 to match a
                                        # saved speaker profile
    profile_store = "~/steno/profiles.json"  # re-ID store location override

    [asr]
    backend = "parakeet"                # ASR backend for finalize + live
    provider = "cpu"                    # ONNX Runtime execution provider for the
                                        # parakeet-onnx backend: cpu (default) |
                                        # dml (any DX12 GPU, Windows) | cuda |
                                        # auto; falls back to cpu with a warning
                                        # if the provider can't run the model

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
    ollama_url = "http://localhost:11434"  # ollama backend: server base URL
                                           # (default from OLLAMA_HOST, else local)

    [notes.export]
    dir = "~/Obsidian/Meetings"         # combined-note export target (unset = off)

Precedence everywhere a value is consumed: CLI flag > environment variable
(``STENOGRAF_ASR_BACKEND``, ``STENOGRAF_ASR_PROVIDER``, ``STENOGRAF_NOTES_BACKEND``,
``STENOGRAF_NOTES_MODEL``, ``OLLAMA_HOST``) > this file > built-in default. Almost every value is a
*default the flag replaces*; the one exception is ``[vocab]``, whose glossary
terms and attendee names *merge* with per-run ``--glossary``/``--attendee``
values — the configured vocabulary is a standing baseline, not an either/or.

Unknown tables and keys are rejected: a typo in a hand-edited file must fail
loudly (naming the file and key), never silently configure nothing. All
validation happens at load time so ``steno doctor`` — and every command's
startup — vets the whole file before any real work begins.

Portability: this module is pure stdlib (``tomllib``/``pathlib``) and works
unchanged on Linux and Windows, as do ``steno settings show``/``edit``
(``click.edit`` handles ``$EDITOR`` vs. notepad; the atomic write uses
``os.replace``, atomic on both POSIX and Windows). One deliberate limit:
backend-name validation is registry-level, not platform-aware — ``backend =
"mlx"`` validates on any platform because the spec is registered everywhere;
whether the backend can *run* (mlx-lm installed, Ollama reachable) is checked
at use, which keeps settings validation independent of what's installed. The
file's location comes from :func:`stenograf.profiles.data_dir` (``%APPDATA%``
on Windows).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from stenograf.profiles import data_dir


class SettingsError(Exception):
    """settings.toml exists but cannot be used; the message names the file."""


SETTINGS_TEMPLATE = """\
# stenograf settings — every key is optional; a missing key keeps its built-in
# default. Uncomment what you want to change, then save; the file is validated
# on the way out. `steno settings show` prints the effective configuration and
# where each value comes from. A CLI flag always beats this file.

[transcript]
# formats = ["md", "json", "txt"]          # any of: md, json, txt, srt, vtt

[vocab]                                    # standing vocabulary — merged with
# glossary_file = "~/steno/glossary.txt"   # per-run --glossary/--attendee flags;
# attendees = ["Anja Müller"]              # file terms are one per line
# glossary_threshold = 0.82                # similarity 0-1 to correct a term

[output]
# dir = "~/Documents/Meetings"             # where meeting folders are created

[speakers]
# diarization = true                       # false = skip speaker separation (fast;
#                                          # a per-run flag or count still overrides)
# reid_threshold = 0.5                     # voice-match strictness 0-1
# profile_store = "~/steno/profiles.json"  # re-ID voiceprint store location

[asr]
# backend = "parakeet"
# provider = "cpu"                         # cpu | dml (Windows GPU) | cuda | auto

[notes]
# backend = "mlx"                          # mlx | ollama | command
# model = "Qwen/Qwen3-8B-MLX-4bit"         # HF repo id (mlx) / Ollama tag
# command = ["claude", "-p"]               # argv for backend = "command"
# timeout_s = 600                          # command backend time limit
# instructions = "~/notes-style.md"        # appended to the system prompt
# thinking = true                          # mlx: run the model's reasoning pass
# ollama_url = "http://localhost:11434"    # ollama server base URL

[notes.export]
# dir = "~/Obsidian/Meetings"              # also write one combined note here
"""
"""The commented-out starter file ``steno settings edit`` creates on first run.

Every table header is live (an empty table is all defaults) and every key is
commented — so the pristine template loads as exactly ``Settings()``, which the
tests pin. Keep it in step with the schema above."""


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
class OutputSettings:
    dir: Path | None = None
    """The output home meeting folders are created in; ``None`` = the default
    (``~/Documents/Meetings``, :func:`stenograf.output.default_output_home`).
    Not one meeting's dir — ``--out`` is that — but the folder of folders."""


@dataclass(frozen=True)
class SpeakerSettings:
    diarization: bool | None = None
    """``False`` skips speaker diarization by default: each channel is
    attributed to one speaker and the diarizer model is never loaded — for
    machines where diarization costs minutes. A per-run ``--diarization`` flag
    or an explicit speaker count above 1 still turns it on; ``None`` = on."""
    reid_threshold: float | None = None
    profile_store: Path | None = None


@dataclass(frozen=True)
class AsrSettings:
    backend: str | None = None
    provider: str | None = None
    """ONNX Runtime execution provider for the ORT-backed backend; ``None`` =
    CPU. Backends with their own runtime (MLX) ignore it."""


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
    output: OutputSettings = field(default_factory=OutputSettings)
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
    if "archive" in data:
        # The Stage C de-scope renamed the table; a stale file must say so, not
        # just "unknown setting: archive".
        raise SettingsError(
            f"invalid settings in {path}: [archive] was renamed to [output] — meetings "
            "now always get their own folder in the output home (set [output] dir; "
            "enabled/out_dir are gone)"
        )
    try:
        top = _Table("", data)
        settings = Settings(
            transcript=_transcript_from_table(top.table("transcript")),
            vocab=_vocab_from_table(top.table("vocab")),
            output=_output_from_table(top.table("output")),
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

    def _err(self, key: str, problem: str) -> NoReturn:
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
        if (lo is None) != (hi is None):
            raise TypeError("number() bounds must be given together")
        value = self._get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            self._err(key, "must be a number")
        if lo is not None and hi is not None and not lo <= value <= hi:
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
        if value is None:
            return {}
        if not isinstance(value, dict):
            self._err(key, "must be a table")
        return value

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


def _output_from_table(data: dict) -> OutputSettings:
    t = _Table("output", data)
    settings = OutputSettings(dir=t.path("dir"))
    t.reject_unknown()
    return settings


def _speakers_from_table(data: dict) -> SpeakerSettings:
    t = _Table("speakers", data)
    settings = SpeakerSettings(
        diarization=t.bool_("diarization"),
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
    provider = t.str_("provider")
    if provider is not None:
        from stenograf.asr.providers import validate_provider_name

        validate_provider_name(provider)
    t.reject_unknown()
    return AsrSettings(backend=backend, provider=provider)


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
