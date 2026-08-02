"""User settings — ``data_dir()/settings.toml``.

Machine-specific configuration and standing preferences that must NOT live in a
:class:`MeetingProfile` (profiles serialize into every transcript; a local
command line or vault path would leak into shared files). Every key is
optional; a missing file is simply all defaults. The schema lives in ONE
place: :data:`SETTINGS_TEMPLATE`, the commented starter file whose commented
values *are* the defaults — ``steno settings edit`` writes it and
``steno settings show`` prints the effective values with their sources.

Preset overlay rules: a key the preset sets wins over ``[notes]``; unset keys
fall through. ``[meetings.*.vocab]`` *merges* (a preset adds vocabulary, it
never removes the standing baseline); ``glossary_threshold`` is a scalar and
replaces. On a path key, ``""`` explicitly switches the standing value OFF —
``[meetings.x.notes.export] dir = ""`` keeps a confidential meeting out of the
vault, ``template = ""``/``instructions = ""`` return to the built-ins. A
preset that names a ``backend`` without a ``model`` drops the standing model
(it was written for the standing backend — ``[notes] model`` must never ride
into another backend as a bogus repo id); a preset ``model`` without a
``backend`` applies to the standing backend. Presets are personal: they live
only in this machine-local file, never in a shared checkout, because
``[notes] command`` is an argv executed unattended after every meeting.

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
file's location comes from :func:`stenograf.paths.data_dir` (``%APPDATA%``
on Windows).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import NoReturn

from stenograf.paths import data_dir


class SettingsError(Exception):
    """settings.toml exists but cannot be used; the message names the file."""


class UnknownPresetError(SettingsError):
    """``--preset`` (or a setup form) named a ``[meetings.<name>]`` section that
    does not exist.

    Its own type — still a :class:`SettingsError`, so every existing catch
    holds — because the remedy differs: not "repair the file" but "pick one of
    the presets it defines", which the CLI reports as a usage error."""


SETTINGS_TEMPLATE = """\
# stenograf settings — every key is optional; a missing key keeps its built-in
# default. The commented values below *are* those defaults (a key that has no
# default shows an example instead), so a line you have not uncommented tells you
# what the tool is already doing. Uncomment what you want to change, then save;
# the file is validated on the way out. `steno settings show` prints the effective
# configuration and where each value comes from. A CLI flag always beats this file.

[transcript]
# formats = ["md", "json", "txt"]          # any of: md, json, txt, srt, vtt

[vocab]                                    # standing vocabulary — merged with
# glossary_file = "~/steno/glossary.txt"   # per-run --glossary/--attendee flags;
# attendees = ["Anja Müller"]              # file terms are one per line
# glossary_threshold = 0.95                # similarity 0-1 to correct a term;
#                                          # lower rewrites correct words (measured)
#
# Write a term the way it is SPOKEN, in the form it appears in the sentence.
# Casing is load-bearing ("Kubernetes", not "kubernetes"), and when the model
# glues a term to its neighbour ("Prometheusalord"), only the compound spelling
# reaches it: list "Prometheus-Alert", not "Alert" on its own.

[output]
# dir = "~/Documents/Meetings"             # where meeting folders are created;
#                                          # the default is Meetings/ in your
#                                          # documents folder, which is localised
#                                          # on Linux (~/Dokumente/Meetings).
#                                          # `steno settings show` prints it
# record_audio = false                     # true = keep audio.wav (raw capture)
#                                          # in every meeting folder, like a bare
#                                          # --record-audio

[speakers]
# diarization = false                      # true = separate speakers within a channel
#                                          # (a per-run --diarization flag or a speaker
#                                          # count above 1 also turns it on)
# reid_threshold = 0.5                     # voice-match strictness 0-1
# profile_store = "~/steno/profiles.json"  # re-ID voiceprint store location

[asr]
# backend = "parakeet"
# provider = "cpu"                         # cpu | dml (Windows GPU) | cuda | auto
# boost = 1.0                              # how hard decoding is steered toward the
#                                          # [vocab] terms; 0 = off. Above ~3 it
#                                          # starts rewriting words you did not list.

[notes]
# auto = false                             # true = generate notes after every meeting
#                                          # (--notes on the CLI or the launcher's
#                                          # switch asks for them per run instead)
# backend = "mlx"                          # mlx | ollama | command
# model = "Qwen/Qwen3-8B-MLX-4bit"         # HF repo id (mlx) / Ollama tag
# command = ["claude", "-p"]               # argv for backend = "command"
# timeout_s = 600                          # command backend time limit
# instructions = "~/notes-style.md"        # appended to the system prompt
# template = "~/steno/protokoll.md"        # protocol layout (markdown headings);
#                                          # unset = the built-in sections
# thinking = true                          # mlx: run the model's reasoning pass
# ollama_url = "http://localhost:11434"    # ollama server base URL

[notes.export]
# dir = "~/Obsidian/Meetings"              # also write one combined note here

# Meeting presets ("Besprechungsart") — everything a *kind* of meeting sets,
# selected per run with `--preset NAME` and listed by `steno presets`. Keys a
# preset leaves unset fall through to the tables above; [meetings.*.vocab]
# merges with the standing [vocab]; "" on a path key switches it off for this
# preset (e.g. export dir = "" keeps a confidential meeting out of the vault).
#
# [meetings.controlling]
# title    = "Controlling-Runde"
# language = "de"
# template = "~/steno/controlling.md"
# instructions = "~/steno/controlling-stil.md"
#
# [meetings.controlling.notes]
# backend = "command"
# command = ["claude", "-p"]
# timeout_s = 1800
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
    (``Meetings`` in the user's documents folder, localised on Linux —
    :func:`stenograf.output.default_output_home`).
    Not one meeting's dir — ``--out`` is that — but the folder of folders."""
    record_audio: bool | None = None
    """``True`` keeps the raw captured audio as each meeting folder's
    ``audio.wav``, exactly as a bare ``--record-audio`` does. Unset (``None``)
    or ``False`` = off, the built-in default: audio never touches disk. A
    ``--record-audio PATH`` still redirects a single run to another file."""


@dataclass(frozen=True)
class SpeakerSettings:
    diarization: bool | None = None
    """``True`` separates speakers within each channel. Unset (``None``) or
    ``False`` = off, the built-in default: each channel is attributed to one
    speaker and the diarizer model is never loaded — it costs minutes on some
    machines. A per-run ``--diarization`` flag or an explicit speaker count
    above 1 also turns it on."""
    reid_threshold: float | None = None
    profile_store: Path | None = None


@dataclass(frozen=True)
class AsrSettings:
    backend: str | None = None
    provider: str | None = None
    """ONNX Runtime execution provider for the ORT-backed backend; ``None`` =
    CPU. Backends with their own runtime (MLX) ignore it."""

    boost: float | None = None
    """How hard the decoder is steered toward ``[vocab]`` terms while it
    transcribes (``stenograf.asr.biasing``); ``None`` = the built-in default.

    A decoder knob, not vocabulary, hence ``[asr]`` and not ``[vocab]``: it prices
    the glossary the *other* tables supply. 0 disables biasing entirely and leaves
    the stock decode loop in place. Raising it past ~3 starts rewriting words that
    are not in the glossary at all — the failure mode NVIDIA warns about, and the
    reason this is a setting rather than a constant."""


@dataclass(frozen=True)
class NotesSettings:
    auto: bool | None = None
    """``True`` generates notes after every meeting. Unset (``None``) or
    ``False`` = off, the built-in default: notes are asked for per run — the
    ``--notes`` flag, or the launcher setup form's switch, which starts on when
    this is ``True``. An explicit ``--no-notes`` (or the switch, turned off)
    still skips them for one run."""
    backend: str | None = None
    model: str | None = None
    command: tuple[str, ...] = ()
    timeout_s: float | None = None
    instructions: Path | None = None
    template: Path | None = None
    """Markdown protocol template (its headings are the output's validated
    structure — :mod:`stenograf.notes.template`); ``None`` = the built-in."""
    ollama_url: str | None = None
    export_dir: Path | None = None
    max_input_chars: int | None = None
    """Single-completion transcript budget override; ``None`` = the backend's
    own default (local models get a smaller one than hosted frontier models)."""
    thinking: bool | None = None
    """Reasoning mode for local models that have one (Qwen3 via the mlx
    backend); ``None`` = the backend's default."""


@dataclass(frozen=True)
class MeetingPreset:
    """One ``[meetings.<name>]`` section — everything a *kind* of meeting sets.

    The notes overlay is sparse (:func:`apply_meeting_preset`); ``vocab``
    *merges* with the standing baseline at the run-config seams rather than
    replacing it; ``title``/``language`` are form defaults a CLI flag or typed
    value still beats. ``cleared`` holds the keys the preset explicitly
    switched OFF with ``""`` (``"template"``, ``"instructions"``,
    ``"export.dir"``) — the overlay's only way to *remove* a standing value,
    since "non-``None`` wins" can only ever add."""

    name: str
    title: str | None = None
    language: str | None = None
    template: Path | None = None
    instructions: Path | None = None
    notes: NotesSettings = field(default_factory=NotesSettings)
    vocab: VocabSettings = field(default_factory=VocabSettings)
    cleared: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Settings:
    transcript: TranscriptSettings = field(default_factory=TranscriptSettings)
    vocab: VocabSettings = field(default_factory=VocabSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    speakers: SpeakerSettings = field(default_factory=SpeakerSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    notes: NotesSettings = field(default_factory=NotesSettings)
    meetings: dict[str, MeetingPreset] = field(default_factory=dict)


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
            meetings=_presets_from_table(top.table("meetings")),
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
        if value == "":
            # Path("") silently normalizes to "." — never what a settings file
            # means. The empty string has exactly one defined use: switching a
            # standing path OFF inside a [meetings.*] preset, which is handled
            # before this reader sees the table.
            self._err(
                key, 'must not be "" (only a [meetings.*] preset uses "" to switch a key off)'
            )
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


def _vocab_from_table(data: dict, label: str = "vocab") -> VocabSettings:
    # ``label`` keeps error messages honest when the same reader parses a
    # preset's [meetings.<name>.vocab] — a typo must name the section it is in.
    t = _Table(label, data)
    settings = VocabSettings(
        glossary_file=t.path("glossary_file"),
        attendees=t.str_list("attendees"),
        glossary_threshold=t.number("glossary_threshold", 0, 1),
    )
    t.reject_unknown()
    return settings


def _output_from_table(data: dict) -> OutputSettings:
    t = _Table("output", data)
    settings = OutputSettings(dir=t.path("dir"), record_audio=t.bool_("record_audio"))
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
    boost = t.number("boost", 0, 10)
    t.reject_unknown()
    return AsrSettings(backend=backend, provider=provider, boost=boost)


def _notes_from_table(data: dict, label: str = "notes") -> NotesSettings:
    # ``label`` keeps error messages honest when the same reader parses a
    # preset's [meetings.<name>.notes] — a typo must name the section it is in.
    t = _Table(label, data)
    backend = t.str_("backend")
    if backend is not None:
        from stenograf.notes.backend import available_backends

        if backend not in available_backends():
            raise ValueError(
                f"unknown notes backend {backend!r} (choose from {', '.join(available_backends())})"
            )
    export = _Table(f"{label}.export", t.table("export"))
    settings = NotesSettings(
        auto=t.bool_("auto"),
        backend=backend,
        model=t.str_("model"),
        command=t.str_list("command"),
        timeout_s=t.number("timeout_s"),
        instructions=t.path("instructions"),
        template=t.path("template"),
        ollama_url=t.str_("ollama_url"),
        export_dir=export.path("dir"),
        max_input_chars=t.pos_int("max_input_chars"),
        thinking=t.bool_("thinking"),
    )
    export.reject_unknown()
    t.reject_unknown()
    return settings


def _presets_from_table(data: dict) -> dict[str, MeetingPreset]:
    """Every ``[meetings.<name>]`` section. The *names* are user-chosen, so
    ``reject_unknown`` cannot catch a typo'd preset name — ``steno presets``
    listing them and ``--preset`` echoing its pick are the mitigation."""
    outer = _Table("meetings", data)
    presets = {}
    for name in data:
        presets[name] = _preset_from_table(name, outer.table(name))
    outer.reject_unknown()
    return presets


def _preset_from_table(name: str, data: dict) -> MeetingPreset:
    label = f"meetings.{name}"
    t = _Table(label, data)
    language = t.str_("language")
    if language is not None:
        from stenograf.config import Language

        codes = [lang.value for lang in Language]
        if language not in codes:
            raise ValueError(f"{label}.language must be one of {', '.join(codes)}")
    cleared: set[str] = set()
    template = _clearable_path(t, "template", cleared)
    instructions = _clearable_path(t, "instructions", cleared)
    notes_raw = dict(t.table("notes"))
    export_raw = notes_raw.get("export")
    if isinstance(export_raw, dict) and export_raw.get("dir") == "":
        # `dir = ""` switches the standing export OFF for this preset (the
        # confidential-meeting case) — recorded as a clear, hidden from the
        # shared reader, which treats "" as the typo it is everywhere else.
        cleared.add("export.dir")
        notes_raw["export"] = {k: v for k, v in export_raw.items() if k != "dir"}
    preset = MeetingPreset(
        name=name,
        title=t.str_("title"),
        language=language,
        template=template,
        instructions=instructions,
        notes=_notes_from_table(notes_raw, label=f"{label}.notes"),
        vocab=_vocab_from_table(t.table("vocab"), label=f"{label}.vocab"),
        cleared=frozenset(cleared),
    )
    t.reject_unknown()
    return preset


def _clearable_path(t: _Table, key: str, cleared: set[str]) -> Path | None:
    """A preset path key where ``""`` means "switch the standing value off"."""
    raw = t.str_(key)
    if raw == "":
        cleared.add(key)
        return None
    return Path(raw).expanduser() if raw is not None else None


def apply_meeting_preset(settings: Settings, name: str) -> tuple[Settings, MeetingPreset]:
    """Overlay preset ``name`` onto ``settings``; unknown names fail loudly.

    Returns the overlaid :class:`Settings` (its ``[notes]`` table carries the
    preset's sparse choices) plus the preset itself, whose ``title`` /
    ``language`` / ``vocab`` the run-config seams consume — those never enter
    ``Settings``, because vocab *merges* and title/language are form defaults a
    flag still beats.

    The backend/model pair rule: a preset backend without a preset model drops
    the standing model (``create_backend``'s stale-model guard compares
    ``spec.name != settings.backend``, so overlaying the backend alone would
    make them *agree* and ride ``[notes] model`` into a backend it was never
    written for). A preset model without a backend applies to the standing
    backend — nulling the backend instead would hand the model to whatever the
    env/built-in default picks, which is the exact leak the rule prevents."""
    preset = settings.meetings.get(name)
    if preset is None:
        available = ", ".join(sorted(settings.meetings)) or "none defined"
        raise UnknownPresetError(f"unknown meeting preset {name!r} (available: {available})")
    changes: dict[str, object] = {}
    for f in fields(NotesSettings):
        value = getattr(preset.notes, f.name)
        if value is not None and value != ():
            changes[f.name] = value
    if preset.notes.backend is not None and preset.notes.model is None:
        changes["model"] = None
    if preset.template is not None:
        changes["template"] = preset.template
    if preset.instructions is not None:
        changes["instructions"] = preset.instructions
    cleared_fields = {
        "template": "template",
        "instructions": "instructions",
        "export.dir": "export_dir",
    }
    for key in preset.cleared:
        changes[cleared_fields[key]] = None
    vocab = settings.vocab
    if preset.vocab.glossary_threshold is not None:
        # The one [vocab] scalar: it replaces (merging makes no sense for a
        # threshold); the file and attendee lists merge at the run-config seam.
        vocab = replace(vocab, glossary_threshold=preset.vocab.glossary_threshold)
    overlaid = replace(settings, notes=replace(settings.notes, **changes), vocab=vocab)  # type: ignore[arg-type]
    return overlaid, preset


def ensure_settings_file() -> tuple[Path, bool]:
    """settings.toml's path, created from the commented template when missing.

    Shared by ``steno settings edit`` and the app's Settings screen — the
    template (every key present, commented out) is the editing surface both
    hand to the user's editor. Returns ``(path, created)``."""
    from stenograf.output import atomic_write_text

    path = settings_path()
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, SETTINGS_TEMPLATE)
    return path, True


def settings_rows(settings: Settings) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """``(table, [(key, value, source), …])`` rows behind every settings report.

    One renderer for ``steno settings show`` and the app's Settings screen, so
    no two entries can disagree about the effective configuration. Values are
    TOML-flavored so a line can be pasted into the file; defaults that aren't
    literal values (an unset optional, a per-backend choice) read as a
    parenthesized description instead."""
    from stenograf.asr.biasing import DEFAULT_ALPHA
    from stenograf.asr.registry import default_backend_name as asr_default
    from stenograf.glossary import DEFAULT_THRESHOLD as GLOSSARY_THRESHOLD
    from stenograf.notes.backend import default_backend_name as notes_default
    from stenograf.notes.backend import settings_defaults as notes_defaults
    from stenograf.notes.ollama import DEFAULT_URL
    from stenograf.output import default_output_home
    from stenograf.profiles import DEFAULT_THRESHOLD as REID_THRESHOLD
    from stenograf.profiles import default_store_path
    from stenograf.transcript import DEFAULT_FORMATS

    # Per-backend notes defaults resolve against the *effective* backend, so the
    # display matches what a notes run would actually use; keys the backend has
    # no say over get a which-backend placeholder.
    notes_backend = notes_default(settings.notes.backend)
    per_backend = notes_defaults(notes_backend)

    # (table, key, file value, effective default, env override) — one row each.
    descriptors = [
        ("transcript", "formats", settings.transcript.formats, DEFAULT_FORMATS, None),
        ("vocab", "glossary_file", settings.vocab.glossary_file, "(none)", None),
        ("vocab", "attendees", settings.vocab.attendees, "(none)", None),
        (
            "vocab",
            "glossary_threshold",
            settings.vocab.glossary_threshold,
            GLOSSARY_THRESHOLD,
            None,
        ),
        ("output", "dir", settings.output.dir, default_output_home(), None),
        ("output", "record_audio", settings.output.record_audio, False, None),
        ("speakers", "diarization", settings.speakers.diarization, False, None),
        ("speakers", "reid_threshold", settings.speakers.reid_threshold, REID_THRESHOLD, None),
        ("speakers", "profile_store", settings.speakers.profile_store, default_store_path(), None),
        ("asr", "backend", settings.asr.backend, asr_default(), "STENOGRAF_ASR_BACKEND"),
        ("asr", "provider", settings.asr.provider, "cpu", "STENOGRAF_ASR_PROVIDER"),
        ("asr", "boost", settings.asr.boost, DEFAULT_ALPHA, None),
        ("notes", "auto", settings.notes.auto, False, None),
        ("notes", "backend", settings.notes.backend, notes_backend, "STENOGRAF_NOTES_BACKEND"),
        (
            "notes",
            "model",
            settings.notes.model,
            per_backend.get("model", "(provenance label — none)"),
            "STENOGRAF_NOTES_MODEL",
        ),
        ("notes", "command", settings.notes.command, "(none)", None),
        (
            "notes",
            "timeout_s",
            settings.notes.timeout_s,
            per_backend.get("timeout_s", "(command backend only)"),
            None,
        ),
        ("notes", "instructions", settings.notes.instructions, "(none)", None),
        ("notes", "template", settings.notes.template, "(built-in)", None),
        ("notes", "ollama_url", settings.notes.ollama_url, DEFAULT_URL, "OLLAMA_HOST"),
        (
            "notes",
            "max_input_chars",
            settings.notes.max_input_chars,
            per_backend["max_input_chars"],
            None,
        ),
        (
            "notes",
            "thinking",
            settings.notes.thinking,
            per_backend.get("thinking", "(mlx backend only)"),
            None,
        ),
        ("notes.export", "dir", settings.notes.export_dir, "(off)", None),
    ]

    def pick(file_value: object, default: object, env_var: str | None) -> tuple[str, str]:
        if env_var and (env_value := os.environ.get(env_var)):
            return _fmt_setting(env_value), f"${env_var}"
        if file_value is not None and file_value != ():
            return _fmt_setting(file_value), "settings.toml"
        return _fmt_setting(default), "default"

    tables: dict[str, list[tuple[str, str, str]]] = {}
    for table, key, file_value, default, env_var in descriptors:
        tables.setdefault(table, []).append((key, *pick(file_value, default, env_var)))
    return list(tables.items())


def _fmt_setting(value: object) -> str:
    """One effective value, TOML-flavored (bools lowercase, arrays bracketed)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
