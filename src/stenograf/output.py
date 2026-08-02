"""Where finished meetings land: a visible folder of self-describing dirs.

stenograf's responsibility ends at producing text — the transcript and the
notes. There is no index and no managed library: each run
writes one date-named folder (``meeting-YYYYMMDD-HHMMSS/``) into a user-visible
output home — ``Meetings`` inside the user's documents folder by default
(:func:`documents_dir`), ``[output] dir`` in settings.toml or ``--out`` to
override — holding plainly named files::

    meeting-20260710-091500/
        transcript.md / .json / .txt / …   # the finalize output (--format)
        transcript.partial.*               # crash checkpoint, removed on success
        transcript.notes.md                # the `steno notes` sibling
        audio.wav                          # only with --record-audio

The filesystem *is* the index: the folder name carries the date, the exported
note's filename carries the title, listing is Finder/``ls``, deleting is ``rm``.
The one remaining lookup — "the newest meeting", for ``steno notes --last`` —
is a name scan (:func:`latest_meeting_dir`). Machine state (voiceprints,
settings.toml, the model cache) stays in the data dir; user documents do not.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from stenograf.paths import documents_dir

if TYPE_CHECKING:
    from stenograf.settings import Settings
    from stenograf.transcript import Transcript

TRANSCRIPT_STEM = "transcript"
"""Basename (without extension) of the transcript files in a meeting dir."""

AUDIO_NAME = "audio.wav"
"""Name of the opt-in ``--record-audio`` WAV inside a meeting dir."""

CHECKPOINT_FORMATS = ("md", "json", "txt")
"""Crash checkpoints render these (no subtitles — pointless for a partial
transcript). :func:`cleanup_checkpoints` must remove exactly this set."""

_DIR_TIMESTAMP = re.compile(r"^meeting-(\d{8})-(\d{6})")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a sibling temp file + ``os.replace`` (atomic on POSIX/Windows).

    A plain ``write_text`` truncates in place, so a crash mid-write leaves a
    corrupt file — and for the ``.partial`` crash-recovery checkpoint that also
    destroys the previous good copy, defeating the artifact meant to survive the
    crash. Writing a sibling temp then atomically renaming means a reader only
    ever sees the whole old file or the whole new one. Creates the parent
    directory on demand — a meeting dir exists from
    its first write, never earlier (see :func:`allocate_meeting_dir`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_output_home() -> Path:
    """The standing folder new meeting dirs are created in: ``<documents>/Meetings``.

    Deliberately a *visible* location — transcripts and notes are user documents,
    not app state — and the documents folder the desktop itself names, so the
    output home is one the file manager already shows (:func:`documents_dir`).
    ``[output] dir`` in settings.toml replaces it; ``--out`` bypasses it for one
    run. ``steno settings show`` prints whichever of the three is in force."""
    return documents_dir() / "Meetings"


def output_home(settings: Settings) -> Path:
    """The output home in force: ``[output] dir`` or the platform default."""
    return settings.output.dir or default_output_home()


def allocate_meeting_dir(home: Path, created_at: datetime) -> Path:
    """Pick this meeting's directory under ``home``: ``meeting-YYYYMMDD-HHMMSS``.

    On a name collision (a second meeting in the same second, or any pre-existing
    entry) append ``-2``, ``-3``, … until the name is free on disk. The directory
    is not created here — the first write (checkpoint, transcript, audio tee)
    creates it, so an aborted run leaves nothing behind."""
    base = f"meeting-{created_at:%Y%m%d-%H%M%S}"
    candidate = home / base
    suffix = 2
    while candidate.exists():
        candidate = home / f"{base}-{suffix}"
        suffix += 1
    return candidate


def latest_meeting_dir(home: Path) -> Path | None:
    """The newest ``meeting-*`` dir in ``home`` holding a ``transcript.json``.

    "Newest" is by directory name, descending — the name encodes the start time,
    so no index or mtime is consulted. Dirs without a ``transcript.json`` (a
    crashed run that left only ``.partial`` checkpoints, an unrelated folder)
    are skipped. ``None`` when the home holds no finished meeting."""
    if not home.is_dir():
        return None
    for child in sorted(home.iterdir(), key=lambda p: p.name, reverse=True):
        if (
            child.is_dir()
            and _DIR_TIMESTAMP.match(child.name)
            and (child / f"{TRANSCRIPT_STEM}.json").is_file()
        ):
            return child
    return None


def write_transcript(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    formats: tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    """Write the transcript in each requested format; returns the written paths.

    ``basename`` is the full file stem (extension excluded) — ``transcript`` for
    a meeting folder, or e.g. ``transcript.partial`` for a crash checkpoint.
    Markdown + JSON + plain text are the default (the only files stenograf
    emits unless the user asks for subtitles); SRT/VTT are opt-in via ``--format``.
    """
    from stenograf.transcript import DEFAULT_FORMATS, FORMATS

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in DEFAULT_FORMATS if formats is None else formats:
        path = out_dir / f"{basename}.{fmt}"
        atomic_write_text(path, FORMATS[fmt](transcript))
        paths.append(path)
    return paths


class PersistOnce:
    """Persist the finalized transcript exactly once, wherever that fires first.

    Both front-ends hand one of these to their view/run so the transcript
    reaches disk at the ``finalized`` event — a force-quit on the "done" screen,
    or even mid-finalize, never loses the meeting. A first call that fails
    leaves ``paths`` unset, so a later call retries and a raise there surfaces
    normally; a second call after success is a no-op returning the
    already-written paths.
    """

    def __init__(self, write: Callable[[Transcript], list[Path]]) -> None:
        self._write = write
        self.paths: list[Path] | None = None

    def __call__(self, transcript: Transcript) -> list[Path]:
        if self.paths is None:
            self.paths = self._write(transcript)
        return self.paths


def prepare_output(
    out: Path | None, created_at: datetime, settings: Settings, *, force: bool = False
) -> tuple[Path, str, Path]:
    """Resolve the directory this run's files land in.

    Returns ``(out_dir, basename, audio_default)``. By default the meeting gets
    a fresh date-named folder under the visible output home (``[output] dir``
    in settings.toml, else ``Meetings`` in the user's documents folder — see
    :func:`default_output_home`); an explicit ``out`` (the CLI's ``--out``)
    uses that path itself as the meeting's folder. Either way the files inside
    are plainly named — ``transcript.{fmt}``, ``audio.wav``.

    File names inside a meeting folder are fixed, so pointing ``out`` at a
    folder that already holds a transcript would silently replace that meeting;
    raises :class:`FileExistsError` unless ``force`` says overwriting is the
    point (a re-run over the same recording). The default path allocates a
    fresh name and cannot collide; ``.partial`` checkpoints don't count —
    resuming after a crash must not demand ``force``."""
    from stenograf.transcript import FORMATS

    if out is not None:
        if not force:
            existing = next(
                (
                    f"{TRANSCRIPT_STEM}.{ext}"
                    for ext in FORMATS
                    if (out / f"{TRANSCRIPT_STEM}.{ext}").exists()
                ),
                None,
            )
            if existing is not None:
                raise FileExistsError(
                    f"{out} already holds {existing} — pass --force to overwrite "
                    "this meeting's files, or drop --out for a fresh folder"
                )
        out_dir = out
    else:
        out_dir = allocate_meeting_dir(output_home(settings), created_at)
    return out_dir, TRANSCRIPT_STEM, out_dir / AUDIO_NAME


def load_transcript(target: Path) -> tuple[Transcript, Path, datetime]:
    """Load a finished meeting's transcript from a folder or a transcript JSON.

    The one way *back* from disk: every consumer of a finished meeting —
    ``steno notes``, both notes screens — resolves its user-named target
    here, so "a meeting folder or its transcript.json" means the same thing
    (and fails with the same message) everywhere. Returns ``(transcript,
    json_path, created_at)``; ``created_at`` is the start time the date-named
    folder encodes, else the file's mtime (a loose transcript file). Raises
    :class:`ValueError` with a user-facing message for a target that holds no
    transcript or one that does not parse."""
    from stenograf.transcript import Transcript

    path = target / f"{TRANSCRIPT_STEM}.json" if target.is_dir() else target
    if not path.is_file():
        raise ValueError(f"{target} holds no {TRANSCRIPT_STEM}.json")
    try:
        transcript = Transcript.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path} is not a readable transcript JSON: {exc}") from exc
    created_at = created_at_from_dir_name(path.parent.name) or datetime.fromtimestamp(
        path.stat().st_mtime
    )
    return transcript, path, created_at


def cleanup_checkpoints(out_dir: Path, basename: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for fmt in CHECKPOINT_FORMATS:
        (out_dir / f"{basename}.partial.{fmt}").unlink(missing_ok=True)


def checkpoint_writer(out_dir: Path, basename: str) -> Callable[[Transcript], None]:
    """Build the ``on_checkpoint`` sink that writes the ``.partial`` crash file.

    Writes silently — the live caption stream stays clean. The final
    transcript supersedes these files, which :func:`cleanup_checkpoints` then
    removes.
    """

    def on_checkpoint(transcript: Transcript) -> None:
        write_transcript(transcript, out_dir, f"{basename}.partial", CHECKPOINT_FORMATS)

    return on_checkpoint


def created_at_from_dir_name(name: str) -> datetime | None:
    """Recover the start time a ``meeting-YYYYMMDD-HHMMSS`` dir name encodes,
    or ``None`` for any other name (then fall back to file mtime)."""
    match = _DIR_TIMESTAMP.match(name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None
