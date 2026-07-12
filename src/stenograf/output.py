"""Where finished meetings land: a visible folder of self-describing dirs.

stenograf's responsibility ends at producing text — the transcript and the
notes. There is no index and no managed library (PLAN.md §5 Stage C): each run
writes one date-named folder (``meeting-YYYYMMDD-HHMMSS/``) into a user-visible
output home — ``~/Documents/Meetings`` by default, ``[output] dir`` in
settings.toml or ``--out`` to override — holding plainly named files::

    meeting-20260710-091500/
        transcript.md / .json / .txt / …   # the finalize output (--format)
        transcript.partial.*               # crash checkpoint, removed on success
        transcript.notes.md / .notes.json  # `steno notes` siblings
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

if TYPE_CHECKING:
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
    ever sees the whole old file or the whole new one (PLAN.md §5 Phase 3→4
    audit). Creates the parent directory on demand — a meeting dir exists from
    its first write, never earlier (see :func:`allocate_meeting_dir`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def default_output_home() -> Path:
    """The standing folder new meeting dirs are created in: ``~/Documents/Meetings``.

    Deliberately a *visible* location — transcripts and notes are user documents,
    not app state (PLAN.md §5 Stage C1). ``[output] dir`` in settings.toml
    replaces it; ``--out`` bypasses it for one run."""
    return Path.home() / "Documents" / "Meetings"


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


def cleanup_checkpoints(out_dir: Path, basename: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for fmt in CHECKPOINT_FORMATS:
        (out_dir / f"{basename}.partial.{fmt}").unlink(missing_ok=True)


def checkpoint_writer(
    out_dir: Path, basename: str, announce: Callable[[str], None] | None = None
) -> Callable[[Transcript], None]:
    """Build the ``on_checkpoint`` sink that writes the ``.partial`` crash file.

    Live views keep the caption stream clean (``announce=None`` → write silently);
    the batch path narrates each write, as it always has. The final transcript
    supersedes these files, which :func:`cleanup_checkpoints` then removes.
    """

    def on_checkpoint(transcript: Transcript) -> None:
        md = write_transcript(transcript, out_dir, f"{basename}.partial", CHECKPOINT_FORMATS)[0]
        if announce is not None:
            announce(f"checkpoint: {md.name} ({len(transcript.entries)} entries)")

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
