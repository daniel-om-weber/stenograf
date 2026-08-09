"""Generate notes for a finished transcript and write them beside it.

The one generation-and-write seam every entry point calls — ``steno notes``,
the ``--notes`` tail after a meeting, the app's notes screen and its meeting
flow — so the settings-snapshot rules, the backend hand-off and the atomic
write cannot fork per front-end. Generation goes through
:func:`stenograf.notes.generate.generate_notes`, which owns the MLX
thread-affinity guard — a caller must never reimplement it.

Progress and failure *presentation* stay with the caller: the CLI echoes,
the app routes to its status line, and :func:`run_notes` is the shared
non-fatal wrapper for a run that reports through a
:class:`~stenograf.view.LiveView`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import signal
from typing import TYPE_CHECKING

from stenograf.output import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime
    from pathlib import Path

    from stenograf.notes.backend import NotesBackend
    from stenograf.notes.model import MeetingNotes
    from stenograf.settings import NotesSettings
    from stenograf.transcript import Transcript
    from stenograf.view import LiveView


def resolve_backend(
    backend_name: str | None,
    model: str | None,
    ollama_url: str | None,
    notes_settings: NotesSettings | None,
    *,
    prebuilt_backend: NotesBackend | None = None,
) -> tuple[NotesBackend, NotesSettings]:
    """The backend a notes run with these arguments uses, plus the settings
    in force. Factored out so the meeting-start warmer
    (:mod:`stenograf.notes.warm`) builds the *identical* backend the notes
    tail would; ``prebuilt_backend`` short-circuits construction with one it
    built earlier — settings still resolve, they also pick the template and
    instructions."""
    if notes_settings is None:
        from stenograf.settings import load_settings

        notes_settings = load_settings().notes
    settings = notes_settings
    if backend_name and settings.backend and backend_name != settings.backend:
        # [notes] model in settings.toml was written for the configured
        # backend and must not ride along to an explicitly chosen other one
        # (an explicit model argument still wins).
        settings = dataclasses.replace(settings, backend=backend_name, model=None)
    if model or ollama_url:
        settings = dataclasses.replace(
            settings,
            model=model or settings.model,
            ollama_url=ollama_url or settings.ollama_url,
        )
    if prebuilt_backend is not None:
        return prebuilt_backend, settings
    from stenograf.notes import create_backend

    return create_backend(backend_name, settings), settings


def generate_and_write_notes(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    backend_name: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    instructions_file: Path | None = None,
    export_dir: Path | None = None,
    no_export: bool = False,
    notes_settings: NotesSettings | None = None,
    on_progress: Callable[[str], None] | None = None,
    prebuilt_backend: NotesBackend | None = None,
) -> tuple[list[Path], MeetingNotes]:
    """Generate notes and write ``<basename>.notes.md`` (plus the combined-note
    export when a target dir is configured). Returns ``(written_paths,
    notes)``; raises typed errors, writing nothing, on failure. Validation
    warnings ride on ``notes.provenance.warnings`` — the callers surface them.

    ``notes_settings`` is the ``[notes]`` table a command already loaded at its
    start (so a ``--notes`` run uses the values in force when the meeting began);
    ``None`` loads it here (the standalone ``steno notes`` path). ``on_progress``
    receives the generation progress lines; ``None`` stays silent — the caller
    owns presentation. ``prebuilt_backend`` is a backend the meeting-start
    warmer already built via :func:`resolve_backend` **with the same
    arguments** — anything else warms one model and generates with another."""
    from stenograf.notes.export import export_note
    from stenograf.notes.generate import generate_notes
    from stenograf.notes.template import DEFAULT_TEMPLATE

    backend, settings = resolve_backend(
        backend_name,
        model,
        ollama_url,
        notes_settings,
        prebuilt_backend=prebuilt_backend,
    )
    position = getattr(backend, "set_position", None)
    if position is not None:
        # The agentic contract (command backend only): the meeting folder is
        # the working directory, the parent — where the meetings live — rides
        # in the environment. Fetching context is the command's job.
        position(out_dir, out_dir.parent)
    instructions = None
    # A per-run --instructions file replaces the standing one, it does not
    # append: the flag exists to try a different style, not to stack two.
    instructions_path = instructions_file or settings.instructions
    if instructions_path is not None:
        instructions = instructions_path.read_text(encoding="utf-8")
    # [notes] template (or a preset's) is the protocol layout; its headings
    # are what validation matches the response against.
    template = DEFAULT_TEMPLATE
    if settings.template is not None:
        template = settings.template.read_text(encoding="utf-8")

    notes = generate_notes(
        transcript,
        backend,
        instructions=instructions,
        on_progress=on_progress,
        template=template,
        fallback_title=f"{created_at:%Y-%m-%d}",
    )

    md_path = out_dir / f"{basename}.notes.md"
    atomic_write_text(md_path, notes.to_markdown())
    # A regenerated meeting must not keep a stale JSON (the pre-markdown
    # format) that disagrees with its markdown.
    (out_dir / f"{basename}.notes.json").unlink(missing_ok=True)
    written = [md_path]

    target = None if no_export else (export_dir or settings.export_dir)
    if target is not None:
        written.append(export_note(transcript, notes, target, created_at=created_at))
    return written, notes


def run_notes(
    view: LiveView,
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    notes_settings: NotesSettings | None,
    backend_name: str | None = None,
    model: str | None = None,
    instructions_file: Path | None = None,
    prebuilt_backend: NotesBackend | None = None,
) -> bool:
    """The notes tail after a run: non-fatal, reported through a view.

    *The* wrapper for every after-the-transcript notes step — the app's meeting
    flow, ``steno start`` and ``steno transcribe`` — so all of them generate
    (and fail) identically. The transcript is already on disk when this runs,
    so a notes failure reports through the view and returns ``False`` — rerun
    later with ``steno notes``. A Ctrl-C during generation warns first and
    skips only on a second press (:func:`_second_interrupt_skips`).
    ``prebuilt_backend`` is the meeting-start warmer's backend; this call
    must then be running on the warmer's thread
    (:class:`stenograf.notes.warm.NotesWarmer`).
    """
    view.status("generating notes…")
    try:
        with _second_interrupt_skips(view):
            written, notes = generate_and_write_notes(
                transcript,
                out_dir,
                basename,
                created_at=created_at,
                notes_settings=notes_settings,
                backend_name=backend_name,
                model=model,
                instructions_file=instructions_file,
                on_progress=lambda message: view.status(f"notes: {message}"),
                prebuilt_backend=prebuilt_backend,
            )
    except KeyboardInterrupt:
        view.error(
            f"notes skipped — the transcript is safe; `steno notes {out_dir}` regenerates them"
        )
        return False
    except Exception as exc:  # noqa: BLE001 — non-fatal by contract
        view.error(
            f"notes failed: {exc} — the transcript is safe; retry with `steno notes {out_dir}`"
        )
        return False
    if notes.provenance is not None and notes.provenance.warnings:
        # A courtesy flash only — the durable copy is the note's own footer,
        # because the done screen's next status line overwrites this one.
        view.error(f"notes warning: {'; '.join(notes.provenance.warnings)}")
    view.status(f"notes: wrote {', '.join(str(p) for p in written)}")
    return True


@contextlib.contextmanager
def _second_interrupt_skips(view: LiveView) -> Iterator[None]:
    """First Ctrl-C during the notes tail warns and keeps going; the second skips.

    A stray Ctrl-C right after a meeting ends must not kill a half-written
    notes run — but an unconditional shield would trap the user behind an
    agentic ``[notes] command`` backend for up to its ``timeout_s`` (the same
    reason the Qt app quits *around* a notes run). So: announce once, and let
    a second press raise into :func:`run_notes`'s KeyboardInterrupt arm. A
    no-op off the main thread — a GUI worker — where handlers cannot be
    installed (and where no terminal delivers SIGINT anyway).
    """
    interrupted = False

    def handler(signum: int, frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        view.error("finishing notes — Ctrl-C again to skip them")

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):  # not on the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
