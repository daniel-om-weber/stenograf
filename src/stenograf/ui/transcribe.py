"""Transcribe a recording — the launcher's batch-finalize workflow.

Phase 7, Task 4 (PLAN.md §5). A :class:`~textual.widgets.DirectoryTree` file
picker, the transcription pipeline in a ``@work(thread=True)`` worker, and a
:class:`~textual.widgets.ProgressBar` driven by the existing ``transcribe``
progress callback (``finalize_file``'s ``on_progress``). The worker body is
the launcher-shaped ``steno transcribe``: everything the CLI resolves from
flags comes from settings.toml instead (channels auto-detected, speaker
counts estimated, re-ID on — rerun with the CLI to override), and the output
lands in a fresh date-named folder under the output home, so it can never
collide with an existing meeting.

Split-channel recordings (a ``--record-audio`` tee, a dual-channel call) take
the same per-channel meeting finalize the CLI runs; its status lines reach
the screen through the ``view`` seam on ``_transcribe_split_channels`` — the
CLI helpers are reused, never reimplemented (the thin-client rule). There is
no window-count progress on that path (the meeting finalize reports per
channel, not per window), so the bar stays at its indeterminate pulse and the
status line carries the detail.

Leaving mid-run is refused: a thread worker cannot be interrupted safely, and
silently letting it finish behind a popped screen would surprise the user
more than the refusal does.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, ProgressBar, Static

from stenograf.ui.widgets import FormScroll, NavDirectoryTree
from stenograf.view import LiveView

if TYPE_CHECKING:
    from collections.abc import Callable

_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
    ".mka",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
}  # anything ffmpeg decodes; video containers included (their audio track is used)


def _shows_in_picker(path: Path) -> bool:
    """Visible in the tree: non-hidden directories and transcribable files."""
    return not path.name.startswith(".") and (
        path.is_dir() or path.suffix.lower() in _AUDIO_SUFFIXES
    )


class _AudioTree(NavDirectoryTree):
    """Directory tree showing only directories and transcribable files."""

    def filter_paths(self, paths: Iterable[Path]) -> list[Path]:
        return [p for p in paths if _shows_in_picker(p)]


class _ScreenStatusView(LiveView):
    """Routes the split-channel finalize's status lines onto the screen."""

    def __init__(self, post: Callable[[str], None]) -> None:
        self._post = post

    def status(self, message: str) -> None:
        self._post(message)

    def error(self, message: str) -> None:
        self._post(f"warning: {message}")


class TranscribeScreen(Screen[None]):
    """Pick an audio file, run the finalize pipeline, watch the progress."""

    DEFAULT_CSS = """
    TranscribeScreen { align: center middle; }
    #panel {
        width: 80; max-width: 95%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #panel-title { text-align: center; text-style: bold; }
    #panel-hint { color: $text-muted; margin: 0 0 1 0; }
    #tree { height: 14; border: round $panel; }
    #picked { margin: 1 0 0 0; }
    #status { color: $text-muted; height: auto; }
    #progress { margin: 1 0 0 0; width: 100%; display: none; }
    #actions { height: auto; margin: 1 0 0 0; }
    #actions Button { width: 1fr; }
    #actions #go { margin: 0 1 0 0; }
    """

    BINDINGS = [Binding("escape", "back", "Back", show=True)]

    def __init__(self, root: Path | None = None) -> None:
        # ``root`` is where the picker starts browsing (tests pass a tmp dir).
        super().__init__()
        self._root = root if root is not None else Path.home()
        self._selected: Path | None = None
        self._busy = False  # a run is in flight (NOT "_running" — MessagePump owns that name)
        self.notices: list[str] = []  # plain-text mirror of the toasts shown
        self.status_text = ""  # plain-text mirror of the status line

    def compose(self) -> ComposeResult:
        with FormScroll(id="panel"):  # arrows walk tree/buttons, not the scrollbar
            yield Static("Transcribe a recording", id="panel-title")
            yield Static("Pick an audio (or video) file, then press Transcribe.", id="panel-hint")
            yield _AudioTree(self._root, id="tree")
            yield Static("No file selected.", id="picked")
            yield Static("", id="status")
            yield ProgressBar(id="progress", show_eta=False)
            with Horizontal(id="actions"):
                yield Button("Transcribe", variant="success", id="go", disabled=True)
                yield Button("Back", id="back")
        yield Footer()

    # -- picking -------------------------------------------------------------

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected = event.path
        self.query_one("#picked", Static).update(f"Selected: {event.path}")
        self.query_one("#go", Button).disabled = self._busy

    # -- leaving -------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go" and self._selected is not None:
            self._start(self._selected)
        elif event.button.id == "back":
            self.action_back()

    def action_back(self) -> None:
        if self._busy:
            self._notice("Still transcribing — it finishes in place.", severity="warning")
            return
        self.dismiss(None)

    # -- the run -------------------------------------------------------------

    def _start(self, audio_file: Path) -> None:
        self._busy = True
        self.query_one("#go", Button).disabled = True
        self.query_one("#tree", DirectoryTree).disabled = True
        self.query_one("#progress", ProgressBar).display = True
        self._set_status(f"transcribing {audio_file.name}…")
        self._transcribe(audio_file)

    @work(thread=True, exclusive=True)
    def _transcribe(self, audio_file: Path) -> None:
        """The pipeline, off the event loop (maintainability rule 1).

        The body is ``steno transcribe`` minus the flags: settings.toml
        supplies formats/vocab/ASR backend, channels and speaker counts are
        auto, and the folder is freshly allocated. All UI mutations are
        marshalled back onto the app thread via :meth:`_post`.
        """
        from stenograf import loaders
        from stenograf.audio import SAMPLE_RATE, load_audio
        from stenograf.cli.run import _collect_terms
        from stenograf.cli.transcribe import (
            _resolve_split_channels,
            _transcribe_split_channels,
        )
        from stenograf.config import MeetingProfile
        from stenograf.output import (
            TRANSCRIPT_STEM,
            allocate_meeting_dir,
            default_output_home,
            write_transcript,
        )
        from stenograf.settings import load_settings
        from stenograf.transcript import DEFAULT_FORMATS

        try:
            settings = load_settings()
            glossary_terms, attendee_names = _collect_terms((), None, (), vocab=settings.vocab)
            out_dir = allocate_meeting_dir(
                settings.output.dir or default_output_home(), datetime.now()
            )
            write_formats = list(settings.transcript.formats or DEFAULT_FORMATS)

            split_pcms, _correlation = _resolve_split_channels(audio_file, "auto")
            # Diarization is off unless [speakers] diarization = true — the
            # launcher's only on switch (or rerun with the CLI's --diarization):
            # counts collapse to one speaker per channel and the diarizer is
            # never loaded.
            diarize = settings.speakers.diarization is True
            profile = MeetingProfile(
                glossary=glossary_terms,
                attendee_names=attendee_names,
                title=audio_file.stem,
            )
            if split_pcms is not None:
                if not diarize:
                    profile = dataclasses.replace(profile, local_speakers=1, remote_speakers=1)
                duration = len(split_pcms[0]) / SAMPLE_RATE
                self._post(self._set_status, "2 voice channels — transcribing per channel…")
                result, elapsed = _transcribe_split_channels(
                    *split_pcms,
                    profile=profile,
                    use_reid=True,
                    reid_threshold=settings.speakers.reid_threshold,
                    glossary_threshold=settings.vocab.glossary_threshold,
                    asr_backend=settings.asr.backend,
                    asr_provider=settings.asr.provider,
                    profile_store=settings.speakers.profile_store,
                    view=_ScreenStatusView(lambda m: self._post(self._set_status, m)),
                )
                transcript = result.transcript
            else:
                from stenograf.pipeline import STAGE_ASR, STAGE_DIARIZATION, finalize_file

                samples = load_audio(audio_file)
                duration = len(samples) / SAMPLE_RATE
                self._post(self._set_status, "loading models…")
                asr, vad, diarizer = loaders.load_backends(
                    need_diarizer=diarize,
                    asr_backend=settings.asr.backend,
                    asr_provider=settings.asr.provider,
                    # Not click: Textual owns stdio (loaders module docstring).
                    announce=lambda message: self._post(self._set_status, message),
                )
                started = time.monotonic()
                reid = None
                if diarizer is not None:  # re-ID relabels diarized speakers only
                    reid = loaders.load_reid(
                        enabled=True,
                        threshold=settings.speakers.reid_threshold,
                        store_path=settings.speakers.profile_store,
                    )

                def progress(stage: str, done: int, total: int) -> None:
                    if stage == STAGE_ASR:
                        self._post(self._set_progress, done, total)
                    elif stage == STAGE_DIARIZATION:
                        self._post(self._set_status, "diarizing…")

                transcript = finalize_file(
                    samples,
                    profile=profile,
                    asr=asr,
                    vad=vad,
                    diarizer=diarizer,
                    num_speakers=None if diarize else 1,
                    reid=reid,
                    glossary_threshold=settings.vocab.glossary_threshold,
                    on_progress=progress,
                )
                elapsed = time.monotonic() - started

            paths = write_transcript(transcript, out_dir, TRANSCRIPT_STEM, write_formats)
        except Exception as exc:  # noqa: BLE001 — every failure lands on the status line
            self._post(self._fail, str(exc))
            return
        speed = duration / elapsed if elapsed else 0.0
        self._post(
            self._finish,
            f"wrote {', '.join(p.name for p in paths)} → {out_dir} "
            f"({elapsed:.1f}s, {speed:.1f}x realtime)",
            out_dir,
        )

    def _post(self, fn: Callable[..., object], *args: object) -> None:
        """Marshal a UI mutation from the worker thread onto the app thread."""
        with contextlib.suppress(Exception):  # app may be shutting down
            self.app.call_from_thread(fn, *args)

    # -- UI-thread mutators ----------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.status_text = message
        self.query_one("#status", Static).update(message)

    def _set_progress(self, done: int, total: int) -> None:
        if done == 0:
            self._set_status(f"transcribing {total} windows…")
        self.query_one("#progress", ProgressBar).update(total=total, progress=done)

    def _finish(self, message: str, out_dir: Path) -> None:
        self._end_run(message)
        self._notice(f"Files in {out_dir}", title="Transcription saved", timeout=10)

    def _fail(self, message: str) -> None:
        self._end_run(f"failed: {message}")
        self._notice(message, title="Transcription failed", severity="error", timeout=10)

    def _end_run(self, message: str) -> None:
        self._busy = False
        self._set_status(message)
        self.query_one("#progress", ProgressBar).display = False
        self.query_one("#tree", DirectoryTree).disabled = False
        self.query_one("#go", Button).disabled = self._selected is None

    def _notice(self, message: str, **notify_kwargs: object) -> None:
        self.notices.append(message)
        self.notify(message, **notify_kwargs)  # type: ignore[arg-type]
