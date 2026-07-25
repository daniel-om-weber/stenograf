"""Transcribe a recording — the launcher's batch-finalize workflow.

A :class:`~textual.widgets.DirectoryTree` file
picker, :func:`stenograf.flow.transcribe_recording` in a ``@work(thread=True)``
worker, and a :class:`~textual.widgets.ProgressBar` driven by that run's
window-progress callback. The run is the launcher-shaped ``steno transcribe``:
everything the CLI resolves from flags comes from settings.toml instead
(channels auto-detected, speaker counts estimated, re-ID on — rerun with the
CLI to override), and the output lands in a fresh date-named folder under the
output home, so it can never collide with an existing meeting.

Split-channel recordings (a ``--record-audio`` tee, a dual-channel call) take
the same per-channel meeting finalize the CLI runs. There is no window-count
progress on that path (the meeting finalize reports per channel, not per
window), so the bar stays at its indeterminate pulse and the status line
carries the detail.

Leaving mid-run is refused: a thread worker cannot be interrupted safely, and
silently letting it finish behind a popped screen would surprise the user
more than the refusal does.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, ProgressBar, Static

from stenograf.ui.widgets import FormScroll, NavDirectoryTree

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

        The run itself is :func:`stenograf.flow.transcribe_recording` — shared
        with the Qt app, so both report the same thing for the same file. All
        UI mutations are marshalled back onto the app thread via :meth:`_post`.
        """
        from stenograf.flow import transcribe_recording

        try:
            result = transcribe_recording(
                audio_file,
                on_status=lambda message: self._post(self._set_status, message),
                on_windows=lambda done, total: self._post(self._set_progress, done, total),
            )
        except Exception as exc:  # noqa: BLE001 — every failure lands on the status line
            self._post(self._fail, str(exc))
            return
        self._post(self._finish, result.summary(), result.out_dir)

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
