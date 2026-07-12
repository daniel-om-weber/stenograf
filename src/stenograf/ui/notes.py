"""Generate notes — the launcher's ``steno notes`` workflow.

Phase 7, Task 5 (PLAN.md §5). Two ways to name a meeting, both dumb:
the *last meeting* default (the ``notes --last`` semantics — the newest
finished ``meeting-*`` folder in the output home, pre-selected at mount) and
a plain :class:`~textual.widgets.DirectoryTree` picker for anything else.

**Guardrail (locked in the plan):** this screen stays a dumb file picker —
never a meeting list with titles, dates, or summaries. That list would be the
meeting browser the product philosophy forbids; keeping the screen dumb
enforces the lock structurally. The tree shows the filesystem exactly as
Finder would: folder names and ``transcript.json`` files, nothing parsed.

Generation goes through the shared notes entry point
(``_generate_and_write_notes``) on a thread worker — the entry point owns the
MLX thread-affinity guard (maintainability rule 2), the worker only keeps the
event loop free. Progress lines land on the status line via ``on_progress``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, Static

from stenograf.ui.widgets import FormScroll, NavDirectoryTree

if TYPE_CHECKING:
    from collections.abc import Callable


class _TranscriptTree(NavDirectoryTree):
    """Directory tree showing only directories and transcript JSON files."""

    def filter_paths(self, paths: Iterable[Path]) -> list[Path]:
        return [
            p
            for p in paths
            if not p.name.startswith(".") and (p.is_dir() or p.suffix.lower() == ".json")
        ]


class NotesScreen(Screen[None]):
    """Pick a meeting (last one pre-selected), generate its notes."""

    DEFAULT_CSS = """
    NotesScreen { align: center middle; }
    #panel {
        width: 80; max-width: 95%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #panel-title { text-align: center; text-style: bold; }
    #panel-hint { color: $text-muted; margin: 0 0 1 0; }
    #tree { height: 12; border: round $panel; }
    #picked { margin: 1 0 0 0; }
    #status { color: $text-muted; height: auto; }
    #actions { height: auto; margin: 1 0 0 0; }
    #actions Button { width: 1fr; }
    #actions #go { margin: 0 1 0 0; }
    """

    BINDINGS = [Binding("escape", "back", "Back", show=True)]

    def __init__(self, root: Path | None = None) -> None:
        # ``root`` overrides where the picker browses (tests pass a tmp dir);
        # by default it is the output home — the folder meetings land in.
        super().__init__()
        self._root = root
        self._target: Path | None = None  # a meeting folder or a transcript JSON
        self._busy = False  # a run is in flight (NOT "_running" — MessagePump owns that name)
        self.notices: list[str] = []  # plain-text mirror of the toasts shown
        self.status_text = ""  # plain-text mirror of the status line

    def compose(self) -> ComposeResult:
        from stenograf.output import default_output_home, latest_meeting_dir

        home = self._root
        if home is None:
            with contextlib.suppress(Exception):  # a broken settings.toml
                from stenograf.settings import load_settings

                home = load_settings().output.dir
            home = home or default_output_home()
        # The newest finished meeting is the default target — most notes runs
        # happen right after the meeting they summarize.
        self._target = latest_meeting_dir(home)

        with FormScroll(id="panel"):  # arrows walk tree/buttons, not the scrollbar
            yield Static("Generate notes", id="panel-title")
            yield Static(
                "The newest meeting is pre-selected; pick another folder "
                "(or transcript.json) to change it.",
                id="panel-hint",
            )
            if home.is_dir():
                yield _TranscriptTree(home, id="tree")
            else:
                yield Static(f"No meetings yet — {home} does not exist.", id="tree-missing")
            yield Static(self._target_label(), id="picked")
            yield Static("", id="status")
            with Horizontal(id="actions"):
                yield Button(
                    "Generate notes", variant="success", id="go", disabled=self._target is None
                )
                yield Button("Back", id="back")
        yield Footer()

    def _target_label(self) -> str:
        if self._target is None:
            return "No meeting selected."
        return f"Meeting: {self._target}"

    # -- picking -------------------------------------------------------------

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._set_target(event.path)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._set_target(event.path)

    def _set_target(self, path: Path) -> None:
        self._target = path
        self.query_one("#picked", Static).update(self._target_label())
        self.query_one("#go", Button).disabled = self._busy

    # -- leaving -------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go" and self._target is not None:
            self._start(self._target)
        elif event.button.id == "back":
            self.action_back()

    def action_back(self) -> None:
        if self._busy:
            self._notice("Still generating — it finishes in place.", severity="warning")
            return
        self.dismiss(None)

    # -- the run -------------------------------------------------------------

    def _start(self, target: Path) -> None:
        self._busy = True
        self.query_one("#go", Button).disabled = True
        self._set_status("generating notes…")
        self._generate(target)

    @work(thread=True, exclusive=True)
    def _generate(self, target: Path) -> None:
        """Resolve the target and run the shared notes entry point (rule 2)."""
        from stenograf.cli.notes import _generate_and_write_notes
        from stenograf.output import TRANSCRIPT_STEM, created_at_from_dir_name
        from stenograf.transcript import Transcript

        try:
            path = target / f"{TRANSCRIPT_STEM}.json" if target.is_dir() else target
            if not path.is_file():
                raise ValueError(f"{target} holds no {TRANSCRIPT_STEM}.json")
            try:
                transcript = Transcript.from_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"{path} is not a readable transcript JSON: {exc}") from exc
            out_dir = path.parent
            created_at = created_at_from_dir_name(out_dir.name) or datetime.fromtimestamp(
                path.stat().st_mtime
            )
            written, _notes = _generate_and_write_notes(
                transcript,
                out_dir,
                path.stem,
                created_at=created_at,
                on_progress=lambda message: self._post(self._set_status, f"notes: {message}"),
            )
        except Exception as exc:  # noqa: BLE001 — every failure lands on the status line
            self._post(self._fail, str(exc))
            return
        self._post(self._finish, f"wrote {', '.join(str(p) for p in written)}")

    def _post(self, fn: Callable[..., object], *args: object) -> None:
        """Marshal a UI mutation from the worker thread onto the app thread."""
        with contextlib.suppress(Exception):  # app may be shutting down
            self.app.call_from_thread(fn, *args)

    # -- UI-thread mutators ----------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.status_text = message
        self.query_one("#status", Static).update(message)

    def _finish(self, message: str) -> None:
        self._end_run(message)
        self._notice(message, title="Notes saved", timeout=10)

    def _fail(self, message: str) -> None:
        self._end_run(f"failed: {message}")
        self._notice(message, title="Notes failed", severity="error", timeout=10)

    def _end_run(self, message: str) -> None:
        self._busy = False
        self._set_status(message)
        self.query_one("#go", Button).disabled = self._target is None

    def _notice(self, message: str, **notify_kwargs: object) -> None:
        self.notices.append(message)
        self.notify(message, **notify_kwargs)  # type: ignore[arg-type]
