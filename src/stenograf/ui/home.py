"""The launcher's home screen: one large labeled button per workflow.

Phase 7, Task 1 (PLAN.md §5). The menu is deliberately small and descriptive —
the audience is people who don't know the subcommands, so every button carries
a one-line description of what it does. Buttons are mouse-first (click to
activate) but fully keyboard-reachable (tab/arrows + enter).

Until their screens exist (Tasks 2–5), workflow buttons are stubs: pressing
one shows a notice naming the CLI command that already does the job. The
notices are mirrored on :attr:`HomeScreen.notices` so tests assert behaviour
without scraping toast widgets (the plain-text-mirror rule).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Static

_MENU: tuple[tuple[str, str, str], ...] = (
    ("start", "Start meeting", "Capture this meeting with live captions."),
    ("transcribe", "Transcribe a recording", "Turn an audio file into a transcript."),
    ("notes", "Generate notes", "Summarize a finished meeting's transcript."),
    ("settings", "Settings", "Show the active configuration."),
    ("doctor", "Check setup", "Verify models, permissions, and audio devices."),
    ("quit", "Quit", "Leave the launcher."),
)
"""``(button id, label, description)`` per menu entry, in display order."""

_STUB_HINT = {
    "start": "steno start",
    "transcribe": "steno transcribe <file>",
    "notes": "steno notes --last",
    "settings": "steno settings show",
    "doctor": "steno doctor",
}
"""The CLI command a stubbed button points at until its screen ships."""


class HomeScreen(Screen[None]):
    """Menu of workflow buttons; every workflow screen is pushed from here."""

    DEFAULT_CSS = """
    HomeScreen { align: center middle; }
    /* height: auto + max-height: 100% — the menu hugs its content but scrolls
       instead of clipping on a terminal shorter than the full button list. */
    #menu {
        width: 56; max-width: 90%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #menu-title { text-align: center; text-style: bold; }
    #menu-tagline { text-align: center; color: $text-muted; margin: 0 0 1 0; }
    #menu Button { width: 100%; }
    #menu .desc { color: $text-muted; margin: 0 0 1 2; }
    """

    BINDINGS = [Binding("q,escape", "app.quit", "Quit", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self.notices: list[str] = []  # plain-text mirror of the toasts shown

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="menu"):
            yield Static("stenograf", id="menu-title")
            yield Static("local meeting transcription", id="menu-tagline")
            for button_id, label, description in _MENU:
                yield Button(label, id=button_id)
                yield Static(description, classes="desc")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
            return
        self._stub(event.button.id or "")

    def _stub(self, button_id: str) -> None:
        """Task-1 placeholder: name the CLI command until the screen exists."""
        message = f"Not built yet — run `{_STUB_HINT[button_id]}` from the command line."
        self.notices.append(message)
        self.notify(message, title="Coming soon", severity="warning")
