"""The launcher's home screen: one large labeled button per workflow.

Phase 7, Task 1 (PLAN.md §5). The menu is deliberately small and descriptive —
the audience is people who don't know the subcommands, so every button carries
a one-line description of what it does. Buttons are mouse-first (click to
activate) but fully keyboard-reachable (tab/arrows + enter).

*Start meeting* pushes the setup form (``ui.setup``) and hands a submitted
request to the meeting flow (``ui.flow``); every other workflow button pushes
its screen directly. Errors surfaced as toasts are mirrored on
:attr:`HomeScreen.notices` so tests assert behaviour without scraping toast
widgets (the plain-text-mirror rule).

Keyboard model: focus starts on the first button (the scroll container is
made non-focusable so it never swallows the initial focus), and the arrow
keys move focus between buttons — as *priority* screen bindings, because the
scroll container between the focused button and the screen binds the same
keys to scrolling and would win the normal bottom-up lookup. Moving focus
auto-scrolls the focused button into view, so a short terminal stays fully
reachable without dedicated scroll keys.
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

    BINDINGS = [
        Binding("q,escape", "app.quit", "Quit", show=True),
        # priority=True: beat #menu's own up/down→scroll bindings (see module doc).
        Binding("down", "app.focus_next", "Down", show=False, priority=True),
        Binding("up", "app.focus_previous", "Up", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.notices: list[str] = []  # plain-text mirror of the toasts shown

    def compose(self) -> ComposeResult:
        menu = VerticalScroll(id="menu")
        menu.can_focus = False  # focus belongs to the buttons; wheel/focus-follow scroll
        with menu:
            yield Static("stenograf", id="menu-title")
            yield Static("local meeting transcription", id="menu-tagline")
            for button_id, label, description in _MENU:
                yield Button(label, id=button_id)
                yield Static(description, classes="desc")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        elif event.button.id == "start":
            self._start_meeting()
        else:
            self.app.push_screen(self._make_screen(event.button.id or ""))

    def _make_screen(self, button_id: str) -> Screen:
        """The workflow screen behind a menu button (lazy: one import per press)."""
        if button_id == "transcribe":
            from stenograf.ui.transcribe import TranscribeScreen

            return TranscribeScreen()
        if button_id == "notes":
            from stenograf.ui.notes import NotesScreen

            return NotesScreen()
        if button_id == "settings":
            from stenograf.ui.settings import SettingsScreen

            return SettingsScreen()
        assert button_id == "doctor", f"unknown menu button {button_id!r}"
        from stenograf.ui.doctor import DoctorScreen

        return DoctorScreen()

    def _start_meeting(self) -> None:
        """Push the setup form; a submitted request starts the meeting flow."""
        # Lazy imports: flow reaches back through meeting → app → home, so a
        # module-level import here would be a cycle.
        from stenograf.ui.setup import MeetingRequest, MeetingSetupScreen

        def on_setup(request: MeetingRequest | None) -> None:
            if request is None:  # the form was cancelled
                return
            from stenograf.ui.flow import start_meeting

            try:
                start_meeting(self.app, request)
            except Exception as exc:  # e.g. an unwritable output home
                message = f"could not start the meeting: {exc}"
                self.notices.append(message)
                self.notify(message, title="Start meeting", severity="error")

        self.app.push_screen(MeetingSetupScreen(), on_setup)
