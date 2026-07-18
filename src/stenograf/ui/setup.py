"""Meeting setup — the few choices that matter before capture starts.

Phase 7, Task 3 (PLAN.md §5). The launcher's pre-meeting form, one concept per
control: which sources to capture (two switches), whether to tell speakers
apart (the diarization switch — the per-channel counts only appear while it is
on), language, an optional title, the audio-recording opt-in, and the notes
toggle. Everything else — formats, vocabulary, re-ID, AEC, checkpoint cadence
— comes from settings.toml exactly as it does for a flagless ``steno start``,
resolved through the same helpers the CLI uses (``cli/run.py``), so the two
entries can never disagree about defaults.

Submitting validates here (a bad profile keeps the form open with the error
shown) and dismisses with a :class:`MeetingRequest`; the flow module turns
that into a running meeting. Errors surfaced as toasts are mirrored on
:attr:`MeetingSetupScreen.notices` (the plain-text-mirror rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Select, Static, Switch

from stenograf.config import Language, MeetingProfile
from stenograf.ui.widgets import FormScroll, FormSelect

if TYPE_CHECKING:
    from stenograf.settings import Settings

_AUTO = -1
"""Select sentinel for "auto-detect" (Select values must not be None)."""

_COUNT_CHOICES = [("Auto-detect", _AUTO)] + [
    (str(n), n)
    for n in range(1, 9)  # MeetingProfile caps counts at 8
]

_LANGUAGE_CHOICES = [("Auto-detect", "auto")] + [
    (lang.name.title(), lang.value) for lang in Language
]


@dataclass(frozen=True)
class MeetingRequest:
    """What the form resolved: the profile to record plus the run-level extras.

    ``settings`` rides along so the flow uses the exact values in force when
    the user pressed Start — not whatever the file says seconds later.
    ``record_audio`` is the CLI's bare ``--record-audio``: keep the raw
    capture as the meeting folder's ``audio.wav``."""

    profile: MeetingProfile
    settings: Settings
    notes: bool
    record_audio: bool


class MeetingSetupScreen(Screen[MeetingRequest | None]):
    """Form pushed by Home's *Start meeting*; dismisses with a request or None."""

    DEFAULT_CSS = """
    MeetingSetupScreen { align: center middle; }
    #form {
        width: 56; max-width: 90%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #form-title { text-align: center; text-style: bold; margin: 0 0 1 0; }
    .field-label { margin: 1 0 0 0; }
    .hint { color: $text-muted; }
    #form Select, #form Input { width: 100%; }
    .switch-row { height: auto; margin: 1 0 0 0; }
    .switch-row Static { padding: 0 0 0 1; width: 1fr; }
    #counts { height: auto; }
    #actions { height: auto; margin: 1 0 0 0; }
    #actions Button { width: 1fr; }
    #actions #go { margin: 0 1 0 0; }
    """

    BINDINGS = [Binding("escape", "cancel", "Back", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self.notices: list[str] = []  # plain-text mirror of the toasts shown

    def compose(self) -> ComposeResult:
        standing = self._standing_settings()
        diarize = standing.speakers.diarization is True
        with FormScroll(id="form"):  # arrows walk the fields, not the scrollbar
            yield Static("Start meeting", id="form-title")
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="mic")
                yield Static("Microphone — people in the room")
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="system")
                yield Static("System audio — calls, videos")
            with Horizontal(classes="switch-row"):
                yield Switch(value=diarize, id="diarize")
                yield Static("Tell speakers apart (diarization)")
            yield Static("Off: each source is one speaker in the transcript.", classes="hint")
            counts = Vertical(id="counts")
            counts.display = diarize  # the counts only mean something while diarizing
            with counts:
                yield Static("Speakers in the room (microphone)", classes="field-label")
                yield FormSelect(_COUNT_CHOICES, value=_AUTO, allow_blank=False, id="local")
                yield Static("Remote speakers (system audio)", classes="field-label")
                yield FormSelect(_COUNT_CHOICES, value=_AUTO, allow_blank=False, id="remote")
                yield Static(
                    "Auto-detect works; exact counts label speakers better.", classes="hint"
                )
            yield Static("Language", classes="field-label")
            yield FormSelect(_LANGUAGE_CHOICES, value="auto", allow_blank=False, id="language")
            yield Static("Title (optional; used by notes)", classes="field-label")
            yield Input(placeholder="e.g. Weekly sync", id="title")
            with Horizontal(classes="switch-row"):
                yield Switch(value=standing.output.record_audio is True, id="record")
                yield Static("Keep the audio recording (audio.wav)")
            with Horizontal(classes="switch-row"):
                yield Switch(value=standing.notes.auto is True, id="notes")
                yield Static("Generate notes after the meeting")
            with Horizontal(id="actions"):
                yield Button("Start", variant="success", id="go")
                yield Button("Back", id="back")
        yield Footer()

    def _standing_settings(self) -> Settings:
        """The settings the form's switches start from.

        The standing switches are off by default; only ``[speakers]
        diarization = true``, ``[output] record_audio = true`` and ``[notes]
        auto = true`` pre-set theirs. A switch beats the settings the same way a
        CLI flag does, for this one meeting. A broken settings file behaves like
        the defaults — :meth:`_submit` is where it is reported.
        """
        from stenograf.settings import Settings, SettingsError, load_settings

        try:
            return load_settings()
        except SettingsError:
            return Settings()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "diarize":
            self.query_one("#counts").display = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._submit()
        elif event.button.id == "back":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        """Resolve form + settings into a MeetingRequest; stay open on any error."""
        # The CLI's resolution seams, reused so both entries share one source of
        # defaults (the C7/thin-client rule): load_settings for the tables,
        # _collect_terms for the [vocab] glossary/attendee baseline.
        from click import ClickException

        from stenograf.cli.run import _collect_terms
        from stenograf.settings import SettingsError, load_settings

        try:
            settings = load_settings()
        except SettingsError as exc:
            self._error(str(exc))
            return
        try:
            glossary_terms, attendee_names = _collect_terms((), None, (), vocab=settings.vocab)
        except ClickException as exc:  # e.g. a stale [vocab] glossary_file path
            self._error(exc.message)
            return

        diarize = self.query_one("#diarize", Switch).value
        language = self.query_one("#language", Select).value
        title = self.query_one("#title", Input).value

        def count(source_id: str, count_id: str) -> int | None:
            """The profile count a source's controls mean: 0 = source off,
            1 = capture as one speaker, None = diarize and estimate."""
            if not self.query_one(f"#{source_id}", Switch).value:
                return 0
            if not diarize:
                return 1
            value = self.query_one(f"#{count_id}", Select).value
            # Not an int = Select.BLANK (never happens: allow_blank is off);
            # _AUTO is the real "estimate the count" sentinel.
            return value if isinstance(value, int) and value != _AUTO else None

        try:
            profile = MeetingProfile(
                language=None if language == "auto" else Language(language),
                local_speakers=count("mic", "local"),
                remote_speakers=count("system", "remote"),
                glossary=glossary_terms,
                attendee_names=attendee_names,
                title=title,
            )
        except ValueError as exc:  # e.g. both sources switched off
            self._error(str(exc))
            return
        self.dismiss(
            MeetingRequest(
                profile=profile,
                settings=settings,
                notes=self.query_one("#notes", Switch).value,
                record_audio=self.query_one("#record", Switch).value,
            )
        )

    def _error(self, message: str) -> None:
        self.notices.append(message)
        self.notify(message, title="Cannot start", severity="error")
