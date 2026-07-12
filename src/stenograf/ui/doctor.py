"""Check setup — ``steno doctor``'s report in a scrollable screen.

Phase 7, Task 5 (PLAN.md §5). The checks run in a ``@work(thread=True)``
worker (they probe helpers, model caches, and — on some platforms — spawn
subprocesses; maintainability rule 1), then the full report renders at once:
``run_checks`` returns a finished list, so there is nothing to stream. The
same ✓/○/✗ vocabulary as the CLI: ○ marks an *optional* check whose failure
doesn't make the machine unhealthy (an opt-in feature it can healthily lack).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Static

if TYPE_CHECKING:
    from collections.abc import Callable

    from stenograf.doctor import Check

_SYMBOL = {True: ("✓", "green"), None: ("○", "yellow"), False: ("✗", "red")}
"""Check rendering: ok → ✓, failed-but-optional → ○, failed → ✗."""


class DoctorScreen(Screen[None]):
    """Machine-readiness report: permissions, OS version, helpers, models."""

    DEFAULT_CSS = """
    DoctorScreen { align: center middle; }
    #panel {
        width: 80; max-width: 95%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #panel-title { text-align: center; text-style: bold; margin: 0 0 1 0; }
    #report { height: auto; }
    #status { color: $text-muted; height: auto; }
    #actions { height: auto; margin: 1 0 0 0; }
    #actions Button { width: 100%; }
    """

    BINDINGS = [Binding("escape", "back", "Back", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []  # plain-text mirror of the rendered report

    def compose(self) -> ComposeResult:
        panel = VerticalScroll(id="panel")
        panel.can_focus = False
        with panel:
            yield Static("Check setup", id="panel-title")
            yield Static(id="report")
            yield Static("running checks…", id="status")
            with Horizontal(id="actions"):
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._run_checks()

    @work(thread=True, exclusive=True)
    def _run_checks(self) -> None:
        from stenograf import doctor

        try:
            checks = doctor.run_checks()
        except Exception as exc:  # noqa: BLE001 — a crashed check must render, not vanish
            self._post(self._set_status, f"doctor failed: {exc}")
            return
        self._post(self._show_report, checks)

    def _post(self, fn: Callable[..., object], *args: object) -> None:
        """Marshal a UI mutation from the worker thread onto the app thread."""
        with contextlib.suppress(Exception):  # screen may have been dismissed
            self.app.call_from_thread(fn, *args)

    # -- UI-thread mutators ----------------------------------------------------

    def _show_report(self, checks: list[Check]) -> None:
        markup_lines = []
        self.lines = []
        for check in checks:
            symbol, color = _SYMBOL[True if check.ok else (None if check.optional else False)]
            self.lines.append(f"{symbol} {check.name}: {check.detail}")
            markup_lines.append(
                f"[{color}]{symbol}[/] [bold]{check.name}[/]: {_escape(check.detail)}"
            )
        self.query_one("#report", Static).update("\n".join(markup_lines))
        problems = sum(1 for c in checks if not (c.ok or c.optional))
        self._set_status(
            "Everything looks good."
            if problems == 0
            else f"{problems} problem(s) found — fix and reopen this screen."
        )

    def _set_status(self, message: str) -> None:
        self.lines.append(message)
        self.query_one("#status", Static).update(_escape(message))

    # -- leaving -----------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.dismiss(None)

    def action_back(self) -> None:
        self.dismiss(None)


def _escape(text: str) -> str:
    """Neutralize [markup] in check details (paths and messages are verbatim)."""
    return text.replace("[", r"\[")
