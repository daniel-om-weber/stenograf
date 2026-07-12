"""Settings — a read-only rendering of ``steno settings show``.

Phase 7, Task 5 (PLAN.md §5). The screen shows the effective configuration —
every key with its value and where it came from (env override, settings.toml,
built-in default) — through the same ``_settings_rows`` helper the CLI
renders from, so the two can never disagree. Editing happens in $EDITOR via
the *Edit* button: the app suspends, ``click.edit`` runs (creating the file
from the commented template first when missing, exactly like ``steno settings
edit``), and the view re-renders on return.

A settings *form* is where TUI effort balloons — explicitly out of scope
(the plan's scope decision); this screen stays read-only plus the one button.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Static


class SettingsScreen(Screen[None]):
    """Effective configuration, value provenance, and an open-in-$EDITOR button."""

    DEFAULT_CSS = """
    SettingsScreen { align: center middle; }
    #panel {
        width: 80; max-width: 95%; height: auto; max-height: 100%;
        border: round $primary; padding: 1 2;
    }
    #panel-title { text-align: center; text-style: bold; margin: 0 0 1 0; }
    #body { height: auto; max-height: 100%; }
    #actions { height: auto; margin: 1 0 0 0; }
    #actions Button { width: 1fr; }
    #actions #edit { margin: 0 1 0 0; }
    """

    BINDINGS = [Binding("escape", "back", "Back", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []  # plain-text mirror of the rendered view
        self.notices: list[str] = []  # plain-text mirror of the toasts shown

    def compose(self) -> ComposeResult:
        panel = VerticalScroll(id="panel")
        panel.can_focus = False
        with panel:
            yield Static("Settings", id="panel-title")
            # Markup off: setting values are verbatim TOML and may contain
            # [brackets] that Rich markup would eat.
            yield Static(id="body", markup=False)
            with Horizontal(id="actions"):
                yield Button("Edit in $EDITOR", id="edit")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._render_settings()

    def _render_settings(self) -> bool:
        """(Re)build the ``settings show`` text; a broken file renders its error.

        Returns whether the file loaded cleanly (drives the post-edit toast)."""
        from stenograf.cli.settings_cmd import _settings_rows
        from stenograf.settings import SettingsError, load_settings, settings_path

        path = settings_path()
        suffix = "" if path.exists() else " (not present — all defaults)"
        self.lines = [f"settings: {path}{suffix}"]
        try:
            settings = load_settings()
        except SettingsError as exc:
            self.lines += ["", str(exc), "Press Edit to fix the file."]
            ok = False
        else:
            for table, rows in _settings_rows(settings):
                self.lines.append("")
                self.lines.append(f"[{table}]")
                width = max(len(key) for key, _, _ in rows)
                for key, value, source in rows:
                    self.lines.append(f"  {key:<{width}} = {value}  ({source})")
            ok = True
        self.query_one("#body", Static).update("\n".join(self.lines))
        return ok

    # -- actions ---------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit":
            self._edit()
        elif event.button.id == "back":
            self.dismiss(None)

    def action_back(self) -> None:
        self.dismiss(None)

    def _edit(self) -> None:
        """Suspend the app, hand settings.toml to $EDITOR, re-render on return."""
        import click

        from stenograf.cli.settings_cmd import _ensure_settings_file

        path, _created = _ensure_settings_file()
        try:
            with self.app.suspend():
                click.edit(filename=str(path))
        except Exception:  # noqa: BLE001 — headless terminal / suspend unsupported
            self._notice(
                f"Cannot open an editor from here — edit {path} directly.",
                severity="warning",
                timeout=10,
            )
            return
        if self._render_settings():
            self._notice(f"{path} OK", title="Settings")
        else:
            self._notice(
                "settings.toml has a problem — see the message above; your edits are kept.",
                severity="error",
                timeout=10,
            )

    def _notice(self, message: str, **notify_kwargs: object) -> None:
        self.notices.append(message)
        self.notify(message, **notify_kwargs)  # type: ignore[arg-type]
