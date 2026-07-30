"""``steno settings`` — inspect and edit the settings.toml defaults."""

from __future__ import annotations

import click

from stenograf.cli.run import _library_errors


@click.group()
def settings() -> None:
    """Inspect and edit the settings.toml defaults."""


@settings.command("show")
@click.option(
    "--preset",
    "preset",
    metavar="NAME",
    default=None,
    help="Show what a --preset NAME run resolves to: the [meetings.NAME] "
    "overlay applied, with its keys attributed to it.",
)
def settings_show(preset: str | None) -> None:
    """Print the effective configuration and where each value comes from.

    Sources: an environment override, a meeting preset, settings.toml, or the
    built-in default. (CLI flags outrank all four but are per-run, so they
    never appear here.)
    """
    from stenograf.flow import settings_report

    lines, ok = settings_report(preset)
    if not ok:
        click.echo(lines[0])  # the settings-path header still helps
        raise click.ClickException(f"{lines[-1]} — fix it with `steno settings edit`")
    for line in lines:
        click.echo(line)


@settings.command("edit")
def settings_edit() -> None:
    """Open settings.toml in $EDITOR and validate it on save.

    A missing file is first created from a fully commented template, so every
    available key is in front of you. Validation failures keep your edits —
    rerun to fix them.
    """
    from stenograf.settings import SettingsError, ensure_settings_file, load_settings

    path, created = ensure_settings_file()
    if created:
        click.echo(f"created {path}")
    click.edit(filename=str(path))
    try:
        load_settings(path)
    except SettingsError as exc:
        raise click.ClickException(
            f"{exc}\nyour edits are saved — run `steno settings edit` again to fix them"
        ) from exc
    click.echo(f"{path} OK")


@click.command()
@_library_errors
def presets() -> None:
    """List the meeting presets ([meetings.<name>] in settings.toml).

    A preset bundles what a *kind* of meeting sets — title, language,
    vocabulary, notes backend, protocol template — selected per run with
    --preset NAME on start, transcribe and notes.
    """
    from stenograf.settings import load_settings

    settings = load_settings()
    if not settings.meetings:
        click.echo(
            "no meeting presets defined — add a [meetings.<name>] section in "
            "settings.toml (`steno settings edit`; the template shows an example)"
        )
        return
    for name in sorted(settings.meetings):
        summary = settings.meetings[name].summary()
        click.echo(f"  {name}" + (f" — {summary}" if summary else ""))
