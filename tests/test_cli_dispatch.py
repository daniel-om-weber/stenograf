"""The entry point itself: bare invocation, display gates, --gui/--tray,
and signal handling (cli/__init__.py)."""

import sys

import pytest
from click.testing import CliRunner

from stenograf import cli, loaders

# -- bare invocation (the desktop-app entry) ----------------------------------


def test_bare_invocation_without_a_tty_prints_help():
    # A pipe/script/cron hitting bare `steno` wants usage text, not a window —
    # the strand-nobody guarantee. CliRunner streams are never TTYs.
    result = CliRunner().invoke(cli.main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "transcribe" in result.output  # the subcommands are listed


def test_bare_invocation_in_an_interactive_terminal_opens_the_app(monkeypatch):
    import stenograf.gui

    calls = []
    monkeypatch.setattr(cli, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli, "_display_available", lambda: True)
    monkeypatch.setattr(stenograf.gui, "run_gui", lambda *, tray: calls.append(tray))

    result = CliRunner().invoke(cli.main, [])

    assert result.exit_code == 0
    assert calls == [False]
    assert "Usage:" not in result.output


def test_bare_invocation_without_a_display_prints_help(monkeypatch):
    # SSH sessions and consoles without a display server: a Qt platform-plugin
    # failure is an uncatchable C++ abort, so the dispatch must never get
    # there — the display gate keeps those sessions on help text.
    import stenograf.gui

    monkeypatch.setattr(cli, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli, "_display_available", lambda: False)
    monkeypatch.setattr(
        stenograf.gui, "run_gui", lambda **kwargs: (_ for _ in ()).throw(AssertionError)
    )

    result = CliRunner().invoke(cli.main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_display_gate_on_linux_wants_a_display_socket(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    for var in ("DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)

    assert not cli._display_available()

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert cli._display_available()


def test_display_gate_on_macos_and_windows_flags_ssh_sessions(monkeypatch):
    # The env fallback (Windows always; macOS when the window-server probe
    # cannot run): a local session always has a window server, so only a
    # remote shell — marked by the SSH variables — is barred.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_macos_aqua_session", lambda: None)
    for var in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(var, raising=False)

    assert cli._display_available()

    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 50000 10.0.0.2 22")
    assert not cli._display_available()


def test_display_gate_on_macos_believes_the_window_server_over_the_env(monkeypatch):
    # tmux servers and sudo both lose SSH_CONNECTION/SSH_TTY (measured
    # 2026-07-30), so the env heuristic false-positives exactly where a Qt
    # launch would be an uncatchable abort. The probe's answer wins in both
    # directions.
    monkeypatch.setattr(sys, "platform", "darwin")
    for var in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(var, raising=False)

    # An SSH login whose environment looks local (the tmux/sudo shape).
    monkeypatch.setattr(cli, "_macos_aqua_session", lambda: False)
    assert not cli._display_available()

    # And the mirror image: a window-server session with stale SSH residue.
    monkeypatch.setattr(cli, "_macos_aqua_session", lambda: True)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 50000 10.0.0.2 22")
    assert cli._display_available()


@pytest.mark.skipif(sys.platform != "darwin", reason="the probe is a macOS API")
def test_the_aqua_probe_gives_an_answer_on_macos():
    # True in a desktop session, False over SSH — either way the probe must
    # resolve on real macOS rather than fall back to the env heuristic.
    assert cli._macos_aqua_session() in (True, False)


def test_the_cli_installs_a_sigterm_handler(monkeypatch):
    # A Linux session-manager logout IS a SIGTERM (and the app-bundle stub
    # forwards one on macOS); the default disposition would kill a run around
    # its finalize. The group callback routes it into the Ctrl-C machinery.
    import signal

    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        CliRunner().invoke(cli.main, [])
        assert signal.getsignal(signal.SIGTERM) is cli._sigterm_becomes_interrupt
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigterm_raises_keyboard_interrupt_but_honors_the_finalize_shield():
    import signal

    # Normally SIGTERM is Ctrl-C: capture ends gracefully, finalize runs.
    with pytest.raises(KeyboardInterrupt):
        cli._sigterm_becomes_interrupt(signal.SIGTERM, None)

    # While session._shield_interrupt has SIGINT ignored, the finalize is the
    # authoritative transcript — SIGTERM must not break in either.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cli._sigterm_becomes_interrupt(signal.SIGTERM, None)  # returns quietly
    finally:
        signal.signal(signal.SIGINT, previous)


def test_gui_flag_opens_the_desktop_app_without_needing_a_tty(monkeypatch):
    # --gui is how the frozen Stenograf.app stub starts the tool: a window
    # needs no TTY and no display heuristic, so neither gate that governs the
    # bare invocation may gate this. CliRunner streams are never TTYs, which
    # is exactly the case tested.
    import stenograf.gui

    calls = []
    monkeypatch.setattr(stenograf.gui, "run_gui", lambda *, tray: calls.append(tray))
    monkeypatch.setattr(cli, "_display_available", lambda: False)

    result = CliRunner().invoke(cli.main, ["--gui"])

    assert result.exit_code == 0
    assert calls == [False]


def test_tray_flag_starts_the_desktop_app_in_the_menu_bar(monkeypatch):
    # The windowless launch, and the one that idles at the
    # wakeup floor. It is a mode of the app, not a second front end.
    import stenograf.gui

    calls = []
    monkeypatch.setattr(stenograf.gui, "run_gui", lambda *, tray: calls.append(tray))

    result = CliRunner().invoke(cli.main, ["--gui", "--tray"])

    assert result.exit_code == 0
    assert calls == [True]


def test_tray_without_gui_is_a_usage_error():
    result = CliRunner().invoke(cli.main, ["--tray"])

    assert result.exit_code != 0
    assert "--gui" in result.output


def test_gui_flag_with_a_subcommand_is_a_usage_error(monkeypatch):
    import stenograf.gui

    monkeypatch.setattr(
        stenograf.gui, "run_gui", lambda **kwargs: (_ for _ in ()).throw(AssertionError)
    )

    result = CliRunner().invoke(cli.main, ["--gui", "profiles", "list"])

    assert result.exit_code != 0
    assert "no subcommand" in result.output


def test_the_desktop_app_says_how_to_repair_a_broken_qt(monkeypatch):
    # PySide6 is a base dependency since the default flip, so an ImportError
    # means a broken or pre-flip install. The likelier shape is Qt *present*
    # but unloadable (missing libEGL, arch-mismatched wheel) — the guard must
    # catch the real import, and the message must be a repair instruction (not
    # the old [gui] extra, which is empty now) carrying the original error.
    import sys

    import click

    from stenograf.gui import run_gui

    monkeypatch.setitem(sys.modules, "stenograf.gui.app", None)  # import → ImportError
    with pytest.raises(click.ClickException) as excinfo:
        run_gui()
    assert "uv tool install --force stenograf" in excinfo.value.message
    assert "[gui]" not in excinfo.value.message
    assert "stenograf.gui.app" in excinfo.value.message  # the original error is kept


def test_subcommands_never_open_the_app(monkeypatch):
    import stenograf.gui

    monkeypatch.setattr(cli, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli, "_display_available", lambda: True)
    monkeypatch.setattr(
        stenograf.gui, "run_gui", lambda **kwargs: (_ for _ in ()).throw(AssertionError)
    )

    result = CliRunner().invoke(cli.main, ["profiles", "list"])

    assert result.exit_code == 0


def test_native_provider_with_announce_never_touches_click(monkeypatch):
    # GUI hosts hand loaders an announce sink because click.echo needs usable
    # stdio the process may not own (a Windows ``pythonw`` launch); with a
    # sink given, the seam must not touch click.
    import click

    from stenograf.config import MeetingProfile
    from stenograf.session import plan_channels

    def boom(*args, **kwargs):
        raise AssertionError("click must not be used when announce is given")

    monkeypatch.setattr(click, "echo", boom)
    monkeypatch.setattr(click, "secho", boom)

    class FakeProvider:
        def __init__(self, *, on_log=None, mic_device=None):
            self.on_log = on_log
            self.mic_device = mic_device

    lines = []
    plans = plan_channels(MeetingProfile(local_speakers=1, remote_speakers=1))
    provider = loaders._native_provider(
        FakeProvider,
        lambda channels, mic_device=None: {ch: f"dev-{ch.value}" for ch in sorted(channels)},
        plans,
        lines.append,
    )
    assert isinstance(provider, FakeProvider)
    assert lines == ["capture: mic ← dev-mic", "capture: system ← dev-system"]


def test_the_macos_provider_preflights_only_when_a_microphone_is_pinned(monkeypatch):
    # macOS has no preflight by default — its helper owns device selection and
    # a subprocess per meeting start would buy nothing. A pinned run is the one
    # case with an answer worth having before the models load: "that microphone
    # is not here".
    import sys as sysmod

    from stenograf.capture import helper as capture_helper
    from stenograf.config import MeetingProfile
    from stenograf.session import plan_channels

    monkeypatch.setattr(sysmod, "platform", "darwin")
    monkeypatch.setattr(capture_helper, "find_helper", lambda: "stenocap")
    asked = []
    monkeypatch.setattr(
        capture_helper,
        "query_devices",
        lambda channels, mic_device=None: (
            asked.append(mic_device),
            {ch: "Fake device" for ch in channels},
        )[1],
    )
    plans = plan_channels(MeetingProfile(local_speakers=1, remote_speakers=1))

    loaders._base_provider(None, plans)
    assert asked == [], "an unpinned macOS run must spawn no device query"

    loaders._base_provider(None, plans, mic_device="usb-1")
    assert asked == ["usb-1"]


def test_a_run_without_a_microphone_carries_no_pin(monkeypatch):
    # A system-audio-only meeting has no mic channel to pin, and passing one
    # would cost macOS a preflight subprocess for a channel that will not exist.
    import sys as sysmod

    from stenograf.capture import helper as capture_helper
    from stenograf.config import MeetingProfile
    from stenograf.session import plan_channels

    monkeypatch.setattr(sysmod, "platform", "darwin")
    monkeypatch.setattr(capture_helper, "find_helper", lambda: "stenocap")
    monkeypatch.setattr(
        capture_helper,
        "query_devices",
        lambda channels, mic_device=None: (_ for _ in ()).throw(
            AssertionError("no device query belongs on a system-only run")
        ),
    )
    plans = plan_channels(MeetingProfile(local_speakers=0, remote_speakers=1))

    provider = loaders._base_provider(None, plans, mic_device="usb-1")
    assert provider is not None
