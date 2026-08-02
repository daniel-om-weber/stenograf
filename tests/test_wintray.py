"""The Windows status icon: identity, promotion, and the HICON conversion.

Windows-only, and skipped everywhere else — the module refuses to import off
win32 by design, so there is nothing here to exercise on macOS or Linux.

Two halves. The promotion rule runs against the **real** ``winreg``, pointed at
a scratch key of this test's own instead of the live
``Control Panel\\NotifyIconSettings``: the functions take an open settings key
precisely so this is possible, and a fake registry would only prove the fake
behaves as written. The icon conversion runs against the real Qt, because a
mark that rasterizes to nothing is exactly the failure a mock cannot see —
:func:`stenograf.gui.wintray._hicon` replaces ``QPixmap::toWinHICON``, which
Qt 6 removed, and it is hand-rolled DIB code.

What is *not* tested here is the shell itself: whether an added icon becomes
visible is Explorer's decision, needs a desktop session, and was measured by
hand (2026-08-01) rather than asserted.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the Windows status icon")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if sys.platform == "win32":
    import winreg

    from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
        STATUS_ICON_GUID,
        _find_icon_key,
        _guid,
        _hicon,
        _process_image_paths,
        promote_under,
    )

_SCRATCH = r"Software\stenograf\test-notify-icon-settings"

OURS = r"C:\Python\python.exe"
"""Stands in for this process's image; the tests pass it in explicitly rather
than reading the real one, so they assert the rule and not the machine."""


@pytest.fixture
def settings():
    """A scratch stand-in for ``NotifyIconSettings``, deleted afterwards."""
    winreg.CreateKey(winreg.HKEY_CURRENT_USER, _SCRATCH).Close()
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SCRATCH, 0, winreg.KEY_READ | winreg.KEY_WRITE)
    yield key
    for name in _subkeys(key):
        winreg.DeleteKey(key, name)
    key.Close()
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _SCRATCH)


def _subkeys(key) -> list[str]:
    names, index = [], 0
    while True:
        try:
            names.append(winreg.EnumKey(key, index))
        except OSError:
            return names
        index += 1


def _icon_entry(
    settings, name: str, path: str, uid: int | None = None, promoted: int | None = None
):
    """One ``NotifyIconSettings``-shaped subkey.

    ``uid=None`` is the GUID-registered shape (the shell stores no id for one);
    an int is the path-registered shape Qt produces.
    """
    with winreg.CreateKey(settings, name) as key:
        winreg.SetValueEx(key, "ExecutablePath", 0, winreg.REG_SZ, path)
        if uid is not None:
            winreg.SetValueEx(key, "UID", 0, winreg.REG_DWORD, uid)
        if promoted is not None:
            winreg.SetValueEx(key, "IsPromoted", 0, winreg.REG_DWORD, promoted)


def _promoted(settings, name: str) -> int | None:
    with winreg.OpenKey(settings, name) as key:
        try:
            return winreg.QueryValueEx(key, "IsPromoted")[0]
        except FileNotFoundError:
            return None


class TestFindingOurKey:
    """Which of the shell's icon keys is this app's."""

    def test_the_guid_registered_key_wins_over_the_path_registered_one(self, settings):
        # The shape a machine upgrading from the QSystemTrayIcon version has:
        # Qt's old key and ours, same executable, and only ours carries no UID.
        _icon_entry(settings, "qt-left-this", OURS, uid=0, promoted=1)
        _icon_entry(settings, "ours", OURS)
        assert _find_icon_key(winreg, settings, {os.path.normcase(OURS)}) == "ours"

    def test_another_app_using_the_same_interpreter_is_not_ours(self, settings):
        # The whole reason for the GUID: several Python apps share one
        # interpreter path, and their tray settings must not be confused.
        _icon_entry(settings, "some-other-python-app", OURS, uid=7)
        assert _find_icon_key(winreg, settings, {os.path.normcase(OURS)}) is None

    def test_two_guid_icons_on_one_interpreter_resolve_to_neither(self, settings):
        # No GUID is stored in the key, so a second GUID-registered app hosted by
        # the same pythonw.exe is indistinguishable from ours. Guessing would
        # write IsPromoted into that app's key — changing a setting for software
        # we do not own — and still leave our own icon in the overflow.
        _icon_entry(settings, "ours-or-maybe-not", OURS)
        _icon_entry(settings, "the-other-one", OURS)
        assert _find_icon_key(winreg, settings, {os.path.normcase(OURS)}) is None
        assert promote_under(winreg, settings, {os.path.normcase(OURS)}) == "missing"
        assert _promoted(settings, "the-other-one") is None

    def test_either_spelling_of_the_interpreter_path_matches(self, settings):
        # uv installs its interpreters behind a version junction and the shell
        # records the resolved target, while GetModuleFileNameW returns the path
        # we were launched by. Matching only the latter found nothing at all
        # (measured 2026-08-01) — that bug is what this pins.
        resolved = r"C:\Python\cpython-3.13.14\python.exe"
        _icon_entry(settings, "ours", resolved)
        launched = r"C:\Python\cpython-3.13\python.exe"
        assert _find_icon_key(winreg, settings, {os.path.normcase(launched)}) is None
        both = {os.path.normcase(launched), os.path.normcase(resolved)}
        assert _find_icon_key(winreg, settings, both) == "ours"

    def test_the_path_comparison_ignores_case(self, settings):
        _icon_entry(settings, "ours", r"C:\PYTHON\PYTHON.EXE")
        assert _find_icon_key(winreg, settings, {os.path.normcase(OURS)}) == "ours"

    def test_this_process_offers_both_spellings_of_itself(self):
        paths = _process_image_paths()
        assert paths, "the running image must be findable"
        assert all(path == os.path.normcase(path) for path in paths)
        assert all(os.path.isfile(path) for path in paths)


class TestPromotion:
    """Windows 11 hides new tray icons; this is the one lever that unhides one."""

    def test_an_undecided_icon_is_promoted(self, settings):
        _icon_entry(settings, "ours", OURS)
        assert promote_under(winreg, settings, {os.path.normcase(OURS)}) == "promoted"
        assert _promoted(settings, "ours") == 1

    def test_a_second_launch_leaves_the_value_alone(self, settings):
        _icon_entry(settings, "ours", OURS)
        promote_under(winreg, settings, {os.path.normcase(OURS)})
        assert promote_under(winreg, settings, {os.path.normcase(OURS)}) == "answered"

    def test_a_user_who_hid_the_icon_keeps_it_hidden(self, settings):
        # Windows writes an explicit 0 when the icon is turned off in Settings.
        # Overwriting that would re-show it on every launch, which is worse than
        # the bug this whole module fixes.
        _icon_entry(settings, "ours", OURS, promoted=0)
        assert promote_under(winreg, settings, {os.path.normcase(OURS)}) == "answered"
        assert _promoted(settings, "ours") == 0

    def test_a_key_the_shell_has_not_written_yet_asks_again(self, settings):
        # Explorer creates the key some time after Shell_NotifyIcon returns, so
        # "missing" has to be distinguishable from "answered" — it is what makes
        # WindowsStatusIcon retry instead of giving up on the first look.
        assert promote_under(winreg, settings, {os.path.normcase(OURS)}) == "missing"


@pytest.fixture(scope="module")
def qt_app():
    """The one application object a process may have (``_hicon`` rasterizes)."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def icon(qt_app):
    """A status icon that is never ``show()``n against the real shell.

    So no icon is registered and nothing appears on the developer's own taskbar
    — only the hidden message window exists, which is all these tests need.
    """
    from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
        WindowsStatusIcon,
    )

    item = WindowsStatusIcon()
    yield item
    item.deleteLater()


def _send(icon, message: int, lparam: int = 0) -> None:
    """Post a window message to ``icon``'s window and let Qt's loop deliver it."""
    import ctypes

    from PySide6.QtCore import QCoreApplication, QEventLoop

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW(
        ctypes.c_void_p(icon._hwnd), ctypes.c_uint(message), 0, ctypes.c_ssize_t(lparam)
    )
    app = QCoreApplication.instance()
    assert app is not None
    for _ in range(20):
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)


class TestTheCallback:
    """What the shell tells us a click was.

    Messages are posted to the icon's window directly rather than clicked, and
    Qt's own event loop delivers them: the packing of the event into
    ``LOWORD(lParam)`` is the part worth pinning, since getting it wrong yields
    an icon that is simply inert.
    """

    def _post(self, icon, event: int) -> None:
        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _CALLBACK_MESSAGE,
        )

        # The icon id rides in the high word; a decoder that read the whole
        # lParam would see 0x0001_0400 rather than the event.
        _send(icon, _CALLBACK_MESSAGE, (1 << 16) | event)

    def test_a_click_on_the_icon_is_an_activation(self, icon):
        from PySide6.QtWidgets import QSystemTrayIcon

        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _NIN_SELECT,
        )

        seen = []
        icon.activated.connect(seen.append)
        self._post(icon, _NIN_SELECT)
        assert seen == [QSystemTrayIcon.ActivationReason.Trigger]

    def test_a_second_icon_gets_its_own_clicks(self, qt_app, icon):
        # A window class may be registered only once per process, so binding the
        # WNDPROC to an instance made every later window dispatch into the first
        # icon — which is why one shared WNDPROC routes by hwnd instead. Found
        # by a test that merely ran second (2026-08-01); pinned here on purpose.
        from PySide6.QtWidgets import QSystemTrayIcon

        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _NIN_SELECT,
            WindowsStatusIcon,
        )

        second = WindowsStatusIcon()
        try:
            assert second._hwnd != icon._hwnd
            first_seen, second_seen = [], []
            icon.activated.connect(first_seen.append)
            second.activated.connect(second_seen.append)
            self._post(second, _NIN_SELECT)
            assert second_seen == [QSystemTrayIcon.ActivationReason.Trigger]
            assert first_seen == [], "the click landed on the wrong icon"
        finally:
            second.deleteLater()

    def test_a_right_click_opens_the_menu_and_is_not_an_activation(self, icon):
        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _WM_CONTEXTMENU,
        )

        class StubMenu:
            def __init__(self):
                self.opened_at = []

            def popup(self, point):
                self.opened_at.append(point)

        menu = StubMenu()
        icon.setContextMenu(menu)
        activations = []
        icon.activated.connect(activations.append)
        self._post(icon, _WM_CONTEXTMENU)
        assert len(menu.opened_at) == 1, "the context menu did not open"
        assert activations == [], "a right click must not also open the window"


class TestGettingOnScreenAndOff:
    """The icon's two hard moments: the shell not ready, and the app leaving.

    Both are driven with ``_add`` replaced, so nothing is ever registered with
    the real Explorer and :meth:`promote` never reaches the live registry.
    """

    def test_a_refused_add_is_retried_when_explorer_says_the_taskbar_exists(
        self, icon, monkeypatch
    ):
        # A login item runs while the shell is still starting: NIM_ADD is
        # refused, and Explorer's TaskbarCreated broadcast is the invitation to
        # try again. Gating that on a *successful* add skipped exactly this case
        # and left the session with no icon at all.
        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _TASKBAR_CREATED,
        )

        adds, promotions = [], []

        def add() -> bool:
            adds.append(True)
            return len(adds) > 1

        monkeypatch.setattr(icon, "_add", add)
        monkeypatch.setattr(icon, "promote", lambda: promotions.append(True))

        assert icon.show() is False, "a refused add must be visible to the caller"
        _send(icon, _TASKBAR_CREATED)
        assert len(adds) == 2, "the broadcast did not re-add the icon"
        assert promotions == [True], "the icon came back, but stayed in the overflow"

    def test_the_broadcast_is_ignored_until_someone_asks_for_an_icon(self, icon, monkeypatch):
        # An Explorer restart must not conjure an icon the app never wanted.
        from stenograf.gui.wintray import (  # pyright: ignore[reportAttributeAccessIssue]
            _TASKBAR_CREATED,
        )

        adds = []

        def add() -> bool:
            adds.append(True)
            return True

        monkeypatch.setattr(icon, "_add", add)
        _send(icon, _TASKBAR_CREATED)
        assert adds == []

    def test_the_icon_is_handed_back_when_the_app_quits(self, qt_app, icon):
        # deleteLater is not virtual in C++, so overriding it only catches the
        # calls Python makes — and on the normal path the icon is a child QObject
        # reaped by its parent, which never routes back through Python (measured
        # 2026-08-01). Missing this leaves a dead icon in the notification area.
        assert icon._hwnd
        qt_app.aboutToQuit.emit()
        assert icon._hwnd == 0, "quitting did not release the icon's window"

    def test_a_gdi_failure_keeps_the_icon_already_on_screen(self, icon, monkeypatch, caplog):
        # setIcon runs inside Tray.__init__ and again as a slot on the meeting's
        # changed signal, where PySide terminates the process on an unhandled
        # exception. Losing the new artwork beats losing the app.
        from stenograf.gui import wintray
        from stenograf.gui.tray import _icon as mark

        def out_of_handles(*_args: object) -> int:
            raise OSError("could not allocate the status icon's bitmap")

        icon.setIcon(mark(None))
        established = icon._icon_handle
        assert established, "the first icon never rasterized"

        monkeypatch.setattr(wintray, "_hicon", out_of_handles)
        icon.setIcon(mark("#ff5f56"))
        assert icon._icon_handle == established
        assert "could not draw" in caplog.text


class TestTheMark:
    """The icon the shell is actually handed."""

    @pytest.mark.parametrize("tint", [None, "#ff5f56"])
    def test_the_mark_survives_the_conversion_to_a_win32_icon(self, qt_app, tint):
        import ctypes

        from stenograf.gui.tray import _icon

        handle = _hicon(_icon(tint), 32)
        try:
            assert handle, "CreateIconIndirect returned nothing"
        finally:
            ctypes.WinDLL("user32").DestroyIcon(ctypes.c_void_p(handle))

    def test_the_status_icon_guid_is_a_guid(self):
        # Hand-copied into a string constant, and a malformed one would fail at
        # the first Shell_NotifyIcon rather than here.
        assert _guid(STATUS_ICON_GUID).Data1 == 0x014CD6ED
