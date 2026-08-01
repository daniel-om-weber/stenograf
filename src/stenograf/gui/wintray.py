"""The Windows status icon, registered under a stable GUID instead of a path.

``QSystemTrayIcon`` works on Windows, and this module exists for one thing it
cannot express: the ``guidItem`` field of ``NOTIFYICONDATA``. Qt owns the struct
it hands to ``Shell_NotifyIcon`` and offers no way to put a GUID in it, and a
GUID can only be set when the icon is *added* — there is no ``NIM_MODIFY`` that
adopts one later. So the icon is registered by hand here, and Qt's own
implementation stays the path everywhere else (:func:`stenograf.gui.tray.install`
picks between them).

**Why the GUID is worth a module.** Without one, Windows files a notification
icon under the *executable path* of the process that added it. Ours is the
Python interpreter — measured on this machine as
``…\\uv\\python\\cpython-3.13.14-windows-x86_64-none\\pythonw.exe`` — so three
things go wrong that no amount of AppUserModelID fixes (the shell does not
consult the AUMID for tray identity):

- the icon's settings, above all whether the user has dragged it out of the
  overflow, are attached to *pythonw* and shared with every other Python app;
- the interpreter path carries its version, so a 3.13.14 → 3.13.15 upgrade is a
  new identity and silently discards that choice;
- an install moved between uv, pipx and a venv is a new identity every time.

With ``NIF_GUID`` the identity is :data:`STATUS_ICON_GUID` and none of that
applies. The documented caveat — a GUID is bound to the binary that registered
it, so moving the binary stops the icon appearing — was **measured not to bite
on Windows 11 26200** (2026-08-01): the same GUID re-registered from a second
copy of the same executable at a different path returned success, reused the one
settings key, and simply updated its ``ExecutablePath``. The documented recovery
(``NIM_DELETE`` the stale registration, then add again) is implemented anyway in
:meth:`WindowsStatusIcon._add`, because older builds do fail that way and the
cost of carrying it is four lines.

**Windows 11 hides every new tray icon**, which is the defect that started this:
a status item the user cannot see is not a status item, and on macOS — where a
menu-bar item is simply always visible — the equivalent question does not exist.
There is no API to promote an icon, but the shell reads the choice out of
``HKCU\\Control Panel\\NotifyIconSettings`` and honours a change *live*, with no
Explorer restart (measured the same day). :func:`promote` writes it exactly
once: when the value is absent, meaning nobody has decided yet. A user who later
hides the icon leaves ``IsPromoted=0`` behind — an explicit answer this code
never overwrites.

Everything below is ctypes against ``shell32``/``user32``/``gdi32``, in the same
spirit as :mod:`stenograf.winlink`: one file written by hand beats a dependency
(``pywin32``, and a wheel) for a struct and five calls.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal
from PySide6.QtGui import QCursor, QImage
from PySide6.QtWidgets import QSystemTrayIcon

if TYPE_CHECKING:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QMenu

if sys.platform != "win32":  # and it narrows the rest to win32 for the type checker
    raise ImportError("stenograf.gui.wintray drives the Windows shell; import it behind a guard")

STATUS_ICON_GUID = "{014cd6ed-6711-4166-81b3-15ab8141ede5}"
"""This app's notification-icon identity, forever.

Generated once and hard-coded on purpose: the whole point is that it does not
change when the interpreter, the install method or the version does. Changing it
is equivalent to shipping a brand-new tray icon — the user's show/hide choice
resets and the old registry entry is orphaned — so it must not be regenerated.
"""

_TIP_LIMIT = 128
_INFO_LIMIT = 256
_INFO_TITLE_LIMIT = 64
"""``szTip`` / ``szInfo`` / ``szInfoTitle`` capacities, in ``WCHAR``. The struct
is fixed-size, so anything longer is truncated on the way in rather than
overflowing (ctypes would raise, which is not worth a crash over a tooltip)."""

_NIM_ADD, _NIM_MODIFY, _NIM_DELETE, _NIM_SETVERSION = 0, 1, 2, 4
_NIF_MESSAGE, _NIF_ICON, _NIF_TIP, _NIF_INFO = 0x01, 0x02, 0x04, 0x10
_NIF_GUID, _NIF_SHOWTIP = 0x20, 0x80
_NOTIFYICON_VERSION_4 = 4
"""Opts into the modern callback shape: the event is in ``LOWORD(lParam)`` and
``wParam`` carries screen coordinates the shell has already made DPI-correct."""

_NIIF_NONE = 0x00

_CALLBACK_MESSAGE = 0x0400 + 17  # WM_APP + n; ours alone, this window has no other use
_WM_DESTROY = 0x0002
_WM_NULL = 0x0000
_WM_CONTEXTMENU = 0x007B
_NIN_SELECT = 0x0400  # WM_USER + 0, under version 4: a plain click on the icon

_SM_CXSMICON = 49
_DIB_RGB_COLORS = 0
_BI_RGB = 0

# use_last_error, or GetLastError() is whatever some unrelated call left behind
# and the diagnostics below are fiction.
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_shell32 = ctypes.WinDLL("shell32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_LRESULT = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(
    _LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


def _bind(function, restype, *argtypes: type) -> None:  # noqa: ANN001 — a ctypes _FuncPtr
    function.restype = restype
    function.argtypes = list(argtypes)


# **Every** call is declared, not just the ones returning a handle. An undeclared
# ctypes function passes Python ints as C ``int``, so any 64-bit handle above
# 2 GB raises "int too long to convert" — and which handles land up there is
# luck, so the bug reproduces on one launch in ten (it did: 2026-08-01, a
# DeleteObject on an HBITMAP that had been fine the run before).
_bind(
    _user32.DefWindowProcW, _LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)
_bind(_user32.RegisterClassW, wintypes.WORD, ctypes.c_void_p)
_bind(_user32.RegisterWindowMessageW, wintypes.UINT, wintypes.LPCWSTR)
_bind(
    _user32.CreateWindowExW,
    wintypes.HWND,
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    ctypes.c_void_p,
)
_bind(_user32.DestroyWindow, wintypes.BOOL, wintypes.HWND)
_bind(_user32.SetForegroundWindow, wintypes.BOOL, wintypes.HWND)
_bind(
    _user32.PostMessageW,
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
_bind(_user32.GetSystemMetrics, ctypes.c_int, ctypes.c_int)
_bind(_user32.CreateIconIndirect, wintypes.HICON, ctypes.c_void_p)
_bind(_user32.DestroyIcon, wintypes.BOOL, wintypes.HICON)
_bind(
    _gdi32.CreateDIBSection,
    wintypes.HBITMAP,
    wintypes.HDC,
    ctypes.c_void_p,
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
)
_bind(
    _gdi32.CreateBitmap,
    wintypes.HBITMAP,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.c_void_p,
)
_bind(_gdi32.DeleteObject, wintypes.BOOL, wintypes.HGDIOBJ)
_bind(_shell32.Shell_NotifyIconW, wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p)
_bind(_kernel32.GetModuleHandleW, wintypes.HMODULE, wintypes.LPCWSTR)
_bind(
    _kernel32.GetModuleFileNameW, wintypes.DWORD, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD
)


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )


class _NotifyIconData(ctypes.Structure):
    """``NOTIFYICONDATAW``, Vista layout.

    The ``uTimeout``/``uVersion`` union is written as ``uVersion`` alone: they
    are the same four bytes and only the version half is ever set here (balloon
    timeouts have been ignored by the shell since Vista).
    """

    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * _TIP_LIMIT),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * _INFO_LIMIT),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * _INFO_TITLE_LIMIT),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", _GUID),
        ("hBalloonIcon", wintypes.HICON),
    )


class _WndClass(ctypes.Structure):
    _fields_ = (
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    )


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


class _BitmapInfo(ctypes.Structure):
    _fields_ = (("bmiHeader", _BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3))


class _IconInfo(ctypes.Structure):
    _fields_ = (
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HBITMAP),
        ("hbmColor", wintypes.HBITMAP),
    )


def _guid(text: str) -> _GUID:
    result = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(text, ctypes.byref(result))
    return result


def _hicon(icon: QIcon, size: int) -> int:
    """Rasterize ``icon`` at ``size`` px into a Win32 ``HICON``.

    Qt 6 dropped ``QPixmap::toWinHICON`` with QtWinExtras, so the conversion is
    done here: a top-down 32-bit DIB (negative ``biHeight``) whose pixels are
    exactly ``QImage::Format_ARGB32``'s — BGRA in memory on little-endian, which
    is what a ``BI_RGB`` DIB expects — plus the 1-bit mask ``CreateIconIndirect``
    insists on and then ignores, because a 32-bit colour bitmap carries its own
    alpha.

    The caller owns the result and must ``DestroyIcon`` it; the two bitmaps are
    the icon's copies once it exists and are released here.
    """
    image = icon.pixmap(size, size).toImage().convertToFormat(QImage.Format.Format_ARGB32)
    width, height = image.width(), image.height()

    info = _BitmapInfo()
    info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
    info.bmiHeader.biWidth = width
    info.bmiHeader.biHeight = -height  # top-down, like QImage's scanline order
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = _BI_RGB

    bits = ctypes.c_void_p()
    colour = _gdi32.CreateDIBSection(
        None, ctypes.byref(info), _DIB_RGB_COLORS, ctypes.byref(bits), None, 0
    )
    if not colour or bits.value is None:
        raise OSError("could not allocate the status icon's bitmap")
    try:
        # QImage may pad its scanlines, so copy row by row rather than in one go.
        stride = image.bytesPerLine()
        source = bytes(image.constBits())
        for row in range(height):
            ctypes.memmove(
                bits.value + row * width * 4,
                source[row * stride : row * stride + width * 4],
                width * 4,
            )
        mask = _gdi32.CreateBitmap(width, height, 1, 1, None)
        if not mask:
            raise OSError("could not allocate the status icon's mask")
        try:
            icon_info = _IconInfo(fIcon=True, xHotspot=0, yHotspot=0, hbmMask=mask, hbmColor=colour)
            handle = _user32.CreateIconIndirect(ctypes.byref(icon_info))
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            return handle
        finally:
            _gdi32.DeleteObject(mask)
    finally:
        _gdi32.DeleteObject(colour)


_WINDOWS: dict[int, WindowsStatusIcon] = {}
"""Every live icon's window, so one shared WNDPROC can route to the right one.

The alternative — a bound method per instance in the window class — looks
simpler and is wrong twice: a window class may only be registered once per
process, so the *second* icon's window would silently dispatch into the *first*
icon (caught by a test whose only sin was running second, 2026-08-01), and the
class would go on holding a ctypes callback into whichever instance was
collected first.
"""

_CLASS_NAME = "StenografStatusIconWindow"
_registered = False


def _dispatch(hwnd: int, message: int, wparam: int, lparam: int) -> int:
    icon = _WINDOWS.get(hwnd)
    if icon is not None and icon._wndproc(message, wparam, lparam):
        return 0
    return _user32.DefWindowProcW(hwnd, message, wparam, lparam)


_DISPATCH = _WNDPROC(_dispatch)
"""Module-level, and referenced forever: a ctypes callback is freed with its
last Python reference, and the window class would then call into nothing."""


_TASKBAR_CREATED = _user32.RegisterWindowMessageW("TaskbarCreated")
"""The broadcast Explorer sends after a restart, so icons can re-add themselves.

Registered at import: the call is a lookup in a system-wide atom table, it
cannot fail meaningfully, and every process asking for this name gets the same
number back."""


def _make_window() -> int:
    """A hidden, ordinary window to receive an icon's callbacks.

    Ordinary rather than message-only (``HWND_MESSAGE``): the context menu needs
    ``SetForegroundWindow`` to dismiss itself when the user clicks away, and the
    foreground rules ignore a message-only window.
    """
    global _registered
    instance = _kernel32.GetModuleHandleW(None)
    if not _registered:
        cls = _WndClass()
        cls.lpfnWndProc = _DISPATCH
        cls.hInstance = instance
        cls.lpszClassName = _CLASS_NAME
        if not _user32.RegisterClassW(ctypes.byref(cls)):
            raise ctypes.WinError(ctypes.get_last_error())
        _registered = True
    hwnd = _user32.CreateWindowExW(
        0, _CLASS_NAME, "Stenograf", 0, 0, 0, 0, 0, None, None, instance, None
    )
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    return hwnd


class WindowsStatusIcon(QObject):
    """A notification-area icon with a stable identity.

    Deliberately shaped like the slice of ``QSystemTrayIcon`` that
    :class:`stenograf.gui.tray.Tray` uses — :meth:`setIcon`, :meth:`setToolTip`,
    :meth:`setContextMenu`, :meth:`show`, :meth:`showMessage` and an
    :attr:`activated` signal carrying Qt's own ``ActivationReason`` — so the
    tray code is written once against one shape and neither platform's branch
    reads as the special case.
    """

    activated = Signal(QSystemTrayIcon.ActivationReason)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._menu: QMenu | None = None
        self._icon_handle = 0
        self._tip = ""
        self._wanted = False  # show() was asked for — not that the shell agreed
        self._added = False  # the shell is currently carrying our icon
        self._promoted = False
        self._hwnd = _make_window()
        _WINDOWS[self._hwnd] = self
        # `deleteLater` is not virtual in C++, so the override below runs only
        # when *Python* calls it — and on the normal path nothing does: the icon
        # is a child QObject reaped by its parent's destruction at exit, where
        # PySide never routes back through Python (verified 2026-08-01). Without
        # this hop the NIM_DELETE is never sent and a quit leaves a dead icon in
        # the notification area, which the shell only reaps when the user happens
        # to hover it. `QSystemTrayIcon` does the same work in its destructor.
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._teardown)

    # -- the window behind the icon ----------------------------------------

    def _wndproc(self, message: int, wparam: int, lparam: int) -> bool:
        """Handle one message for this icon's window; ``False`` to pass it on."""
        if message == _CALLBACK_MESSAGE:
            self._callback(lparam & 0xFFFF)
            return True
        if message == _TASKBAR_CREATED and self._wanted:
            # Explorer restarted and every icon on the old taskbar went with it.
            # Gated on `_wanted` rather than `_added`, because the case this is
            # most needed for is the one where the add *never took*: a login item
            # runs while the shell is still starting, `NIM_ADD` is refused, and
            # this broadcast is Explorer saying "the taskbar exists now, re-add
            # yourself". On `_added` that recovery would be skipped precisely
            # when it is the only one left, and the session would run iconless.
            self._added = False
            if self._add():
                self.promote()  # a no-op unless the first attempt never got there
            return True
        if message == _WM_DESTROY:
            self._remove()
        return False

    def _callback(self, event: int) -> None:
        if event == _WM_CONTEXTMENU:
            self._popup()
        elif event == _NIN_SELECT:
            self.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)

    def _popup(self) -> None:
        """Show the context menu where the pointer is.

        ``SetForegroundWindow`` first is the long-standing shell requirement: a
        menu raised by a background window never receives the click that should
        dismiss it and stays on screen. The ``WM_NULL`` afterwards is the other
        half of the same fix — it gives the message loop one more turn, without
        which the menu can survive its own dismissal.
        """
        if self._menu is None:
            return
        _user32.SetForegroundWindow(self._hwnd)
        self._menu.popup(QCursor.pos())
        _user32.PostMessageW(self._hwnd, _WM_NULL, 0, 0)

    # -- the icon ----------------------------------------------------------

    def _data(self, flags: int) -> _NotifyIconData:
        """A ``NOTIFYICONDATA`` carrying our identity and nothing stale.

        ``uID`` is left at 0 and never used: with ``NIF_GUID`` the GUID *is* the
        identity, and passing both invites the two to disagree.
        """
        data = _NotifyIconData()
        data.cbSize = ctypes.sizeof(_NotifyIconData)
        data.hWnd = self._hwnd
        data.uFlags = flags | _NIF_GUID
        data.guidItem = _guid(STATUS_ICON_GUID)
        return data

    def _add(self) -> bool:
        data = self._data(_NIF_MESSAGE | _NIF_ICON | _NIF_TIP | _NIF_SHOWTIP)
        data.uCallbackMessage = _CALLBACK_MESSAGE
        data.hIcon = self._icon_handle
        data.szTip = self._tip[: _TIP_LIMIT - 1]
        if not _shell32.Shell_NotifyIconW(_NIM_ADD, ctypes.byref(data)):
            # Documented failure: this GUID is still registered to an executable
            # at another path. Not reproducible on Windows 11 26200 (the add
            # simply succeeds and the path is updated), but older builds refuse
            # until the stale registration is dropped.
            _shell32.Shell_NotifyIconW(_NIM_DELETE, ctypes.byref(data))
            if not _shell32.Shell_NotifyIconW(_NIM_ADD, ctypes.byref(data)):
                return False
        version = self._data(0)
        version.uVersion = _NOTIFYICON_VERSION_4
        _shell32.Shell_NotifyIconW(_NIM_SETVERSION, ctypes.byref(version))
        self._added = True
        return True

    def _modify(self, flags: int, **fields: object) -> None:
        if not self._added:
            return
        data = self._data(flags)
        for name, value in fields.items():
            setattr(data, name, value)
        _shell32.Shell_NotifyIconW(_NIM_MODIFY, ctypes.byref(data))

    def _remove(self) -> None:
        if self._added:
            _shell32.Shell_NotifyIconW(_NIM_DELETE, ctypes.byref(self._data(0)))
            self._added = False
        if self._icon_handle:
            _user32.DestroyIcon(self._icon_handle)
            self._icon_handle = 0

    # -- the QSystemTrayIcon-shaped surface --------------------------------

    def setIcon(self, icon: QIcon) -> None:  # noqa: N802 — mirrors QSystemTrayIcon
        """Draw ``icon`` into the notification area, or keep the one already there.

        A GDI failure must not escape: this is called from ``Tray.__init__``,
        where a raise would abort ``install()`` over artwork, and again as a slot
        on the meeting's ``changed`` signal, where PySide terminates the process
        on an unhandled exception. Under exhausted GDI handles the stale icon is
        a far better outcome than either.
        """
        try:
            handle = _hicon(icon, _user32.GetSystemMetrics(_SM_CXSMICON))
        except OSError as exc:
            print(f"could not draw the Stenograf status icon ({exc})", file=sys.stderr)
            return
        previous = self._icon_handle
        self._icon_handle = handle
        self._modify(_NIF_ICON, hIcon=self._icon_handle)
        if previous:  # after the shell has taken the new one, never before
            _user32.DestroyIcon(previous)

    def setToolTip(self, tip: str) -> None:  # noqa: N802 — mirrors QSystemTrayIcon
        self._tip = tip
        self._modify(_NIF_TIP | _NIF_SHOWTIP, szTip=tip[: _TIP_LIMIT - 1])

    def setContextMenu(self, menu: QMenu) -> None:  # noqa: N802 — mirrors QSystemTrayIcon
        self._menu = menu

    def show(self) -> bool:
        """Put the icon in the notification area — and make it visible there.

        ``False`` means the shell refused the icon outright, which
        ``QSystemTrayIcon.show`` has no way of telling anyone; the caller uses it
        to fall back (:meth:`stenograf.gui.tray.Tray._show_icon`), since a status
        item nobody can see is how the app disappears entirely in tray mode. The
        intent is remembered either way, so the next ``TaskbarCreated`` broadcast
        tries again on its own.
        """
        self._wanted = True
        if not self._add():
            return False
        self.promote()
        return True

    def showMessage(  # noqa: N802 — mirrors QSystemTrayIcon
        self, title: str, message: str, _icon: object = None, _timeout: int = 10000
    ) -> None:
        """Raise a notification from this icon.

        The signature keeps ``QSystemTrayIcon``'s trailing icon and timeout
        arguments so callers need no branch; both are ignored, as the shell has
        ignored the timeout since Vista and takes the balloon's artwork from the
        icon itself.
        """
        self._modify(
            _NIF_INFO,
            szInfo=message[: _INFO_LIMIT - 1],
            szInfoTitle=title[: _INFO_TITLE_LIMIT - 1],
            dwInfoFlags=_NIIF_NONE,
        )

    def promote(self) -> None:
        """Ask the shell to show this icon on the taskbar rather than hide it.

        **Explorer writes the icon's settings key asynchronously**, some way
        after ``NIM_ADD`` returns — measured as not-yet-there immediately after
        the add, and present a few seconds later (2026-08-01). So this retries
        on a widening schedule rather than asking once, and stops the moment the
        question is answered either way. Every attempt runs on the GUI thread
        through ``QTimer``: the work is three registry calls, and a thread for
        that would cost more than it saves.
        """
        if self._promoted:
            return
        self._promoted = True
        self._attempt_promotion(0)

    def _attempt_promotion(self, attempt: int) -> None:
        if promote() != "missing" or attempt >= len(_PROMOTION_RETRIES_MS):
            return
        QTimer.singleShot(
            _PROMOTION_RETRIES_MS[attempt], lambda: self._attempt_promotion(attempt + 1)
        )

    def _teardown(self) -> None:
        """Give the icon and its window back to the shell; idempotent.

        Reached from ``aboutToQuit`` (the normal exit) and from
        :meth:`deleteLater` (the tray's own fallback, and the tests) — see
        :meth:`__init__` for why the override alone is not enough.
        """
        self._wanted = False
        self._remove()
        if self._hwnd:
            _WINDOWS.pop(self._hwnd, None)
            _user32.DestroyWindow(self._hwnd)
            self._hwnd = 0

    def deleteLater(self) -> None:  # noqa: N802 — QObject's own spelling
        self._teardown()
        super().deleteLater()


_PROMOTION_RETRIES_MS = (250, 750, 1500, 3000, 6000)
"""Backoff for :meth:`WindowsStatusIcon._attempt_promotion`, ~11 s in total.

Long enough to outlast a busy Explorer, short enough that a user watching the
launch sees the icon appear rather than arrive later. Giving up is not a
failure: the icon is in the overflow, exactly where it would have been before
any of this."""


def promote() -> str:
    """Make this app's status icon visible, if the user has not already decided.

    Windows 11 puts every newly seen notification icon in the overflow flyout,
    and offers no API to come out of it — the choice lives in
    ``HKCU\\Control Panel\\NotifyIconSettings\\<key>``, one key per icon, keyed
    by a hash the shell computes and does not publish. So the key is *found*
    rather than derived: ours is the one whose ``ExecutablePath`` is this
    process's image and which carries no ``UID`` value, because a
    GUID-registered icon stores no id (measured 2026-08-01; path-registered
    icons, including the one Qt used to add, always do).

    Writing ``IsPromoted=1`` takes effect within seconds and needs no Explorer
    restart. It is written **only when the value is absent**, i.e. when nobody
    has answered the question yet: Windows stores an explicit ``0`` when the
    user hides an icon through Settings, and that answer is theirs to keep.

    Returns ``"promoted"`` when the value was written, ``"answered"`` when one
    was already there, and ``"missing"`` when no key is unambiguously ours — the
    shell has not written it yet (worth asking again about, and the reason for
    the retries), or two of them match and :func:`_find_icon_key` refuses to
    guess. Never raises: an icon in the overflow is a worse app, a failed launch
    is a broken one.
    """
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Control Panel\NotifyIconSettings"
        ) as settings:
            return promote_under(winreg, settings, _process_image_paths())
    except OSError:
        return "missing"


def promote_under(winreg, settings, images: set[str]) -> str:  # noqa: ANN001 — stdlib module
    """:func:`promote`'s decision, against an already-open settings key.

    Split out so the rule can be tested against a scratch key with the real
    ``winreg`` — the alternative being a fake registry, which would only prove
    the fake behaves as expected.
    """
    key_name = _find_icon_key(winreg, settings, images)
    if key_name is None:
        return "missing"
    with winreg.OpenKey(settings, key_name, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            winreg.QueryValueEx(key, "IsPromoted")
        except FileNotFoundError:
            winreg.SetValueEx(key, "IsPromoted", 0, winreg.REG_DWORD, 1)
            return "promoted"
        return "answered"  # by the user, or by an earlier run


def _process_image_paths() -> set[str]:
    """Every spelling of this process's executable the shell might have recorded.

    ``GetModuleFileNameW(NULL)`` rather than ``sys.executable``, which on a venv
    install is the launcher rather than the running image. But the raw answer is
    not enough either: uv installs its interpreters behind a version junction
    (``cpython-3.13-…`` → ``cpython-3.13.14-…``) and Explorer records the
    *resolved* target while ``GetModuleFileNameW`` returns the path we were
    launched by (measured 2026-08-01 — this mismatch is why the first
    implementation found nothing). Both are returned, normcased, and a key
    matching either is ours.
    """
    buffer = ctypes.create_unicode_buffer(32768)
    _kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    module = buffer.value
    return {os.path.normcase(module), os.path.normcase(os.path.realpath(module))}


def _find_icon_key(winreg, settings, images: set[str]) -> str | None:  # noqa: ANN001 — stdlib module
    """The ``NotifyIconSettings`` subkey holding our icon's settings, if any.

    The path-keyed entry Qt left behind on an earlier version is skipped by the
    ``UID`` rule, so on an upgraded machine this still resolves to ours.

    **An ambiguous answer is no answer.** The key stores no GUID, so two
    GUID-registered icons hosted by the same interpreter are indistinguishable
    here — and a shared interpreter path is the entire reason this module exists.
    Picking one would write ``IsPromoted`` into a stranger's key, changing a
    setting for an app we do not own, while leaving ours in the overflow anyway.
    ``None`` at least keeps the damage to our own icon.
    """
    found = None
    index = 0
    while True:
        try:
            name = winreg.EnumKey(settings, index)
        except OSError:
            break
        index += 1
        try:
            with winreg.OpenKey(settings, name) as key:
                path, _ = winreg.QueryValueEx(key, "ExecutablePath")
                if not isinstance(path, str) or os.path.normcase(path) not in images:
                    continue
                try:
                    winreg.QueryValueEx(key, "UID")
                    continue  # a path-registered icon, i.e. not ours
                except FileNotFoundError:
                    pass
        except OSError:
            continue
        if found is not None:
            return None
        found = name
    return found


__all__ = ["STATUS_ICON_GUID", "WindowsStatusIcon", "promote", "promote_under"]
