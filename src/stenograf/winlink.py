"""Windows shell links (``.lnk``) through COM, with ctypes and nothing else.

What ``steno setup`` needs from a Windows launcher is one thing a batch file
cannot give it: an **AppUserModelID**. The shell matches a running window to the
shortcut that launched it by that string alone, and that match is what makes the
window group under its own taskbar button, what makes a pinned shortcut light up
as the running app instead of starting a second one, and what lets a toast be
attributed to *Stenograf* rather than to ``pythonw.exe``. It lives in the link's
**property store**, which is why this module talks to ``IShellLink`` directly:
the usual PowerShell detour through ``WScript.Shell`` can set a target and an
icon but has no way to reach ``IPropertyStore``, so a link written that way
would look right and match nothing. VBScript would reach it and is rejected for
a different reason — Windows is in the middle of removing it.

**Why raw vtable dispatch.** A ``.lnk`` is a COM object serialized by
``IPersistFile``; there is no file format to write. ``pywin32`` and ``comtypes``
both do this in a few lines, and neither is worth a dependency (and a wheel) for
one file written once per install — so the three interfaces are driven by hand.
That is less frightening than it looks: a COM interface pointer points at a
pointer to an array of function pointers, so a method call is "read slot *n*,
call it with the interface as the first argument", which is :meth:`_Interface.call`.
The slot numbers below are the interfaces' declaration order and are frozen
forever by COM's own binary-compatibility rule; they are the one thing here that
would be a bug to "clean up".

Failures surface as ``OSError``: every method is bound with a
:data:`ctypes.HRESULT` result type, which makes ctypes raise on any failing
HRESULT rather than returning a number nobody checks. Callers that would rather
degrade than fail — ``steno setup`` writes a batch file instead — catch it.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

if sys.platform != "win32":  # and it narrows the rest to win32 for the type checker
    raise ImportError("stenograf.winlink drives the Windows shell; import it behind a guard")

# -- the COM vocabulary this module needs ------------------------------------

_CLSID_SHELL_LINK = "{00021401-0000-0000-C000-000000000046}"
_IID_SHELL_LINK_W = "{000214F9-0000-0000-C000-000000000046}"
_IID_PERSIST_FILE = "{0000010B-0000-0000-C000-000000000046}"
_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_FMTID_APP_USER_MODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
_PID_APP_USER_MODEL_ID = 5
"""``PKEY_AppUserModel_ID`` — the property the whole module exists to set."""

# Vtable slots, counted from IUnknown's three. Declaration order, fixed by COM.
_QUERY_INTERFACE, _RELEASE = 0, 2
_LINK_GET_PATH, _LINK_GET_DESCRIPTION, _LINK_SET_DESCRIPTION = 3, 6, 7
_LINK_SET_WORKING_DIRECTORY, _LINK_GET_ARGUMENTS, _LINK_SET_ARGUMENTS = 9, 10, 11
_LINK_GET_ICON_LOCATION, _LINK_SET_ICON_LOCATION, _LINK_SET_PATH = 16, 17, 20
_FILE_LOAD, _FILE_SAVE = 5, 6
_STORE_GET_VALUE, _STORE_SET_VALUE, _STORE_COMMIT = 5, 6, 7

_CLSCTX_INPROC_SERVER = 1
_COINIT_APARTMENTTHREADED = 2
_RPC_E_CHANGED_MODE = -2147417850
"""``CoInitializeEx`` refusing to change an already-chosen apartment model."""
_STGM_READ = 0
_VT_LPWSTR = 31
_TEXT_LIMIT = 1024
"""Characters reserved for any string read back — ``INFOTIPSIZE``, which is the
longest a shell link stores. A path may in principle exceed it; a shortcut whose
target does is not one we wrote."""
_FIND_DATA_SIZE = 1024
"""Scratch for the ``WIN32_FIND_DATAW`` ``GetPath`` fills in and we ignore (592
bytes, rounded up). Not optional: the parameter is documented as optional but
shell32 writes through it regardless of what it was handed."""


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    )


class _PropertyKey(ctypes.Structure):
    _fields_ = (("fmtid", _GUID), ("pid", ctypes.c_ulong))


class _PropVariant(ctypes.Structure):
    """As much of a ``PROPVARIANT`` as one string needs.

    The real union is wider, but its size is what matters — the callee writes
    through this pointer — and two pointers past the tag reproduce it exactly on
    both 32- and 64-bit (16 and 24 bytes). The value is kept as a raw address
    rather than a ``c_wchar_p`` on purpose: ctypes would decode a returned
    ``c_wchar_p`` field into a ``str`` and lose the pointer that
    ``PropVariantClear`` has to free.
    """

    _fields_ = (
        ("vt", ctypes.c_ushort),
        ("_reserved", ctypes.c_ushort * 3),
        ("value", ctypes.c_void_p),
        ("_tail", ctypes.c_void_p),
    )


def _guid(text: str) -> _GUID:
    result = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(text, ctypes.byref(result))
    return result


def _property_key(fmtid: str, pid: int) -> _PropertyKey:
    return _PropertyKey(fmtid=_guid(fmtid), pid=pid)


# -- the plumbing ------------------------------------------------------------


def _argument_type(value: object) -> type:
    """Which C type one Python argument crosses the boundary as.

    Every parameter in the four interfaces below is a wide string, a plain int
    or a pointer, so inferring beats spelling out an argtypes tuple per call.
    Buffers and ``byref`` results all go as ``void *``, which is what they are.
    """
    if isinstance(value, str):
        return ctypes.c_wchar_p
    if isinstance(value, int):
        return ctypes.c_int
    return ctypes.c_void_p


class _Interface:
    """One COM interface pointer: vtable dispatch, and a ``Release`` on the way out."""

    def __init__(self, pointer: ctypes.c_void_p) -> None:
        self._pointer = pointer

    def __enter__(self) -> _Interface:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def _slot(self, index: int) -> int:
        # The pointer points at the vtable pointer; the vtable is an array of
        # function pointers. Two dereferences, and the second is the method.
        vtable = ctypes.cast(self._pointer, ctypes.POINTER(ctypes.c_void_p))[0]
        return ctypes.cast(ctypes.c_void_p(vtable), ctypes.POINTER(ctypes.c_void_p))[index]

    def call(self, slot: int, *arguments: object) -> None:
        """Invoke a method, raising ``OSError`` if it returns a failing HRESULT."""
        signature = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, *(_argument_type(a) for a in arguments)
        )
        signature(self._slot(slot))(self._pointer, *arguments)

    def query(self, iid: str) -> _Interface:
        """Ask this object for another of its interfaces."""
        pointer = ctypes.c_void_p()
        self.call(_QUERY_INTERFACE, ctypes.byref(_guid(iid)), ctypes.byref(pointer))
        return _Interface(pointer)

    def text(self, slot: int, *extra: object) -> str:
        """Call a ``Get*(buffer, length, …)`` method and return what it wrote."""
        buffer = ctypes.create_unicode_buffer(_TEXT_LIMIT)
        self.call(slot, buffer, _TEXT_LIMIT, *extra)
        return buffer.value

    def release(self) -> None:
        if not self._pointer:
            return
        # Release returns a refcount, not an HRESULT; binding it as one would
        # make ctypes read every nonzero count as a failure and raise.
        signature = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        signature(self._slot(_RELEASE))(self._pointer)
        self._pointer = ctypes.c_void_p()


@contextmanager
def _apartment() -> Iterator[None]:
    """COM for the duration, and the thread left exactly as it was found."""
    result = ctypes.windll.ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    if result == _RPC_E_CHANGED_MODE:
        # Something else — Qt, a shell extension — already put this thread in
        # the multi-threaded apartment. ShellLink is happy in either, and the
        # apartment is not ours to change or to balance with an uninitialize.
        yield
        return
    if result < 0:
        raise OSError(f"COM could not be initialized (0x{result & 0xFFFFFFFF:08X})")
    try:
        yield
    finally:
        ctypes.windll.ole32.CoUninitialize()


@contextmanager
def _shell_link() -> Iterator[_Interface]:
    """A fresh, empty shell link object, released afterwards."""
    with _apartment():
        pointer = ctypes.c_void_p()
        ctypes.oledll.ole32.CoCreateInstance(
            ctypes.byref(_guid(_CLSID_SHELL_LINK)),
            None,
            _CLSCTX_INPROC_SERVER,
            ctypes.byref(_guid(_IID_SHELL_LINK_W)),
            ctypes.byref(pointer),
        )
        with _Interface(pointer) as link:
            yield link


# -- what the rest of the project calls --------------------------------------


@dataclass(frozen=True)
class Shortcut:
    """A shell link as :func:`read_shortcut` found it on disk."""

    target: str
    arguments: str
    description: str
    icon: str
    app_id: str
    """The AppUserModelID, or ``""`` where the link declares none — which is
    every shortcut not written by this module."""


def write_shortcut(
    path: Path,
    *,
    target: Path | str,
    arguments: str = "",
    working_directory: Path | str = "",
    description: str = "",
    icon: Path | str = "",
    app_id: str = "",
) -> Path:
    """Write ``path`` as a shell link and return it. Raises ``OSError`` on refusal.

    Overwrites whatever was there: a ``.lnk`` carries nothing worth merging, and
    ``steno setup`` is expected to be re-runnable. Empty optional arguments are
    left unset rather than set to an empty string, so a reader can tell "not
    declared" from "declared blank".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _shell_link() as link:
        link.call(_LINK_SET_PATH, str(target))
        if arguments:
            link.call(_LINK_SET_ARGUMENTS, arguments)
        if working_directory:
            link.call(_LINK_SET_WORKING_DIRECTORY, str(working_directory))
        if description:
            link.call(_LINK_SET_DESCRIPTION, description)
        if icon:
            link.call(_LINK_SET_ICON_LOCATION, str(icon), 0)  # index 0: the whole file
        if app_id:
            _set_app_id(link, app_id)
        with link.query(_IID_PERSIST_FILE) as file:
            file.call(_FILE_SAVE, str(path), 1)  # fRemember: adopt this as the link's file
    return path


def read_shortcut(path: Path) -> Shortcut:
    """Read a shell link back. Raises ``OSError`` if it cannot be loaded."""
    with _shell_link() as link:
        with link.query(_IID_PERSIST_FILE) as file:
            file.call(_FILE_LOAD, str(path), _STGM_READ)
        # SLGP_RAWPATH (4) returns what was stored rather than what the shell
        # can resolve today, which is what a "did we write this?" check wants.
        target = link.text(_LINK_GET_PATH, ctypes.create_string_buffer(_FIND_DATA_SIZE), 4)
        return Shortcut(
            target=target,
            arguments=link.text(_LINK_GET_ARGUMENTS),
            description=link.text(_LINK_GET_DESCRIPTION),
            icon=link.text(_LINK_GET_ICON_LOCATION, ctypes.byref(ctypes.c_int())),
            app_id=_get_app_id(link),
        )


def _set_app_id(link: _Interface, app_id: str) -> None:
    with link.query(_IID_PROPERTY_STORE) as store:
        key = _property_key(_FMTID_APP_USER_MODEL, _PID_APP_USER_MODEL_ID)
        # The buffer has to outlive the SetValue call — the variant holds its
        # bare address — and must not be cleared afterwards: SetValue copies
        # what it is given, and this string was never COM-allocated.
        text = ctypes.create_unicode_buffer(app_id)
        variant = _PropVariant(vt=_VT_LPWSTR, value=ctypes.cast(text, ctypes.c_void_p))
        store.call(_STORE_SET_VALUE, ctypes.byref(key), ctypes.byref(variant))
        store.call(_STORE_COMMIT)


def _get_app_id(link: _Interface) -> str:
    with link.query(_IID_PROPERTY_STORE) as store:
        key = _property_key(_FMTID_APP_USER_MODEL, _PID_APP_USER_MODEL_ID)
        variant = _PropVariant()
        store.call(_STORE_GET_VALUE, ctypes.byref(key), ctypes.byref(variant))
        try:
            if variant.vt != _VT_LPWSTR or not variant.value:
                return ""  # VT_EMPTY: a link with no id, i.e. one we did not write
            return ctypes.cast(variant.value, ctypes.c_wchar_p).value or ""
        finally:
            # This one *was* COM-allocated, by the callee.
            ctypes.windll.ole32.PropVariantClear(ctypes.byref(variant))


__all__ = ["Shortcut", "read_shortcut", "write_shortcut"]
