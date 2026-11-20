"""The script that replaces an installed stenograf with a newer one.

**This file's name, its two commands and the three files they exchange are a
contract with every older release, and none of them may change.** An app that
wants to update does not run its own copy of this file: it downloads the new
release into uv's cache, runs *that* release's copy, and so a mistake made here
is repaired by the next release instead of staying on users' machines. What an
old app relies on, forever:

1. ``python -m stenograf._updater prepare --out DIR``, run inside a trial copy
   of the new release (``uv tool run --from stenograf==X``). It checks that this
   release can run on this machine, leaves ``DIR/updater.py`` (this file) and
   exits 0, or exits 1 with its last line of output saying why, for the user.
2. ``<base python> -I DIR/updater.py apply DIR/request.json``, started detached
   just before the app quits. :class:`Request` is the request an old app
   writes; a field may be added with a default, never renamed or required.
3. ``result.json`` in the request's ``work_dir`` when it is over: ``format``,
   ``ok``, ``version``, ``message`` (one sentence for the user), ``log``, and
   ``rolled_back`` (the version was installed and taken out again — the only
   failure an app holds against the version). An old app that was rolled back
   shows the sentence and nothing else.
4. What the *apps* of two releases share. The old app's request names the
   work folder and the lock file; the new app, started by this script, works
   them out again by itself (``update.root``, ``update.work_dir``). A release
   that works them out differently holds a lock this script is not looking at
   and is judged dead while it runs — and rolled back from under itself. So
   these are as fixed as the commands: the per-user folder, the subfolder per
   environment, and in it ``in-use.lock`` (held shared by every stenograf
   process), ``started.json`` (``{"version": X}``, written by the new app once
   its window is built, when ``state.json`` carries ``pending``) and
   ``updater.alive``. ``python -m stenograf --version`` ends with the version.
5. What the old app decides, and this script therefore cannot change for it:
   the trial requirement (``stenograf==X``, no extras), the command that starts
   the app again, and which launcher files are saved for a rollback.

``apply`` runs on the interpreter the environment was *built from*, not the one
inside it, because it renames that environment away: standard library only, no
import from stenograf, and nothing newer than Python 3.12.

How the swap works, and why each step is there, is measured in
``eval/update_mechanics.sh``:

- The old environment is renamed aside and a fresh one installed at the same
  path, since every launcher embeds that path. An in-place upgrade under a
  running process leaves it holding two versions at once.
- uv writes its receipt last, so an install that died halfway is a folder with
  no receipt. Whatever stands at the environment path after the old one was
  moved away is therefore this script's own doing, receipt or not, and is
  moved to the trash on the way back, never deleted in the critical path.
- The install runs offline and asks for at least the target version: uv then
  takes the newest version in its cache, which is the one ``prepare`` just ran,
  and fails rather than quietly reinstalling the old one from a pruned cache.
  A pin, constraints or ``--exclude-newer`` would stick in the receipt and
  block every later upgrade.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

REQUEST_FORMAT = 1
RESULT_FORMAT = 1

EXIT_WAIT_S = 90.0
"""How long the app and its launcher get to exit. The app only hands over when
it is idle, so there is no finalize to wait for — this covers Qt teardown."""

LOCK_WAIT_S = 15.0
RENAME_WAIT_S = 60.0
"""Windows refuses to rename a folder while any file in it is open, and a virus
scanner reading a just-written file holds it open for a moment."""

QUIET_S = 10.0
"""How long the lock must stay free before a start counts as dead. A ``steno``
command in a terminal holds the same lock for a moment; without the pause its
exit would read as the app's, while the app is still on its way up."""

INSTALL_WAIT_S = 900.0
START_WAIT_S = 120.0
"""How long the new app gets to report that its window is built. The first
start after an install compiles every module, which is slow on a cold disk."""

NEVER_STARTED_S = 45.0
"""With no process holding the lock after this long, the new app never started."""

REINSTALL_HINT = "uv tool install --force --python 3.13 stenograf"

_SENTINEL = ".stenograf-update-trash"


class LockHeld(OSError):
    """The lock is held in a way that conflicts with the one asked for."""


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    _kernel32.LockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _ERROR_LOCK_VIOLATION = 33
    _ERROR_IO_PENDING = 997
    _SYNCHRONIZE = 0x00100000
    _WAIT_TIMEOUT = 0x102

    def _lock(handle: IO[bytes], *, exclusive: bool) -> None:
        # msvcrt.locking only knows exclusive locks; the shared half needs the
        # Win32 call. One byte at offset 0 is the whole lock.
        flags = _LOCKFILE_FAIL_IMMEDIATELY | (_LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0)
        overlapped = _Overlapped()
        if _kernel32.LockFileEx(
            msvcrt.get_osfhandle(handle.fileno()), flags, 0, 1, 0, ctypes.byref(overlapped)
        ):
            return
        code = ctypes.get_last_error()
        if code in (_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING):
            raise LockHeld(code, "the lock is held")
        raise ctypes.WinError(code)

    def process_alive(pid: int) -> bool:
        # Never os.kill(pid, 0) here: on Windows that *terminates* the process.
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return _kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            _kernel32.CloseHandle(handle)

    def detached() -> dict[str, Any]:
        """``Popen`` arguments for a child that outlives this process, windowless."""
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
        return {"creationflags": flags}

    def quiet() -> dict[str, Any]:
        """``Popen`` arguments that keep a console from flashing up."""
        return {"creationflags": subprocess.CREATE_NO_WINDOW}

else:
    import fcntl

    def _lock(handle: IO[bytes], *, exclusive: bool) -> None:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise LockHeld(exc.errno, "the lock is held") from exc
            raise

    def process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def detached() -> dict[str, Any]:
        """``Popen`` arguments for a child that outlives this process."""
        return {"start_new_session": True}

    def quiet() -> dict[str, Any]:
        """``Popen`` arguments that keep a console from flashing up (Windows only)."""
        return {}


def acquire(path: Path, *, exclusive: bool) -> IO[bytes]:
    """Take the lock on ``path`` without waiting, and return the open file that holds it.

    Every stenograf process holds this file *shared* for as long as it lives;
    the updater takes it *exclusively* for the swap. The operating system drops
    the lock with the process, a crash included, which is what makes it a
    better answer to "is stenograf still running?" than a process list.

    Raises :class:`LockHeld` when someone else's lock is in the way, and any
    other ``OSError`` when this file system cannot lock at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        _lock(handle, exclusive=exclusive)
    except BaseException:
        handle.close()
        raise
    return handle


def acquire_within(path: Path, *, exclusive: bool, seconds: float) -> IO[bytes]:
    """:func:`acquire`, retried while the lock is held, for up to ``seconds``."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return acquire(path, exclusive=exclusive)
        except LockHeld:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def names_stenograf(environment: Path) -> bool:
    """Whether ``environment`` is a uv tool environment whose receipt asks for stenograf."""
    try:
        receipt = tomllib.loads((environment / "uv-receipt.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    tool = receipt.get("tool")
    requirements = tool.get("requirements") if isinstance(tool, dict) else None
    if not isinstance(requirements, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("name") == "stenograf" for entry in requirements
    )


def environment_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def parked_path(tool_dir: Path) -> Path:
    """Where the old environment waits: beside uv's tool directory, not in it.

    Beside it so the move is a rename on one file system; not inside it because
    uv warns about every foreign folder it finds there."""
    return tool_dir.parent / f"{tool_dir.name}-stenograf-previous"


def trash_path(tool_dir: Path) -> Path:
    return tool_dir.parent / f"{tool_dir.name}-stenograf-failed"


def discard(path: Path) -> None:
    """Delete a parked or trash folder, if it is recognizably one of ours.

    Only ever called with :func:`parked_path` or :func:`trash_path`; the check
    is there so that a wrong path in a request can never become a deleted
    directory. Errors are swallowed: on Windows the running app may hold files
    that are hard links into this folder, and a later start tries again."""
    if not path.is_dir() or path.is_symlink():
        return
    mark = path / _SENTINEL
    if not (names_stenograf(path) or mark.is_file()):
        return
    # What proves the folder ours goes last. Deleting a gigabyte takes seconds,
    # and a logout in the middle must leave something a later start still
    # recognizes — a folder it does not recognize blocks every later update.
    with contextlib.suppress(OSError):
        mark.write_text("", encoding="utf-8")
    try:
        children = list(path.iterdir())
    except OSError:
        return
    for child in children:
        if child.name in (_SENTINEL, "uv-receipt.toml"):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                child.unlink()
    shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class Request:
    """What the app asks of ``apply`` — the contract's request file.

    Written by an older release than the one reading it, so a field may be
    added here with a default and must never be renamed, retyped or required.
    """

    target_version: str
    python: str
    """``3.13`` — the interpreter version ``prepare`` ran on, so the install
    is the same resolution."""
    environment: Path
    tool_dir: Path
    bin_dir: Path
    """Where uv puts the ``steno`` command. Passed on explicitly because uv
    finds its directories through environment variables that an app started
    from the Dock or the Start menu does not have."""
    uv: str
    work_dir: Path
    """The result, the log, the journal, the launcher copies and the "started"
    marker all go here."""
    lock_file: Path
    wait_pids: tuple[int, ...] = ()
    """The app and, where it has one, the launcher that is its parent. On macOS
    a launcher that is still alive turns ``open`` into "bring the old app
    forward"; on Windows a running launcher blocks the rename."""
    restart: tuple[str, ...] = ()
    """The command that starts the app the way the user starts it; empty for
    nothing to restart."""
    launcher_files: tuple[Path, ...] = ()
    """Files the new release rewrites and a rollback must put back."""
    direct: bool = False
    """``restart`` is the app itself, not a launcher that hands over and exits:
    only then does the command's own exit code say anything about the app."""
    format: int = REQUEST_FORMAT

    @classmethod
    def read(cls, path: Path) -> Request:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            target_version=str(data["target_version"]),
            python=str(data["python"]),
            environment=Path(data["environment"]),
            tool_dir=Path(data["tool_dir"]),
            bin_dir=Path(data["bin_dir"]),
            uv=str(data["uv"]),
            work_dir=Path(data["work_dir"]),
            lock_file=Path(data["lock_file"]),
            wait_pids=tuple(int(pid) for pid in data.get("wait_pids", ())),
            restart=tuple(str(part) for part in data.get("restart", ())),
            launcher_files=tuple(Path(item) for item in data.get("launcher_files", ())),
            direct=bool(data.get("direct", False)),
            format=int(data.get("format", REQUEST_FORMAT)),
        )

    def write(self, path: Path) -> None:
        data = {
            "format": self.format,
            "target_version": self.target_version,
            "python": self.python,
            "environment": str(self.environment),
            "tool_dir": str(self.tool_dir),
            "bin_dir": str(self.bin_dir),
            "uv": self.uv,
            "work_dir": str(self.work_dir),
            "lock_file": str(self.lock_file),
            "wait_pids": list(self.wait_pids),
            "restart": list(self.restart),
            "launcher_files": [str(item) for item in self.launcher_files],
            "direct": self.direct,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def journal_path(work_dir: Path) -> Path:
    """Names the parked and trash folders while a swap is open, so that a later
    start can remove what an interrupted updater left behind."""
    return work_dir / "journal.json"


def marker_path(work_dir: Path) -> Path:
    """Written by the freshly started app once its window is built."""
    return work_dir / "started.json"


def result_path(work_dir: Path) -> Path:
    return work_dir / "result.json"


def alive_path(work_dir: Path) -> Path:
    """Locked by ``apply`` for as long as it runs.

    The main lock cannot say this: it is released while the updater waits for
    the new app, which is exactly when the previous version is still parked
    for a rollback and nobody else may tidy it away."""
    return work_dir / "updater.alive"


class _Failed(Exception):
    """The swap cannot go on; the message is the sentence the user reads."""


@dataclass
class _Swap:
    request: Request
    log: IO[str]
    parked: Path = field(init=False)
    trash: Path = field(init=False)
    launcher_copies: Path = field(init=False)
    open: bool = False
    """The environment is parked and the swap neither committed nor undone."""
    installed: bool = False
    """uv finished and left a complete environment. A rollback before that
    took out nothing of the new version: uv missing, a full disk, a cache
    pruned since ``prepare``, all of which say nothing against the release."""
    rolled_back: bool = False
    """Written to the result: the old app holds the version against itself
    for good, so only a rollback of an *installed* version counts."""

    def __post_init__(self) -> None:
        self.parked = parked_path(self.request.tool_dir)
        self.trash = trash_path(self.request.tool_dir)
        self.launcher_copies = self.request.work_dir / "launchers"

    def note(self, text: str) -> None:
        self.log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        self.log.flush()

    # -- the steps ---------------------------------------------------------

    def wait_for_exit(self) -> bool:
        deadline = time.monotonic() + EXIT_WAIT_S
        while any(process_alive(pid) for pid in self.request.wait_pids):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)
        return True

    def copy_launchers(self) -> None:
        shutil.rmtree(self.launcher_copies, ignore_errors=True)
        self.launcher_copies.mkdir(parents=True)
        for index, source in enumerate(self.request.launcher_files):
            if source.is_file():
                shutil.copy2(source, self.launcher_copies / str(index))

    def restore_launchers(self) -> None:
        for index, target in enumerate(self.request.launcher_files):
            copy = self.launcher_copies / str(index)
            if copy.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(copy, target)

    def park(self) -> None:
        journal_path(self.request.work_dir).write_text(
            json.dumps({"parked": str(self.parked), "trash": str(self.trash)}), encoding="utf-8"
        )
        _rename(self.request.environment, self.parked)
        self.open = True
        self.note(f"parked {self.request.environment} at {self.parked}")

    def install(self) -> None:
        request = self.request
        command = [
            request.uv,
            "tool",
            "install",
            "--offline",
            "--force",
            "--python",
            request.python,
            f"stenograf>={request.target_version}",
        ]
        environment = {
            **os.environ,
            "UV_TOOL_DIR": str(request.tool_dir),
            "UV_TOOL_BIN_DIR": str(request.bin_dir),
        }
        self.note("running " + " ".join(command))
        try:
            done = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=INSTALL_WAIT_S,
                **quiet(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _Failed(f"The new version could not be installed ({exc}).") from exc
        self.note(done.stdout)
        if done.returncode != 0:
            raise _Failed(
                "The new version could not be installed from the downloaded files. "
                "Run the update again."
            )
        if not names_stenograf(request.environment):
            raise _Failed("The new version was not installed completely.")
        self.installed = True

    def verify(self) -> None:
        """Ask the installed program which version it is, and believe only that.

        Not the receipt: uv has reported an upgrade done whose files it could
        not replace (astral-sh/uv issue 11930)."""
        python = environment_python(self.request.environment)
        try:
            done = subprocess.run(
                [str(python), "-m", "stenograf", "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=180,
                **quiet(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _Failed(f"The new version does not run ({exc}).") from exc
        self.note(f"installed program says: {done.stdout.strip()}")
        words = done.stdout.split()
        installed = words[-1] if done.returncode == 0 and words else ""
        if installed != self.request.target_version:
            raise _Failed(
                f"Version {installed or 'unknown'} was installed instead of "
                f"{self.request.target_version}."
            )

    def refresh_launchers(self) -> None:
        """Let the new release rewrite its own launchers — if there were any.

        A failure is logged and not fatal: the environment's path has not
        changed, so the launchers that were there still start the app."""
        if not any(source.is_file() for source in self.request.launcher_files):
            return
        python = environment_python(self.request.environment)
        try:
            done = subprocess.run(
                [str(python), "-m", "stenograf._updater", "post-install"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=180,
                **quiet(),
            )
            self.note(f"launcher refresh exited {done.returncode}: {done.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.note(f"launcher refresh did not run: {exc}")

    def start_app(self) -> subprocess.Popen[Any] | None:
        if not self.request.restart:
            return None
        with contextlib.suppress(OSError):
            marker_path(self.request.work_dir).unlink()
        self.note("starting " + " ".join(self.request.restart))
        try:
            return subprocess.Popen(
                list(self.request.restart),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path.home()),
                **detached(),
            )
        except OSError as exc:
            self.note(f"the app could not be started: {exc}")
            return None

    def await_start(self, started: subprocess.Popen[Any] | None = None) -> str:
        """``started``, ``alive`` (running, never said so) or ``dead``.

        The app is usually started through its launcher, so there is no
        process id to watch and the lock is how a death is seen: an app that
        held it and no longer does has gone. The probe takes the lock
        exclusively for a moment, which an app starting in that very moment
        waits out. ``started`` is the command that was run; where the request
        says that *is* the app, its failing says so at once. A launcher's exit
        code says nothing: ``open`` returns 0 whatever becomes of the app, and
        Windows' ``explorer.exe`` returns 1 when all went well."""
        marker = marker_path(self.request.work_dir)
        began = time.monotonic()
        seen_alive = False
        free_since: float | None = None
        while time.monotonic() - began < START_WAIT_S:
            if _marker_version(marker) == self.request.target_version:
                return "started"
            direct = started if self.request.direct else None
            if direct is not None and direct.poll() not in (None, 0):
                if _marker_version(marker) == self.request.target_version:
                    return "started"
                return "dead"
            try:
                acquire(self.request.lock_file, exclusive=True).close()
            except LockHeld:
                seen_alive = True
                free_since = None
            except OSError:
                return "alive"  # cannot tell; never roll back on a guess
            else:
                now = time.monotonic()
                free_since = now if free_since is None else free_since
                gone = seen_alive or now - began > NEVER_STARTED_S
                if gone and now - free_since >= QUIET_S:
                    # Once more: the marker may have landed just before the exit.
                    if _marker_version(marker) == self.request.target_version:
                        return "started"
                    return "dead"
            time.sleep(0.5)
        return "alive" if seen_alive else "dead"

    def roll_back(self) -> None:
        environment = self.request.environment
        if environment.exists():
            discard(self.trash)
            # Marked before it moves: a half-installed folder has no receipt,
            # and one that reached the trash unmarked could never be removed.
            with contextlib.suppress(OSError):
                (environment / _SENTINEL).write_text("", encoding="utf-8")
            _rename(environment, self.trash)
        _rename(self.parked, environment)
        self.open = False
        self.rolled_back = self.installed
        self.restore_launchers()
        self.note("rolled back to the previous version")
        self.close_journal()
        discard(self.trash)

    def commit(self) -> None:
        self.open = False
        discard(self.parked)
        self.close_journal()
        self.note("removed the previous version")

    def close_journal(self) -> None:
        shutil.rmtree(self.launcher_copies, ignore_errors=True)
        with contextlib.suppress(OSError):
            journal_path(self.request.work_dir).unlink()


def apply(request_file: Path) -> int:
    """Replace the installed environment with the requested version; 0 on success."""
    request = Request.read(request_file)
    request.work_dir.mkdir(parents=True, exist_ok=True)
    log_file = request.work_dir / "updater.log"
    with log_file.open("a", encoding="utf-8") as log, contextlib.ExitStack() as held:
        swap = _Swap(request, log)
        swap.note(f"--- update to {request.target_version} (request format {request.format}) ---")
        try:
            held.enter_context(acquire(alive_path(request.work_dir), exclusive=True))
        except LockHeld:
            # Its result file is the one that counts; writing a second would
            # tell the user the update failed while it is still running.
            swap.note("another updater is at work on this installation; leaving it to it")
            return 1
        except OSError as exc:
            swap.note(f"could not mark this updater as running: {exc}")
        try:
            ok, message, restart = _run(swap)
        except Exception as exc:  # noqa: BLE001 — the user must get a result file whatever went wrong
            swap.note(f"unexpected failure: {exc!r}")
            ok, message, restart = _after_crash(swap, exc)
        swap.note(("done: " if ok else "failed: ") + message)
        # A full disk must not cost the user the restart as well.
        with contextlib.suppress(OSError):
            result_path(request.work_dir).write_text(
                json.dumps(
                    {
                        "format": RESULT_FORMAT,
                        "ok": ok,
                        "version": request.target_version,
                        "message": message,
                        "log": str(log_file),
                        "rolled_back": swap.rolled_back,
                    }
                ),
                encoding="utf-8",
            )
        if restart:
            swap.start_app()
    return 0 if ok else 1


def _after_crash(swap: _Swap, exc: Exception) -> tuple[bool, str, bool]:
    """Where an exception nobody planned for leaves the machine."""
    reason = f"The update stopped unexpectedly ({exc})."
    if not swap.open:
        # Nothing is parked, so whatever is installed is whole: start it.
        return False, reason, names_stenograf(swap.request.environment)
    try:
        lock = acquire_within(swap.request.lock_file, exclusive=True, seconds=LOCK_WAIT_S)
    except OSError:
        # Someone is running from the new environment. Taking it from under
        # them is worse than leaving the previous version parked.
        return False, reason, False
    with lock, _uninterrupted():
        return _undo(swap, reason)


def _run(swap: _Swap) -> tuple[bool, str, bool]:
    """The swap itself: ``(ok, sentence for the user, start the app afterwards)``."""
    request = swap.request
    version = request.target_version
    if not swap.wait_for_exit():
        return False, "Stenograf did not close, so the update was not started.", False
    try:
        lock = acquire_within(request.lock_file, exclusive=True, seconds=LOCK_WAIT_S)
    except LockHeld:
        return (
            False,
            "Stenograf is in use elsewhere — a meeting in a terminal? The update was not started.",
            True,
        )
    except OSError as exc:
        return False, f"The update was not started ({exc}).", True

    with lock, _uninterrupted():
        if not names_stenograf(request.environment):
            return False, "This is not a standard installation, so it was left alone.", True
        discard(swap.parked)
        discard(swap.trash)
        if swap.parked.exists() or swap.trash.exists():
            return False, "Leftovers of an earlier update are in the way.", True
        swap.copy_launchers()
        try:
            swap.park()
        except OSError as exc:
            swap.close_journal()
            swap.note(f"could not park the environment: {exc}")
            return (
                False,
                "The installed version could not be moved aside (a file in it is in use). "
                "The update was not started.",
                True,
            )
        # Every way out of here that is not success undoes the swap *inside*
        # this block: once the lock is released the app can be started, and a
        # rollback would then take the environment from under it.
        try:
            swap.install()
            swap.verify()
            swap.refresh_launchers()
        except _Failed as failure:
            return _undo(swap, str(failure))
        except Exception as exc:  # noqa: BLE001 — see above
            swap.note(f"unexpected failure: {exc!r}")
            return _undo(swap, f"The update stopped unexpectedly ({exc}).")

    if not request.restart:
        swap.commit()
        return True, f"Updated to {version}.", False
    verdict = swap.await_start(swap.start_app())
    swap.note(f"the new app is {verdict}")
    if verdict == "started":
        swap.commit()
        return True, f"Updated to {version}.", False
    if verdict == "alive":
        # Not confirmed and not dead: the previous version stays parked, and
        # the next start that gets as far as its window removes it.
        swap.open = False
        return True, f"Updated to {version}.", False
    try:
        lock = acquire_within(request.lock_file, exclusive=True, seconds=LOCK_WAIT_S)
    except OSError:
        swap.open = False
        return True, f"Updated to {version}.", False  # someone runs after all
    with lock, _uninterrupted():
        return _undo(swap, f"Version {version} did not start.")


def _undo(swap: _Swap, reason: str) -> tuple[bool, str, bool]:
    try:
        swap.roll_back()
    except OSError as exc:
        swap.note(f"rollback failed: {exc}")
        return (
            False,
            f"{reason} The previous version could not be restored either. "
            f"Run the install command again:  {REINSTALL_HINT}",
            False,
        )
    return False, f"{reason} The previous version was restored.", True


def _rename(source: Path, target: Path) -> None:
    """``os.rename``, retried while Windows reports a file inside as in use."""
    deadline = time.monotonic() + RENAME_WAIT_S
    delay = 0.1
    while True:
        try:
            os.rename(source, target)
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
        else:
            return


def _marker_version(marker: Path) -> str | None:
    try:
        return str(json.loads(marker.read_text(encoding="utf-8")).get("version"))
    except (OSError, ValueError, AttributeError):
        return None


@contextlib.contextmanager
def _uninterrupted() -> Iterator[None]:
    """Ignore requests to terminate while the environment is being swapped.

    A logout sends one, and stopping between "old one moved away" and "new one
    in place" leaves an install nothing can start. The swap is seconds long, so
    finishing — or rolling back — beats obeying. Windows has no equivalent for
    a windowless process and is simply killed."""
    if sys.platform == "win32":
        yield
        return
    previous = {}
    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):  # not the main thread
            previous[number] = signal.signal(number, signal.SIG_IGN)
    try:
        yield
    finally:
        for number, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, handler)


# -- prepare: runs inside the new release, and may import it ----------------


def prepare(out: Path) -> int:
    """Check that this release can run here and leave ``updater.py`` in ``out``."""
    problem = _self_check()
    if problem is not None:
        print(problem)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), out / "updater.py")
    _fetch_missing_models()
    from stenograf import __version__

    print(f"stenograf {__version__} is ready to install")
    return 0


def _self_check() -> str | None:
    """The reason this release cannot be used on this machine, if there is one."""
    import platform

    from stenograf.capture.helper import HELPER_NAME
    from stenograf.doctor import _desktop_app_check

    # Every release also publishes a generic package without the capture
    # helper. It installs and starts everywhere, so nothing else would notice
    # that an install landed on it — a check that runs while a release is still
    # uploading can.
    own_package = {
        "darwin": ("arm64",),
        "win32": ("AMD64", "x86_64"),
        "linux": ("x86_64",),
    }
    if platform.machine() in own_package.get(sys.platform, ()):
        helper = Path(__file__).parent / "bin" / HELPER_NAME
        if not helper.is_file():
            return "This release has no capture helper for this system yet. Try again later."
        try:
            ran = subprocess.run(
                [str(helper), "--help"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                **quiet(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"This release's capture helper does not run here ({exc})."
        if ran.returncode != 0:
            return "This release's capture helper does not run here."
    check = _desktop_app_check()
    if not check.ok:
        return f"This release's window does not open here: {check.detail}"
    return None


def _fetch_missing_models() -> None:
    """Download the model files this release needs and the machine lacks.

    So that the first meeting after the update does not stop to fetch them.
    Files only — loading the speech model to prove it works would put several
    GB next to the running app on every update. A failed download is not a
    failed update: the models still download on first use."""
    # huggingface_hub's progress bars write to sys.stderr, and under pythonw
    # there is none: the write raises or the download hangs (measured
    # 2026-11-20 with sys.stderr set to None, macOS).
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 — lives as long as the process
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 — as above
    try:
        from stenograf import assets

        for asset in (assets.SILERO_VAD, *assets.NEMOTRON, assets.SPEAKER_EMBEDDING):
            if assets.cached_path(asset) is None:
                assets.fetch(asset, lambda *_: None)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(f"models were not prefetched: {exc}", file=sys.stderr)
    try:
        _fetch_speech_model()
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(f"the speech model was not prefetched: {exc}", file=sys.stderr)


def _fetch_speech_model() -> None:
    """Download the speech model the next meeting will load, without loading it.

    The ONNX model lives in stenograf's own cache folder, keyed by its
    revision, so a release that moves or re-pins it would otherwise cost the
    first meeting after the update a ~2.5 GB download. That download now
    happens here instead, while the update line says "Preparing…", with no
    progress of its own; an unchanged model costs nothing."""
    from stenograf.asr import default_backend_name
    from stenograf.settings import SettingsError, load_settings

    try:
        configured = load_settings().asr.backend
    except SettingsError:
        configured = None
    name = default_backend_name(configured)
    if name == "parakeet-onnx":
        from stenograf.asr import parakeet_onnx

        parakeet_onnx.model_dir()
    elif name == "parakeet":
        from huggingface_hub import hf_hub_download

        from stenograf.asr import parakeet

        if parakeet._cached_snapshot(parakeet.MODEL_ID) is None:
            for filename in ("config.json", "model.safetensors"):
                hf_hub_download(parakeet.MODEL_ID, filename)


def post_install() -> int:
    """Rewrite this release's launchers that are still there; run by ``apply``
    in the new environment."""
    from stenograf.shortcut import refresh_shortcut

    try:
        launcher = refresh_shortcut()
    except OSError as exc:
        print(f"launcher not written: {exc}")
        return 1
    print(f"launcher: {launcher}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stenograf._updater")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--out", type=Path, required=True)
    commands.add_parser("apply").add_argument("request", type=Path)
    commands.add_parser("post-install")
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        return prepare(arguments.out)
    if arguments.command == "post-install":
        return post_install()
    # The environment is about to be renamed, and a folder that is some
    # process's working directory cannot be on Windows.
    with contextlib.suppress(OSError):
        os.chdir(arguments.request.parent)
    return apply(arguments.request)


if __name__ == "__main__":
    sys.exit(main())
