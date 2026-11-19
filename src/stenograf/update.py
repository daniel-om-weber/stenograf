"""Noticing a newer release, and handing the install to the updater.

The library behind the app's "Update" line and the terminal's one-line notice.
Both front-ends ask this module; neither holds update logic. What it does:

- **Check**: one request a day to PyPI's JSON API, in the background, switched
  off by ``[updates] check = false``. See :func:`newest_release` for which
  release counts.
- **Decide what to offer**: only a *standard* install — a uv tool environment
  whose receipt asks for plain ``stenograf`` — updates itself. A pin, a
  checkout or a pip install is something a person chose, and gets a notice or
  nothing (:func:`describe_install`).
- **Prepare**: download the new release into uv's cache and run it once from
  there, while the app stays open and nothing installed changes.
- **Hand over**: write the request, start the *new* release's updater detached
  on the base interpreter, and let the app quit. The swap itself, and the
  contract that binds this release to every later one, is
  :mod:`stenograf._updater`.

Standard library only, like :mod:`stenograf.paths`: the lock is taken before
anything heavy loads.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from stenograf import __version__, _updater

if TYPE_CHECKING:
    from collections.abc import Callable

CHECK_URL = "https://pypi.org/pypi/stenograf/json"
URL_OVERRIDE = "STENOGRAF_UPDATE_URL"
"""Points the check at another address. The end-to-end update test serves a
release from a local file with it; it has to exist in every release that can
update, since that test drives the *installed* release."""

UNMEASURED_OVERRIDE = "STENOGRAF_UPDATE_UNMEASURED"

CHECK_INTERVAL_S = 24 * 3600
NOTICE_INTERVAL_S = 24 * 3600
SETTLE_S = 3600
"""A release is offered an hour after its last file was uploaded. Files arrive
one by one, and the generic package without the capture helper is a valid
install for every system until the system's own package is there."""

FETCH_TIMEOUT_S = 10.0

UPGRADE_COMMAND = "uv tool install --upgrade --python 3.13 stenograf"
RELEASE_URL = "https://github.com/daniel-om-weber/stenograf/releases/tag/v{version}"

SWAP_MEASURED = ("darwin", "linux")
"""Where renaming the environment aside and back has been measured
(``eval/update_mechanics.sh``, 2026-11-19). Elsewhere the app shows the notice
instead of the button until the same has been measured there."""

MIN_UV = (0, 5, 0)
"""Below this the commands used here are not all there. Tested with 0.13.7 and
0.13.8 only (2026-11-19); an older uv that lacks something fails in
:func:`prepare`, before anything installed is touched."""

_FINAL_VERSION = re.compile(r"\d+(\.\d+)*")
_LOWER_BOUND = re.compile(r"\s*>=\s*\d+(\.\d+)*\s*")
_CLAUSE = re.compile(r"(>=|<=|==|!=|~=|<|>)\s*(\d+(?:\.\d+)*)(\.\*)?")

_SOURCE_KEYS = frozenset({"directory", "path", "git", "url", "editable"})
_EMPTY_EXTRAS = frozenset({"gui", "ollama"})


class UpdateError(Exception):
    """An update step failed; the message is the sentence the user reads."""


class UpdateCancelled(UpdateError):
    pass


# -- where things live ------------------------------------------------------


def root() -> Path:
    """The per-user folder for update state, at a place nothing can move.

    Deliberately not :func:`stenograf.paths.data_dir`: ``$STENOGRAF_DATA`` can
    relocate that, and an app started from the Dock does not see a variable
    that a terminal session sets. Both must agree on where the lock is."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "stenograf" / "update"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "stenograf" / "update"
    return Path.home() / ".local" / "state" / "stenograf" / "update"


def work_dir() -> Path:
    """This environment's folder under :func:`root`.

    Keyed by the environment's path, so a uv tool install and a development
    checkout on one machine neither share a lock nor see each other's state."""
    key = hashlib.sha256(os.path.normcase(os.path.abspath(sys.prefix)).encode()).hexdigest()[:12]
    return root() / key


def lock_path() -> Path:
    return work_dir() / "in-use.lock"


# -- the lock every process holds -------------------------------------------

_held: IO[bytes] | None = None
_lock_state = ""


def hold_lock() -> str:
    """Take this process's share of the lock, for as long as it lives.

    ``held`` normally; ``updating`` when the updater holds it exclusively, in
    which case this process must not go on — its program files are being
    replaced; ``unsupported`` where the file system cannot lock (some network
    home directories), which costs nothing but self-update."""
    global _held, _lock_state
    if _lock_state in ("held", "unsupported"):
        return _lock_state
    try:
        # The updater also takes the lock for an instant at a time while it
        # watches the new app start; a few seconds of patience covers that.
        _held = _updater.acquire_within(lock_path(), exclusive=False, seconds=3.0)
    except _updater.LockHeld:
        return "updating"
    except OSError:
        _lock_state = "unsupported"
        return _lock_state
    _lock_state = "held"
    return _lock_state


# -- versions and what a release requires ------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    """``1.2.3`` as a tuple; ``None`` for anything that is not a final release.

    Without trailing zeros, so that ``0.8`` and ``0.8.0`` are one version and
    neither is offered as an update to the other."""
    if not _FINAL_VERSION.fullmatch(text.strip()):
        return None
    parts = [int(part) for part in text.strip().split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def python_allowed(spec: str | None, version: tuple[int, ...]) -> bool | None:
    """Whether ``requires_python`` admits ``version``; ``None`` if it cannot be read.

    The clauses a release of this project uses, not all of PEP 440: anything
    else answers ``None``, and the caller then offers no self-update."""
    if not spec or not spec.strip():
        return True
    for clause in spec.split(","):
        match = _CLAUSE.fullmatch(clause.strip())
        if match is None:
            return None
        operator, number, wildcard = match.groups()
        bound = tuple(int(part) for part in number.split("."))
        prefix = version[: len(bound)]
        if wildcard:
            if operator not in ("==", "!="):
                return None
            holds = (prefix == bound) == (operator == "==")
        elif operator == ">=":
            holds = version >= bound
        elif operator == ">":
            holds = version > bound
        elif operator == "<=":
            holds = _padded(version, bound) <= _padded(bound, version)
        elif operator == "<":
            holds = version < bound
        elif operator == "==":
            holds = _padded(version, bound) == _padded(bound, version)
        elif operator == "!=":
            holds = _padded(version, bound) != _padded(bound, version)
        else:  # ~=X.Y means >=X.Y and the same X
            holds = len(bound) >= 2 and version >= bound and prefix[:-1] == bound[:-1]
        if not holds:
            return False
    return True


def _padded(value: tuple[int, ...], other: tuple[int, ...]) -> tuple[int, ...]:
    return value + (0,) * (len(other) - len(value))


@dataclass(frozen=True)
class Release:
    version: str
    requires_python: str | None


def newest_release(
    payload: dict[str, Any], *, installed: tuple[int, ...], now: float
) -> Release | None:
    """The release worth offering out of PyPI's answer, or ``None``.

    Newer than the installed one, a final release, not withdrawn ("yanked" —
    the brake for a bad release), and uploaded completely an hour ago
    (:data:`SETTLE_S`)."""
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        return None
    best: tuple[tuple[int, ...], str, list[dict[str, Any]]] | None = None
    for text, files in releases.items():
        version = parse_version(str(text))
        if version is None or version <= installed or not isinstance(files, list) or not files:
            continue
        if any(not isinstance(item, dict) or item.get("yanked") for item in files):
            continue
        uploaded = [_timestamp(item.get("upload_time_iso_8601")) for item in files]
        if any(stamp is None or now - stamp < SETTLE_S for stamp in uploaded):
            continue
        if best is None or version > best[0]:
            best = (version, str(text), files)
    if best is None:
        return None
    requires = next(
        (item["requires_python"] for item in best[2] if item.get("requires_python")), None
    )
    return Release(version=best[1], requires_python=requires)


def _timestamp(text: object) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def fetch() -> dict[str, Any]:
    """PyPI's record of every stenograf release. Carries no user data."""
    request = urllib.request.Request(
        os.environ.get(URL_OVERRIDE) or CHECK_URL,
        headers={"User-Agent": f"stenograf/{__version__}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("the release list is not a JSON object")
    return payload


# -- what kind of install this is -------------------------------------------


@dataclass(frozen=True)
class Install:
    kind: str
    """``standard`` (updates itself), ``pinned`` (a uv tool install someone
    constrained: a notice), ``pip`` (a notice without a command) or ``source``
    (a checkout, git or URL install: nothing at all)."""
    environment: Path | None = None
    bin_dir: Path | None = None


def describe_install(prefix: Path | None = None) -> Install:
    """Classify the running install from uv's receipt, if it has one."""
    environment = Path(prefix if prefix is not None else sys.prefix)
    receipt = environment / "uv-receipt.toml"
    if not receipt.is_file():
        return Install(kind="source" if _installed_from_source() else "pip")
    try:
        tool = tomllib.loads(receipt.read_text(encoding="utf-8")).get("tool")
    except (OSError, ValueError):
        return Install(kind="pinned", environment=environment)
    if not isinstance(tool, dict):
        return Install(kind="pinned", environment=environment)
    requirements = tool.get("requirements")
    if not isinstance(requirements, list) or len(requirements) != 1:
        return Install(kind="pinned", environment=environment)
    requirement = requirements[0]
    if not isinstance(requirement, dict) or requirement.get("name") != "stenograf":
        return Install(kind="pinned", environment=environment)
    if _SOURCE_KEYS & requirement.keys():
        return Install(kind="source", environment=environment)

    bin_dir = None
    for entry in tool.get("entrypoints") or ():
        if isinstance(entry, dict) and isinstance(entry.get("install-path"), str):
            bin_dir = Path(entry["install-path"]).parent
            break

    specifier = requirement.get("specifier")
    plain = (
        requirement.keys() <= {"name", "extras", "specifier"}
        and set(requirement.get("extras") or ()) <= _EMPTY_EXTRAS
        and (specifier is None or _LOWER_BOUND.fullmatch(str(specifier)) is not None)
        # Each of these sticks in the receipt and holds every later upgrade
        # back, so each is a decision, not a default.
        and not tool.get("constraints")
        and not tool.get("overrides")
        and not tool.get("build-constraints")
        and not tool.get("options")
    )
    kind = "standard" if plain and bin_dir is not None else "pinned"
    return Install(kind=kind, environment=environment, bin_dir=bin_dir)


def _installed_from_source() -> bool:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return distribution("stenograf").read_text("direct_url.json") is not None
    except PackageNotFoundError:
        return True


_uv: tuple[str | None] | None = None


def find_uv() -> str | None:
    """The uv to run, or ``None`` when there is none new enough.

    Found, never installed or updated: on PATH, then where uv's own installer
    and the common package managers put it — a Linux desktop session often has
    no ``~/.local/bin`` on its PATH."""
    global _uv
    if _uv is not None:
        return _uv[0]
    name = "uv.exe" if sys.platform == "win32" else "uv"
    candidates = [
        shutil.which("uv"),
        str(Path.home() / ".local" / "bin" / name),
        str(Path.home() / ".cargo" / "bin" / name),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]
    found = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and _uv_version(candidate) >= MIN_UV:
            found = candidate
            break
    _uv = (found,)
    return found


def _uv_version(uv: str) -> tuple[int, ...]:
    try:
        done = subprocess.run(
            [uv, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            **_updater.quiet(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    match = re.search(r"\d+(\.\d+)+", done.stdout)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


# -- stored state -----------------------------------------------------------


def load_state() -> dict[str, Any]:
    try:
        state = json.loads((work_dir() / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(state: dict[str, Any]) -> None:
    """Best effort, and atomic: a background check and a command's exit can
    both write, and a half-written file would read as no state at all."""
    folder = work_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        scratch = folder / f"state.{os.getpid()}.{threading.get_ident()}.tmp"
        scratch.write_text(json.dumps(state), encoding="utf-8")
        os.replace(scratch, folder / "state.json")
    except OSError:
        pass


# -- the check --------------------------------------------------------------


def enabled() -> bool:
    """Whether this install checks at all: the one switch, and not a checkout."""
    if parse_version(__version__) is None or describe_install().kind == "source":
        return False
    from stenograf.settings import SettingsError, load_settings

    try:
        return load_settings().updates.check is not False
    except SettingsError:
        # A file that cannot be read may well say "off"; do not guess "on".
        return False


def check(*, force: bool = False) -> None:
    """Ask PyPI whether a newer release exists and store the answer.

    A no-op when switched off, and — unless forced — when the last answer is
    less than a day old. Every failure is silent: no network is the normal
    case on a train, not an error worth a word."""
    if not enabled():
        return
    state = load_state()
    now = time.time()
    # A stored time in the future — the clock was wrong once — counts as old,
    # or the check would stay silent until real time caught up with it.
    if not force and 0 <= now - float(state.get("checked_at") or 0) < CHECK_INTERVAL_S:
        return
    installed = parse_version(__version__)
    if installed is None:
        return
    try:
        release = newest_release(fetch(), installed=installed, now=now)
    except Exception:  # noqa: BLE001 — see the docstring
        return
    state = load_state()
    state["checked_at"] = now
    state["latest"] = release.version if release else None
    state["requires_python"] = release.requires_python if release else None
    save_state(state)


def check_in_background() -> None:
    """:func:`check` on a daemon thread, for a command that has minutes to spare."""
    threading.Thread(target=check, name="update-check", daemon=True).start()


@dataclass(frozen=True)
class Offer:
    version: str
    installable: bool
    """Whether this install can take it with a click; otherwise it is a notice."""
    command: str | None
    """The command to run by hand, where there is one to give."""
    reason: str = ""
    """Why it is only a notice, in words for the user."""

    @property
    def url(self) -> str:
        return RELEASE_URL.format(version=self.version)


def offer(*, include_failed: bool = False) -> Offer | None:
    """The newer release the last check found, if this install should hear of it.

    A version that already failed here is left out unless asked for: without
    that a broken update would be offered again every day."""
    if not enabled():
        return None
    state = load_state()
    latest = state.get("latest")
    installed = parse_version(__version__)
    if not isinstance(latest, str) or installed is None:
        return None
    newer = parse_version(latest)
    if newer is None or newer <= installed:
        return None
    if not include_failed and latest in (state.get("failed") or ()):
        return None
    install = describe_install()
    if install.kind == "pip":
        return Offer(
            latest,
            False,
            None,
            "This copy was not installed with the install command; "
            "update it the way it was installed.",
        )
    if install.kind == "pinned":
        return Offer(latest, False, None, "This install is pinned and stays on its version.")
    reason = _blocker(state.get("requires_python"))
    return Offer(latest, reason is None, UPGRADE_COMMAND, reason or "")


def _blocker(requires_python: object) -> str | None:
    """Why a standard install cannot take the release with a click, if it cannot."""
    if sys.platform not in SWAP_MEASURED and not os.environ.get(UNMEASURED_OVERRIDE):
        return "Updating with one click is not enabled on this system yet."
    if hold_lock() != "held":
        return "This home folder cannot lock files, so the app cannot update itself."
    spec = requires_python if isinstance(requires_python, str) else None
    allowed = python_allowed(spec, tuple(sys.version_info[:3]))
    if allowed is None:
        return "Which Python this release needs could not be read."
    if not allowed:
        # The update keeps the Python it has. A release that drops it comes
        # about once a year, and the install command names the new one.
        return "This release needs a newer Python."
    if find_uv() is None:
        return "uv was not found, so the app cannot update itself."
    return None


def notice() -> str | None:
    """The line a terminal command prints as it ends, at most once a day."""
    state = load_state()
    now = time.time()
    # Before anything else: working out the offer runs `uv --version`, and a
    # command that has already said its line today must not pay for that.
    if 0 <= now - float(state.get("noticed_at") or 0) < NOTICE_INTERVAL_S:
        return None
    found = offer(include_failed=True)
    if found is None:
        return None
    state = load_state()
    state["noticed_at"] = now
    save_state(state)
    return describe(found)


def describe(found: Offer) -> str:
    line = f"stenograf {found.version} is available (this is {__version__})."
    if found.command is None:
        return f"{line} {found.reason}".rstrip()
    return f"{line} To update — not while a meeting is running:\n  {found.command}"


# -- prepare and hand over --------------------------------------------------


def running_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def prepare(version: str, *, cancelled: Callable[[], bool] = lambda: False) -> Path:
    """Download ``version`` into uv's cache, run it once, and fetch its updater.

    Nothing installed changes. Returns the folder that now holds
    ``updater.py``; raises :class:`UpdateError` with a sentence for the user,
    or :class:`UpdateCancelled`."""
    uv = find_uv()
    if uv is None:
        raise UpdateError("uv was not found.")
    folder = work_dir() / "prepared"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    command = [
        uv,
        "tool",
        "run",
        # The same interpreter the install will use: uv would otherwise pick
        # its own, and ignores a release's upper bound when it does.
        "--python",
        running_python(),
        "--from",
        f"stenograf=={version}",
        "python",
        "-m",
        "stenograf._updater",
        "prepare",
        "--out",
        str(folder),
    ]
    output = work_dir() / "prepare.log"
    with output.open("w", encoding="utf-8") as sink:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                cwd=str(work_dir()),
                **_child_group(),
            )
        except OSError as exc:
            raise UpdateError(f"uv could not be started ({exc}).") from exc
        while process.poll() is None:
            if cancelled():
                _stop(process)
                raise UpdateCancelled("The update was cancelled.")
            time.sleep(0.2)
    if process.returncode != 0:
        raise UpdateError(_last_line(output) or "The new version could not be downloaded.")
    if not (folder / "updater.py").is_file():
        raise UpdateError("The new version did not provide its updater.")
    return folder


def _child_group() -> dict[str, Any]:
    """Its own process group, so a cancel reaches uv *and* what uv started."""
    if sys.platform == "win32":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        }
    return {"start_new_session": True}


def _stop(process: subprocess.Popen[Any]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
            **_updater.quiet(),
        )
    else:
        import signal

        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _last_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


_LAUNCHERS = ("open", "explorer.exe")
"""What :func:`restart_command` starts the app through. Their exit codes say
nothing about the app, and the updater must know not to read them."""


def hand_over(
    version: str,
    prepared: Path,
    *,
    restart: list[str],
    wait_pids: list[int],
    launcher_files: list[Path],
) -> None:
    """Start the new release's updater and return; the caller then exits with 0.

    The updater runs on the interpreter this environment was built from, which
    lies outside it, so the environment can be renamed away underneath."""
    install = describe_install()
    uv = find_uv()
    if install.kind != "standard" or install.environment is None or install.bin_dir is None:
        raise UpdateError("This install cannot update itself.")
    if uv is None:
        raise UpdateError("uv was not found.")
    request_file = prepared / "request.json"
    _updater.Request(
        target_version=version,
        python=running_python(),
        environment=install.environment,
        tool_dir=install.environment.parent,
        bin_dir=install.bin_dir,
        uv=uv,
        work_dir=work_dir(),
        lock_file=lock_path(),
        wait_pids=tuple(wait_pids),
        restart=tuple(restart),
        launcher_files=tuple(launcher_files),
        direct=bool(restart) and Path(restart[0]).name not in _LAUNCHERS,
    ).write(request_file)

    state = load_state()
    state["pending"] = {"target": version, "from": __version__}
    save_state(state)
    with contextlib.suppress(OSError):
        _updater.result_path(work_dir()).unlink()
    try:
        subprocess.Popen(
            [_base_python(), "-I", str(prepared / "updater.py"), "apply", str(request_file)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_dir()),
            **_updater.detached(),
        )
    except OSError as exc:
        state.pop("pending", None)
        save_state(state)
        raise UpdateError(f"The updater could not be started ({exc}).") from exc


def _base_python() -> str:
    base = Path(getattr(sys, "_base_executable", None) or sys.executable)
    if sys.platform == "win32":
        windowless = base.with_name("pythonw.exe")
        if windowless.is_file():
            return str(windowless)
    return str(base)


def restart_command(*, tray: bool) -> list[str]:
    """The command that starts the app the way the user starts it.

    macOS goes through ``open`` on the bundle, never the program directly: the
    microphone permission belongs to the bundle's launcher, and an app started
    beside it would be a different program to the system. The bundle may have
    been moved, so its location is the one the launcher reported."""
    extra = ["--tray"] if tray else []
    bundle = os.environ.get("STENOGRAF_APP_BUNDLE")
    if sys.platform == "darwin" and bundle:
        return ["/usr/bin/open", bundle, *(["--args", *extra] if extra else [])]
    if sys.platform == "win32":
        from stenograf import shortcut

        link = shortcut._windows_link_paths()[0]
        if link.is_file() and not tray:  # a shell link cannot carry --tray
            return ["explorer.exe", str(link)]
        return [shortcut._windowed_python(), "-m", "stenograf", "--gui", *extra]
    return [sys.executable, "-m", "stenograf", "--gui", *extra]


def launch_chain() -> list[int]:
    """This process and, where it has one, the launcher that is its parent.

    The updater waits for all of them. On macOS the bundle's launcher stays
    alive as the app's parent, and while it lives ``open`` only brings the old
    app forward; on Windows the environment's own ``python.exe`` is a launcher
    whose file lies in the folder about to be renamed."""
    chain = [os.getpid()]
    launched_by_bundle = sys.platform == "darwin" and os.environ.get("STENOGRAF_APP_BUNDLE")
    if launched_by_bundle or sys.platform == "win32":
        chain.append(os.getppid())
    return chain


def launcher_files() -> list[Path]:
    """The launcher files the new release rewrites, for a rollback to put back."""
    from stenograf import shortcut

    if sys.platform == "darwin":
        candidates = [shortcut.launch_target_path()]
    elif sys.platform == "win32":
        candidates = shortcut._windows_link_paths()
    else:
        data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
        candidates = [data_home / "applications" / f"{shortcut.DESKTOP_FILE_NAME}.desktop"]
    return [path for path in candidates if path.is_file()]


# -- after the restart ------------------------------------------------------


def mark_started() -> None:
    """Tell a waiting updater that this version's window is built.

    That is what lets it delete the previous version; until then a version
    that installs fine and dies at start can still be rolled back."""
    if "pending" not in load_state():
        return
    with contextlib.suppress(OSError):
        _updater.marker_path(work_dir()).write_text(
            json.dumps({"version": __version__}), encoding="utf-8"
        )


@dataclass(frozen=True)
class Report:
    updated: bool
    version: str
    message: str


def settle() -> Report | None:
    """What became of an update this install asked for, once, at the next start.

    Also removes what an interrupted updater left on disk."""
    state = load_state()
    pending = state.pop("pending", None)
    _clear_leftovers()
    if not isinstance(pending, dict) or not isinstance(pending.get("target"), str):
        return None
    target = pending["target"]
    result = _read_result()
    if __version__ == target:
        report = Report(True, target, f"Updated to {target}.")
    else:
        ours = result.get("version") == target
        told = result.get("message") if ours else None
        message = (
            told if isinstance(told, str) and told else f"The update to {target} did not finish."
        )
        if ours and result.get("rolled_back") is True:
            # Only a version that was installed here and had to be taken out
            # again. An update that never began — a meeting in a terminal, a
            # file in use — says nothing against the version.
            failed = [item for item in state.get("failed") or () if isinstance(item, str)]
            state["failed"] = sorted({*failed, target})
        report = Report(False, target, message)
    save_state(state)
    with contextlib.suppress(OSError):
        _updater.result_path(work_dir()).unlink()
    return report


def _read_result() -> dict[str, Any]:
    try:
        result = json.loads(_updater.result_path(work_dir()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _clear_leftovers() -> None:
    """Remove a parked or failed environment that an updater did not get to.

    Only when this process runs from a complete environment of its own: then
    whatever is parked is, by definition, not the one in use."""
    journal = _updater.journal_path(work_dir())
    install = describe_install()
    if not journal.is_file() or install.environment is None:
        return
    if not _updater.names_stenograf(install.environment):
        return
    tool_dir = install.environment.parent
    try:
        # An updater that still runs is waiting to hear whether this very
        # process starts, with the previous version parked in case it does
        # not. It tidies up itself; deleting under it would leave a failed
        # start with nothing to go back to.
        alive = _updater.acquire(_updater.alive_path(work_dir()), exclusive=True)
    except OSError:
        return
    folder = work_dir()

    def clear() -> None:
        with alive:
            _updater.discard(_updater.parked_path(tool_dir))
            _updater.discard(_updater.trash_path(tool_dir))
            shutil.rmtree(folder / "launchers", ignore_errors=True)
            with contextlib.suppress(OSError):
                journal.unlink()

    threading.Thread(target=clear, name="update-cleanup", daemon=True).start()
