"""The self-update, end to end, on a real uv — without touching anyone's install.

    python scripts/update_e2e.py --wheel dist/stenograf-<version>-<tags>.whl
    python scripts/update_e2e.py --wheel dist/stenograf-<version>-<tags>.whl --from-published

Takes ONE built wheel and makes three releases of it by rewriting the version:
``old``, ``new``, and ``broken`` (identical, except that its app exits at
start). Then, inside a throwaway home directory:

1. installs ``old`` the way the installer does (a standard ``uv tool install``);
2. has the *installed program* find ``new``, prepare it and hand over to the
   updater — and expects the same folder to hold ``new`` afterwards, started,
   with nothing left parked;
3. does the same toward ``broken`` — and expects ``new`` back, started again,
   its launchers as they were, and ``broken`` remembered as failed.

That proves the wheel updates from *itself*. A user's update is two releases
talking: the installed one's ``update.py`` writes the request and the new
one's ``_updater.py`` reads it (the contract in ``_updater.py``'s docstring),
so a release that breaks the contract passes the run above on both sides at
once. ``--from-published`` therefore starts from the newest release on PyPI
instead of ``old``: it installs that, updates it toward ``broken`` (the new
updater must put the *published* install back, and the published app must
start and record the failure), then toward ``new``.

What stands in for PyPI: the release list is a local file
(``STENOGRAF_UPDATE_URL``) and the three wheels are offered to uv through a
config file's ``find-links``; every dependency still comes from the real index.
uv writes that setting into the install's receipt whichever way it is given
(config file or ``UV_FIND_LINKS``, measured 2026-11-19 on uv 0.13.7), and an
install with recorded options is one this feature rightly leaves to its owner
— so the test removes that one entry after each install, and fails if uv
recorded anything else. What
stands in for the user: a few lines run by the installed interpreter that call
the same library functions the app's Update button calls. The app is started
headless and exits by itself after a few seconds.

Not covered, because it cannot be done without a person's real launcher: the
restart through ``open Stenograf.app`` on macOS and through the shell link on
Windows. Run with any Python 3.12+; needs only the standard library and ``uv``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

_RUN = int(time.time()) % 100_000_000
OLD, NEW, BROKEN = f"90.{_RUN}.0", f"90.{_RUN}.1", f"90.{_RUN}.2"
"""Higher than any real release, and different on every run: uv caches a build
by its version and where it came from, so a second run with the same numbers
would install the *first* run's code and pass or fail on that (it did, on
Linux, 2026-11-19). :func:`Sandbox.same_code` is the check that nothing like
it happens again."""
LINGER_S = 20
"""How long the headless app stays up before it exits by itself."""

BREAK = '''

def run(*, tray: bool = False) -> int:  # this release exists to fail at start
    raise SystemExit("broken on purpose (scripts/update_e2e.py)")
'''

APP = f"""
import os, sys, threading
timer = threading.Timer({LINGER_S}, lambda: os._exit(0))
timer.daemon = True  # or a release that fails at start would linger too
timer.start()
sys.argv = ["steno", "--gui"]
from stenograf.cli import main
main()
"""

DRIVER = """
import hashlib, sys
from stenograf import update

wanted, app = sys.argv[1], sys.argv[2]
assert update.hold_lock() == "held", "could not take the shared lock"
update.check(force=True)
offer = update.offer()
if offer is None or not offer.installable or offer.version != wanted:
    sys.exit(f"expected an installable offer of {wanted}, got {offer}")
folder = update.prepare(offer.version)
print("updater sha256", hashlib.sha256((folder / "updater.py").read_bytes()).hexdigest())
update.hand_over(
    offer.version,
    folder,
    restart=[sys.executable, "-c", app],
    wait_pids=update.launch_chain(),
    launcher_files=update.launcher_files(),
)
print("handed over", offer.version)
"""
"""What the app's Update button does (``gui/updates.py``). With
``--from-published`` this runs on the *published* release, so it may call only
what every release still being updated from has: when a later release changes
one of these functions, keep the old call here beside the new one."""

FACTS = """
import json
from stenograf import update
print(json.dumps({
    "kind": update.describe_install().kind,
    "work": str(update.work_dir()),
    "lock": str(update.lock_path()),
    "launchers": [str(path) for path in update.launcher_files()],
    "offer": getattr(update.offer(), "version", None),
    "offer_if_asked": getattr(update.offer(include_failed=True), "version", None),
    "state": update.load_state(),
    "updater": update._updater.__file__,
}))
"""

LOCK_FREE = """
import sys
from stenograf import _updater, update
try:
    _updater.acquire(update.lock_path(), exclusive=True).close()
except OSError:
    sys.exit(1)
"""


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# -- three releases from one wheel ------------------------------------------


def reversion(wheel: Path, version: str, out: Path, *, broken: bool = False) -> Path:
    """A copy of ``wheel`` that says it is ``version``.

    Rewrites the metadata and regenerates RECORD; file modes are kept, since
    the capture helper inside must stay executable."""
    name, built, *tags = wheel.name.removesuffix(".whl").split("-")
    target = out / ("-".join([name, version, *tags]) + ".whl")
    old_info, new_info = f"{name}-{built}.dist-info/", f"{name}-{version}.dist-info/"
    records: list[str] = []
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(target, "w") as sink:
        for item in source.infolist():
            if item.filename == old_info + "RECORD":
                continue
            data = source.read(item)
            path = item.filename
            if path.startswith(old_info):
                path = new_info + path[len(old_info) :]
            if path == new_info + "METADATA":
                text = data.decode("utf-8")
                assert f"\nVersion: {built}\n" in text, "no Version line in METADATA"
                data = text.replace(f"\nVersion: {built}\n", f"\nVersion: {version}\n").encode()
            if broken and path == "stenograf/gui/app.py":
                data += BREAK.encode()
            entry = zipfile.ZipInfo(path, date_time=item.date_time)
            entry.external_attr = item.external_attr
            entry.compress_type = zipfile.ZIP_DEFLATED
            sink.writestr(entry, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            records.append(f"{path},sha256={digest.decode()},{len(data)}")
        records.append(f"{new_info}RECORD,,")
        sink.writestr(new_info + "RECORD", "\n".join(records) + "\n")
    return target


def publish(listing: Path, versions: list[str]) -> None:
    """The release list the check reads, in the shape of PyPI's."""
    files = [
        {
            "filename": "stenograf.whl",
            "yanked": False,
            "upload_time_iso_8601": "2026-01-01T00:00:00.000000Z",
            "requires_python": ">=3.12,<3.14",
        }
    ]
    listing.write_text(json.dumps({"releases": dict.fromkeys(versions, files)}), encoding="utf-8")


# -- the throwaway home -----------------------------------------------------


class Sandbox:
    def __init__(self, work: Path, uv: str, model_cache: Path | None) -> None:
        self.work, self.uv = work, uv
        self.wheels = work / "wheels"
        self.staging = work / "staging"
        self.listing = work / "releases.json"
        self.tool_dir, self.bin_dir = work / "uv-tools", work / "uv-bin"
        home, scratch = work / "home", work / "t"
        for folder in (self.wheels, self.staging, home, scratch):
            folder.mkdir(parents=True)
        config = work / "uv.toml"
        config.write_text(f'find-links = ["{self.wheels.as_posix()}"]\n', encoding="utf-8")

        def real(*query: str) -> str:
            return subprocess.run(
                [uv, *query], capture_output=True, text=True, check=True
            ).stdout.strip()

        # Asked before HOME moves: the downloads and the interpreters are the
        # machine's own, so a run costs minutes, not gigabytes.
        env = dict(os.environ)
        env.update(
            UV_CACHE_DIR=real("cache", "dir"),
            UV_PYTHON_INSTALL_DIR=real("python", "dir"),
            HOME=str(home),
            USERPROFILE=str(home),
            APPDATA=str(home / "AppData" / "Roaming"),
            LOCALAPPDATA=str(home / "AppData" / "Local"),
            TMPDIR=str(scratch),
            TEMP=str(scratch),
            TMP=str(scratch),
            UV_CONFIG_FILE=str(config),
            STENOGRAF_UPDATE_URL=self.listing.as_uri(),
            QT_QPA_PLATFORM="offscreen",
        )
        if model_cache is not None:
            env["STENOGRAF_CACHE"] = str(model_cache)
        for name in list(env):
            # Nothing may point back at the real user: not their data folders,
            # not their uv tool directory, not the app bundle they started from.
            if name.startswith(("XDG_", "STENOGRAF_APP", "UV_TOOL")) and name != "XDG_RUNTIME_DIR":
                del env[name]
        self.env = env
        self.environment = self.tool_dir / "stenograf"
        scripts = "Scripts" if sys.platform == "win32" else "bin"
        self.python = self.environment / scripts / ("python.exe" if os.name == "nt" else "python")

    def install(self) -> None:
        """What install.sh runs — and the only step that is told where uv's
        tool directory is. The updater has to find that out from the receipt."""
        subprocess.run(
            [self.uv, "tool", "install", "--upgrade", "--python", "3.13", "stenograf"],
            env={
                **self.env,
                "UV_TOOL_DIR": str(self.tool_dir),
                "UV_TOOL_BIN_DIR": str(self.bin_dir),
            },
            check=True,
        )

    def forget_test_source(self) -> None:
        """Remove the one receipt entry that exists only because of this test.

        Anything else under ``[tool.options]`` is something uv recorded about
        the install itself, and that must fail here rather than be hidden."""
        receipt = self.environment / "uv-receipt.toml"
        head, found, options = receipt.read_text(encoding="utf-8").partition("\n[tool.options]\n")
        if not found:
            return
        lines = [line for line in options.splitlines() if line.strip()]
        assert len(lines) == 1 and lines[0].startswith("find-links = ["), options
        receipt.write_text(head, encoding="utf-8")

    def run(self, code: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.python), "-c", code, *args],
            env=self.env,
            capture_output=True,
            text=True,
            check=check,
        )

    def facts(self) -> dict:
        return json.loads(self.run(FACTS).stdout.splitlines()[-1])

    def version(self) -> str:
        done = subprocess.run(
            [str(self.python), "-m", "stenograf", "--version"],
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.split()[-1]

    def same_code(self, wheel: Path) -> None:
        """The installed updater, and the copy that just ran, are this wheel's.

        A pass means nothing if uv served some other build from its cache."""
        with zipfile.ZipFile(wheel) as archive:
            built = archive.read("stenograf/_updater.py")
        facts = self.facts()
        assert Path(facts["updater"]).read_bytes() == built, "the install is not this wheel"
        ran = Path(facts["work"]) / "prepared" / "updater.py"
        if ran.is_file():
            assert ran.read_bytes() == built, "the updater that ran is not this wheel's"

    def leftovers(self) -> list[str]:
        return sorted(path.name for path in self.work.iterdir() if "stenograf-" in path.name)

    def update_to(self, version: str, wheel: Path) -> None:
        """Hand over to ``version``, whose updater must be ``wheel``'s."""
        done = self.run(DRIVER, version, APP, check=False)
        sys.stdout.write(done.stdout)
        if done.returncode != 0:
            raise SystemExit(f"the hand-over failed:\n{done.stdout}\n{done.stderr}")
        with zipfile.ZipFile(wheel) as archive:
            built = hashlib.sha256(archive.read("stenograf/_updater.py")).hexdigest()
        assert f"updater sha256 {built}" in done.stdout, "the updater handed to is not this wheel's"

    def wait(self, what: str, condition, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not condition():
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out waiting for: {what}")
            time.sleep(1)

    def app_gone(self) -> bool:
        return self.run(LOCK_FREE, check=False).returncode == 0


def state_of(work_dir: Path) -> dict:
    try:
        return json.loads((work_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"pending": "unreadable"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", type=Path, required=True, help="the built wheel to test")
    parser.add_argument("--work", type=Path, help="an empty or missing folder to work in")
    parser.add_argument("--model-cache", type=Path, help="reuse these model files")
    parser.add_argument("--keep", action="store_true", help="leave the work folder behind")
    parser.add_argument(
        "--from-published",
        action="store_true",
        help="start from the newest release on PyPI instead of a copy of the wheel",
    )
    args = parser.parse_args()

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH")
    work = (args.work or Path.cwd() / "update-e2e").resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty")
    box = Sandbox(work, uv, args.model_cache)
    began = time.monotonic()
    try:
        (check_published if args.from_published else check)(box, args.wheel.resolve())
    except BaseException:
        log = next(work.rglob("updater.log"), None)
        if log is not None:
            print(f"--- {log} ---\n{log.read_text(encoding='utf-8', errors='replace')}")
        elif next(work.rglob("request.json"), None) is not None:
            # Seen 2026-11-24 with a renamed request field: apply() reads the
            # request before it opens its log, so a broken contract dies silently.
            print("--- no updater.log: the updater died before opening it, most likely")
            print("    reading request.json, so check _updater.py's contract ---")
        prepare_log = next(work.rglob("prepare.log"), None)
        if prepare_log is not None:
            tail = prepare_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            print(f"--- {prepare_log} (end) ---\n{tail}")
        raise
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    say(f"PASS in {time.monotonic() - began:.0f} s")
    return 0


def check(box: Sandbox, wheel: Path) -> None:
    say(f"making releases {OLD}, {NEW} and {BROKEN} (broken) from {wheel.name}")
    reversion(wheel, OLD, box.wheels)
    new = reversion(wheel, NEW, box.staging)
    broken = reversion(wheel, BROKEN, box.staging, broken=True)
    publish(box.listing, [OLD])

    say(f"installing {OLD} into {box.tool_dir}")
    box.install()
    box.forget_test_source()
    assert box.version() == OLD, box.version()
    box.same_code(wheel)
    facts = box.facts()
    assert facts["kind"] == "standard", f"not a standard install: {facts['kind']}"
    assert facts["offer"] is None, facts
    work_dir = Path(facts["work"])
    if sys.platform != "win32":  # there the launchers are the real user's shell links
        # What `steno setup` does: the updater's post-install only rewrites
        # launchers that are already there, so it cannot stand in for setup.
        box.run("from stenograf.shortcut import install_shortcut; install_shortcut()")
        assert box.facts()["launchers"], "setup wrote no launcher to refresh"

    # -- a good update ------------------------------------------------------
    shutil.move(new, box.wheels)
    publish(box.listing, [OLD, NEW])
    say(f"updating {OLD} -> {NEW}")
    box.update_to(NEW, wheel)
    box.wait("the new app to report its start", lambda: "pending" not in state_of(work_dir), 600)
    box.wait("the previous version to be removed", lambda: not box.leftovers(), 60)
    box.wait("the app to exit", box.app_gone, LINGER_S + 60)
    assert box.version() == NEW, box.version()
    assert (box.bin_dir / ("steno.exe" if os.name == "nt" else "steno")).exists()
    box.forget_test_source()
    box.same_code(wheel)
    facts = box.facts()
    assert facts["kind"] == "standard", "the updated install can no longer update itself"
    launchers = {path: Path(path).read_bytes() for path in facts["launchers"]}
    say(f"now {NEW}, started, nothing parked")

    # -- an update whose app dies at start ------------------------------------
    shutil.move(broken, box.wheels)
    publish(box.listing, [OLD, NEW, BROKEN])
    say(f"updating {NEW} -> {BROKEN}, which exits at start")
    box.update_to(BROKEN, wheel)
    box.wait(
        "the previous version to come back and report the failure",
        lambda: BROKEN in (state_of(work_dir).get("failed") or ())
        and "pending" not in state_of(work_dir),
        600,
    )
    box.wait("the failed version to be removed", lambda: not box.leftovers(), 60)
    box.wait("the app to exit", box.app_gone, LINGER_S + 60)
    assert box.version() == NEW, f"not rolled back: {box.version()}"
    box.same_code(wheel)
    for path, before in launchers.items():
        assert Path(path).read_bytes() == before, f"{path} was not put back"
    facts = box.facts()
    assert facts["offer"] is None, "a version that failed here is offered again unasked"
    assert facts["offer_if_asked"] == BROKEN, facts
    say(f"rolled back to {NEW}, started again, {BROKEN} remembered as failed")


def check_published(box: Sandbox, wheel: Path) -> None:
    # The broken release is the lower number: a failed version that is also
    # the newest hides every offer, so the good update could not follow it.
    broken_version, new_version = f"90.{_RUN}.1", f"90.{_RUN}.2"
    broken = reversion(wheel, broken_version, box.staging, broken=True)
    new = reversion(wheel, new_version, box.staging)

    say("installing the newest release on PyPI the way install.sh does")
    box.install()  # find-links is empty here, so this is PyPI's newest
    box.forget_test_source()
    old = box.version()
    assert not old.startswith("90."), f"installed {old}, not a published release"
    say(f"installed {old} from PyPI")
    publish(box.listing, [old])
    facts = box.facts()
    assert facts["kind"] == "standard", f"not a standard install: {facts['kind']}"
    assert facts["offer"] is None, facts
    work_dir = Path(facts["work"])
    if sys.platform != "win32":
        box.run("from stenograf.shortcut import install_shortcut; install_shortcut()")
    launchers = {path: Path(path).read_bytes() for path in box.facts()["launchers"]}
    if sys.platform != "win32":
        assert launchers, "setup wrote no launcher to refresh"

    # -- the new updater puts the published install back ---------------------
    shutil.move(broken, box.wheels)
    publish(box.listing, [old, broken_version])
    say(f"updating {old} -> {broken_version}, which exits at start")
    box.update_to(broken_version, wheel)
    box.wait(
        f"{old} to come back and report the failure",
        lambda: (
            broken_version in (state_of(work_dir).get("failed") or ())
            and "pending" not in state_of(work_dir)
        ),
        600,
    )
    box.wait("the failed version to be removed", lambda: not box.leftovers(), 60)
    box.wait("the app to exit", box.app_gone, LINGER_S + 60)
    assert box.version() == old, f"not rolled back: {box.version()}"
    for path, before in launchers.items():
        assert Path(path).read_bytes() == before, f"{path} was not put back"
    facts = box.facts()
    assert facts["offer"] is None, "a version that failed here is offered again unasked"
    say(f"rolled back to {old}, started again, {broken_version} remembered as failed")

    # -- and updates it ------------------------------------------------------
    shutil.move(new, box.wheels)
    publish(box.listing, [old, broken_version, new_version])
    say(f"updating {old} -> {new_version}")
    box.update_to(new_version, wheel)
    box.wait("the new app to report its start", lambda: "pending" not in state_of(work_dir), 600)
    box.wait("the previous version to be removed", lambda: not box.leftovers(), 60)
    box.wait("the app to exit", box.app_gone, LINGER_S + 60)
    assert box.version() == new_version, box.version()
    assert (box.bin_dir / ("steno.exe" if os.name == "nt" else "steno")).exists()
    box.forget_test_source()
    box.same_code(wheel)
    facts = box.facts()
    assert facts["kind"] == "standard", "the updated install can no longer update itself"
    assert facts["offer"] is None, facts
    say(f"{old} updated itself to this wheel ({new_version}), started, nothing parked")


if __name__ == "__main__":
    sys.exit(main())
