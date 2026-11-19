# PLAN-UPDATE — updating stenograf from inside the app with one click

Status: **built for macOS and Linux on 2026-11-19; not committed, not released.**
Steps 1 to 8 of §8 are done and §11 says what was verified and how, and where
the build departs from the plan. Still open: the Windows measurements (step 0,
skipped for now on Daniel's word — until then Windows shows the notice instead
of the button), the gates on real machines (§9), and one half of the release
gate that cannot exist before a release carrying the updater is published
(§11). The plan itself was written and independently reviewed the same day;
the review's two blockers were reproduced (§2, M9 and M10) and are fixed in the
design below.

Scope: the desktop app notices that a newer stenograf has been released, offers
it, and installs it on the user's click — on macOS, Windows and Linux, with the
same mechanism. The audience is the person who installed with one pasted
command and never opens a terminal again. People who use stenograf from the
terminal get a one-line notice that a release exists.

Not in scope: updating silently, updating `uv` itself, a packaged or signed
installer, a `steno update` command, and a "go back to the old version"
feature for the user.

## 1. What exists today (read from the code, 2026-11-19)

- **One install channel on every system.** `install.sh` / `install.ps1` install
  `uv` if missing, run `uv tool install --upgrade --python 3.13 stenograf`
  (from PyPI, the Python package index), then `steno setup`. Re-running the
  install command is the documented way to upgrade. There is no update check
  and no way to upgrade without a terminal.
- **uv keeps the install in one folder**, the "tool environment"
  (`<uv tool dir>/stenograf`, 1.1 GB, about 14 700 files on macOS). Inside it
  lies `uv-receipt.toml`, uv's record of how the tool was installed, including
  where it put the `steno` command. uv writes that file last.
- **The launchers point into that folder by absolute path** (`shortcut.py`):
  the Windows Start-menu and Desktop shortcuts name its `pythonw.exe`, the Linux
  menu entry names its `python`, and on macOS the file
  `~/Library/Application Support/stenograf/launch-target` names its `steno`.
  The Windows tray icon is also registered against a program path
  (`gui/wintray.py`). So the folder's *location* must not change across an
  update.
- **The macOS app bundle is frozen** (`native/appbundle/README.md`): the
  microphone permission is tied to its exact bytes. Its small launcher program
  only starts whatever `launch-target` names, as a child process, stays alive
  as that child's parent, and shows an alert when the program is missing or
  exits with an error. While that launcher is alive, opening the app again only
  brings the running one forward. An update never touches the bundle.
- **`steno setup` cannot be run unattended as it is**: on macOS it starts the
  capture helper to raise the permission prompts. `steno setup --models-only`
  skips the prompts but also loads the speech model into memory (several GB) to
  prove it works.
- **Models download on first use**, at the start of a meeting, if `steno setup`
  did not fetch them. The app has no screen for model downloads.
- **The app stays open for days**: closing the window puts it in the menu bar
  or tray. `StenografGui.quit_app` already waits for a running meeting before
  quitting; one app per user is enforced by `claim_single_instance`.
- **The code loads its parts late**: about 265 imports sit inside functions, so
  a running app keeps reading program files from disk long after it started.
- **Each release also publishes a generic package** without the capture helper,
  next to the three system-specific ones. It installs and starts everywhere and
  cannot record a meeting.
- **Daniel's own install is not from PyPI.** Its receipt names the checkout
  directory. Self-update must leave such installs alone.

## 2. What was measured (2026-11-19)

On macOS (Apple Silicon, APFS, uv 0.13.7) and on Linux (gpdmini: CachyOS,
x86_64, btrfs, uv 0.13.8). Re-run with `sh eval/update_mechanics.sh` (about a
minute, in a throwaway tool directory; the real install is not touched). All
timings had every package already in uv's download cache, so they are a lower
bound. **Every row gave the same result on both systems**; the numbers in the
table are the Mac's, and Linux's are below it.

| # | Question | Result |
|---|---|---|
| M1 | What happens to a process that is running while its environment is upgraded? | The upgrade succeeds. The process then holds 0.6.0 in memory and loads the rest from 0.7.0 files. It did not crash in this one trial; it is a mixed-version process all the same. |
| M2 | Does an in-place upgrade move the environment? | No. Same folder, same Python path. |
| M3 | What does uv remember in the receipt? | An exact version pin, `--constraints` and `--exclude-newer` are all stored, and a later `uv tool upgrade` obeys them ("Nothing to upgrade"). The installer's command without those flags writes a clean receipt again. |
| M4 | What does an upgrade leave behind when the index is unreachable? | It fails before changing anything. |
| M5 | Can the old environment be set aside and brought back? | Yes. Renamed away, a fresh install took its place (under 1 s), and renaming the old one back gave a working 0.6.0 that uv also lists as 0.6.0. A parked folder *inside* uv's tool directory makes uv print a warning, so it must be parked outside. |
| M6 | Can the new version be downloaded and tried before anything installed changes? | Yes. `uv tool run --from stenograf==0.7.0 steno --version` ran the new version from uv's cache while the installed one stayed 0.6.0. |
| M7 | What does an install without network pick? | The newest version *that is in the cache*, not the newest on the index (tested from an empty cache with a small package, httpie: 3.2.3 was installed although 3.2.4 exists). Asked for a version the cache does not hold, it fails cleanly. |
| M8 | Can a Python outside the environment be found from inside it? | Yes. `sys._base_executable` is uv's own Python, outside the tool environment. |
| M9 | What does an install leave behind when it is killed halfway? | A folder with **no receipt** — uv writes it last. Killed late, the folder is complete (1.1 GB) and its `steno` runs; killed early, it is a fragment. uv calls either a "malformed tool". |
| M10 | Does re-running the installer's command repair a broken state? | **No.** With the folder half-installed, and also with the folder gone, it fails — in the second case always with "Executable already exists: steno (use `--force` to overwrite)". With `--force` it repairs both. |
| M11 | Does a lower bound in the receipt (`stenograf>=X`) block later upgrades, as a pin does? | No (httpie: receipt `>=3.2.3`, `uv tool upgrade` moved it to 3.2.4). |
| M12 | Does one lock file do what U8 needs? | Yes. While two processes held it shared, the exclusive lock was refused; still refused after one of them was killed; granted once both were killed (nothing cleaned up — the system released it); and a new shared lock was refused while the exclusive one was held. |
| M13 | Does a detached process outlive the one that started it? | Yes — started on the base Python from a process that exits at once. On Linux also from inside a systemd scope, which is how a desktop starts an app. |

Linux in numbers: the environment is 704 MB in about 5 500 files (no MLX
there); a fresh environment took 0.18 s, the in-place upgrade 5.6 s including
the index query; the trial run from the cache 0.9 s. Not covered on Linux: a
lock file on a network home directory, and everything that needs a desktop
session (§9).

Measured by the reviewer and not repeated here: `uv tool run --python 3.14`
runs stenograf 0.7.0 although the release declares "below 3.14" — uv ignores
that upper limit, as `pyproject.toml` already records.

Not measured, taken from elsewhere: on Windows, upgrading a tool that is in use
fails with "file is being used by another process" and can leave uv believing
it is up to date (uv issue 11930, one report, about `ruff`). Two other Python
tools (aider, seiscat) refuse to self-update on Windows and print the command
instead.

## 3. Decisions, with reasons

**U1 — The update goes through uv and PyPI, the channel that installed the
tool.** No second distribution channel, no packaged app. The per-system
updaters (Sparkle, WinSparkle, AppImage, Homebrew, winget) all need a signed,
packaged application, which this project has ruled out: notarization, and
PySide6's licence, which a normal pip dependency satisfies and a bundle does
not.

**U2 — The app offers; the user decides.** Nothing is installed without a
click on "Update". An update is never started while a meeting, a transcription
or a notes run is active; never as a side effect of quitting, logging out or
shutting down (the system would kill it halfway); and nothing about updates may
delay the app's start — opening the app one minute before a meeting is the
normal case.

**U3 — Only a standard install updates itself.** Standard means: the running
program lives in a uv tool environment whose receipt asks for `stenograf` from
the index, with no version requirement or only a lower bound, at most the empty
extras `gui` and `ollama`, and no constraints, overrides or `exclude-newer`. A
missing Python entry is fine (older installs have none). An exact pin, a
directory, a git or URL source are things a person chose (M3): a pinned install
gets a notice naming the new version, a checkout or git install gets nothing, a
pip install gets a notice without a command.

**U4 — Two phases: prepare while the app is open, apply after it has closed.**
*Prepare* downloads the new version into uv's cache and runs it once from there
(M6); nothing installed changes, the user sees progress and can cancel, and a
release that cannot be resolved, downloaded or started on this machine fails
here, harmlessly. *Apply* is the short part that needs the app closed. The
split exists because of M1 (a running app must not have its files replaced)
and the Windows file locks.

**U5 — Apply replaces the whole environment and keeps the old one until the
new one has proven itself.** The old folder is renamed away ("parked"), a fresh
install goes into the same path, and only after the new app has started is the
old folder deleted. If anything fails, the old folder is renamed back — exactly
the bytes that worked, with no network and no dependency resolution (M5). The
rules that make the failure path work:
- The parked folder is a neighbour of uv's tool directory: same disk, so the
  move is a rename, and not inside the tool directory (M5).
- Before installing, the updater writes down that the environment path is now
  empty. Whatever stands there afterwards is its own doing, receipt or not
  (M9), and on rollback it is *renamed* to a trash neighbour, never deleted on
  the way back. The trash is removed later.
- Every rename is retried for up to a minute. On Windows a virus scanner that
  is reading a just-written file blocks the rename of its folder for a moment.
- The launcher files (`launch-target`, the Linux menu entry, the two Windows
  shortcuts) are copied before the swap and put back on rollback, because the
  new version rewrites them.
- Between parking and the end, the updater ignores requests to terminate
  (macOS, Linux), so a logout lets it finish or roll back. Windows cannot do
  this.

Rejected alternatives:
- *Upgrade in place and reinstall the old versions on failure*: the reinstall
  needs the network or an intact cache at the worst moment, and needs
  constraints, which stick in the receipt (M3).
- *Install into a second folder and switch the launchers over*: the path
  changes with every update, and every launcher embeds it (§1).

**U6 — The updater that runs is the new release's, not the old one's.** An
updater built into version N would install N+1, and any mistake in it would
stay on users' machines for good. Prepare has already put the new release into
uv's cache, so the app copies the updater script out of *that* copy and runs
it. A faulty updater is then repaired by the next release. The script uses only
Python's standard library and runs on uv's own Python (M8), not on the
environment it replaces — so it must keep running on the oldest Python any
install may have (3.12).

**U7 — What an old version must be able to rely on forever is therefore
small.** Three things, written down as a contract and never changed:
1. *Prepare*: in the trial copy of the new version the app runs
   `python -m stenograf._updater prepare --out <folder>`. The new version
   checks itself (U9) and leaves `updater.py` in the folder. Exit code 0 or 1,
   plus one line of text for the user.
2. *Apply*: the app writes `request.json` into the folder and starts
   `<uv's Python> -I <folder>/updater.py apply <folder>/request.json`. The
   request carries a format number, so a newer updater knows what an older app
   wrote.
3. *Result*: the updater leaves `result.json` with four fields — success yes or
   no, the version, one sentence for the user, the path of the log. An old app
   that was rolled back only has to show the sentence.
Everything else — self-check, launchers, models, locks during the swap,
rollback, clean-up — is the new release's own code talking to itself.

**U8 — "Is stenograf still running?" is answered by one file lock.** One lock
file per environment, outside the environment. Every stenograf process holds
it *shared* for as long as it lives; the system releases it when the process
ends, crashes included. The updater takes it *exclusively* for the swap. It
cannot get it while anything runs — which covers a meeting in a terminal next
to the app — and nothing can start while it holds it: a process that cannot get
its shared lock says "Stenograf is being updated, try again in a minute" and
exits. The lock file sits at a fixed place derived from the environment's path,
deliberately ignoring `$STENOGRAF_DATA`: the app started from the Dock and a
terminal session do not share environment variables (the same reason
`launch_target_path` is fixed). If locking is not possible at all (some network
home directories), the app runs normally and self-update is off.

**U9 — The new version checks itself, including what the generic package
lacks.** Prepare's self-check: the program imports, the interface loads without
a display, and on the three systems that have their own package the capture
helper is present — otherwise an install caught in the middle of a release
upload gets the generic package, starts fine and cannot record (§1). It then
downloads model files the new version needs and this machine lacks. It does not
load the speech model: that costs several GB next to the running app on every
update. No hardware, no permission prompt.

**U10 — The check asks PyPI once a day, in the background.** One request to
`https://pypi.org/pypi/stenograf/json`, at start if the last check is older
than 24 hours and then every 24 hours while the app runs. A release is offered
if it is a final release (no pre-releases), is not withdrawn ("yanked" — the
brake for a bad release), is newer than the installed one, and its last file
was uploaded more than an hour ago. Failure is silent. The request carries no
user data; the user agent is `stenograf/<version>`. `[updates] check = false`
in settings.toml switches it off — for the app and the terminal alike (U20) —
and the README says that the check exists. An environment variable can point
the check at another address; CI needs it (§8, step 7), so it must be in the
first release that has this feature.

**U11 — Apply installs exactly what prepare tried.** Same explicit Python in
both. Apply runs without network and asks for `stenograf>=<new>`: uv then takes
the newest version in the cache, which is the one prepare just tried (M7), and
fails cleanly if the cache was emptied in between instead of quietly
reinstalling the old version. Afterwards the updater asks the installed program
for its version; anything but the expected one is a failure. The lower bound
ends up in the receipt and does not block later upgrades (M11). No pin, no
constraints, no `exclude-newer` — each would (M3).

**U12 — uv is told where the install lives.** uv finds its tool directory
through environment variables and its own configuration, which the app started
from the Dock or the Start menu may not see. The updater therefore passes both
locations explicitly: the tool directory is the parent of the environment
folder, the directory for the `steno` command is the one the receipt records.
uv itself is looked up on PATH and in the installer's default location, never
installed or updated by the app; if it is missing or too old, the app shows the
notice with the manual command instead of the button.

**U13 — The Python version stays.** If the new release still supports the
Python the install runs on, the update keeps it. If not, there is no
self-update for that release, only the notice with the install command. That
happens about once a year, and the install command already names the version.

**U14 — The app waits for the whole chain to end, and ends cleanly.** On macOS
the app is the child of the bundle's launcher; on Windows it is the child of a
small launcher inside the environment. The request names both process numbers
and the updater waits for both: on macOS a still-living launcher would turn the
restart into "bring the old app forward" (§1), on Windows a running launcher
blocks the rename. The app exits with code 0 on purpose at hand-over, so the
macOS launcher shows no alert.

**U15 — The new app is started the way the user starts it.** macOS: `open` on
the bundle at the location the launcher reported (`STENOGRAF_APP_BUNDLE` — the
bundle may have been moved), never the program directly, or the microphone
permission would be attributed to the wrong program. Windows: the Start-menu
shortcut. Linux: the command of the menu entry. A `--tray` start is kept.

**U16 — A version that failed here is not offered again on its own.** It is
remembered; the next newer release is offered normally, and "Check for
updates" in Settings tries again on request. Without this a broken update
would be offered every day.

**U17 — The updater checks what it is about to move or delete.** Before
parking it verifies that the folder is a uv tool environment whose receipt
names stenograf. Afterwards it only ever touches folders it named itself: the
parked one, the trash one, and the environment path it recorded as empty.

**U18 — The install command can repair a broken install.** Today it cannot
(M10), and it is what the macOS alert and the README send people to. Both
installers retry with `--force` when the normal command fails.

**U19 — Rollback restores the program, not the user's data.** A release that
changes how something is stored (the voiceprint file, settings) either keeps it
readable for the previous version or changes it only after the update is
confirmed. This is a rule for releases, not code in the updater.

**U20 — In the terminal there is a notice, and no update command.** When a
newer release is known, a command prints one line on the error stream as it
ends: the version, the install command, and "not while a meeting is running"
(the install command has no rollback and does not see a meeting in another
terminal, M1). Rules:
- Only in an interactive terminal — never in a pipe, a script or CI, by the
  same test bare `steno` already uses to decide whether to open the window.
- At most once a day.
- The check runs in the background of the long-running commands (`steno
  start`, `transcribe`, `notes`) and never delays any command; short commands
  only read what the last check stored. `steno doctor` shows the same as a row.
- `[updates] check = false` is the one switch for all of it: no request to
  PyPI from the app or the terminal, no line on the home screen, no notice.
Without the notice, someone who only uses the terminal never learns that a
release exists. `steno update` is left out: the install command already runs
outside the environment, which is the hard part of self-update, and handing a
terminal over to the updater has no clean form on Windows. It can reuse the
same updater once the app's path has passed the gates in §9.

## 4. The flow

1. **Check** (background thread, U10). Result and time are stored.
2. **Offer.** A slim line on the home screen: "Stenograf 0.8.0 is available ·
   What's new · Update", and the same as an entry in the menu-bar menu. "What's
   new" opens the release page in the browser. No dialog, no notification, and
   nothing at all while a meeting is running.
3. **Prepare** (after the click, app stays usable): the trial run of U7.1 with
   the Python of U13. Shown as "Preparing update…" with Cancel. If a meeting is
   started meanwhile, prepare is cancelled.
4. **Hand over.** If the app is still idle when prepare has succeeded, it
   writes the request, starts the updater detached, and quits with code 0. If
   it is no longer idle, the line reads "Ready — restart to update" and waits
   for a click. Quitting in that state just quits (U2).
5. **Apply** (updater): wait for the process chain to end (U14); take the lock
   exclusively (U8); verify and park the old environment, copy the launcher
   files (U5, U17); install without network (U11); check the version; run the
   new version's own launcher refresh.
6. **Restart and confirm.** Release the lock, start the app (U15). The new app
   writes a "started" marker once its window is built; the updater then deletes
   the parked folder and writes the result. If no marker appears and nothing
   holds the shared lock, the new app has died: the updater rolls back and
   starts the old app.
7. **Report.** The app shows one line at its next start: "Updated to 0.8.0 ·
   What's new", or "The update to 0.8.0 did not work and was undone" with the
   reason and the log file.

## 5. What can go wrong

| Failure | What the user sees | State afterwards |
|---|---|---|
| No network, PyPI down, check fails | Nothing | Unchanged |
| Download or resolution fails in prepare | "Could not prepare the update", reason, app keeps running | Unchanged |
| New version does not start here, or arrives without the capture helper (prepare) | Same | Unchanged; version is not offered again (U16) |
| A terminal meeting is running | "Stenograf is in use elsewhere — try again later" | Unchanged |
| The app does not exit, or exits with an error | After a wait: "update could not start" | Unchanged |
| The old folder cannot be renamed (Windows, a file is held open) | After a minute of retries: app restarts, "update could not start" | Unchanged |
| uv's cache was emptied between prepare and apply | App restarts in the old version, "prepare again" | Old folder renamed back |
| Install fails or is incomplete after parking | App restarts in the old version, failure line | Half-installed folder moved to trash, old folder renamed back |
| The wrong version got installed | Same | Same |
| New app crashes at start | Old app starts, failure line | Rolled back, launcher files restored |
| New app hangs at start | Nothing automatic | New version installed, parked folder kept; the next successful start cleans up |
| The user opens the app during the swap | "Stenograf is being updated"; on macOS, if the folder is momentarily absent, the launcher's "not installed" alert | Fine on the next try |
| Logout or shutdown during the swap | macOS, Linux: the updater finishes or rolls back first. Windows: see next row | — |
| The updater is killed or the power fails during the swap | macOS alert "Stenograf is not installed"; elsewhere the launcher does nothing | **Broken until the install command is re-run** (which then works, U18). A later start removes the leftover folders. |
| Rollback itself cannot rename (Windows, after a minute of retries) | Same as the row above | Same |

The last two rows are the accepted hole. Its length is the install time: under
a second on the Mac and on Linux, unknown on Windows (step 0). Closing it completely would
need a launcher that can repair the install, and the macOS launcher cannot be
changed.

## 6. Privacy and trust

- The app is advertised as fully local. The daily check is a new, recurring
  network call; it reveals an IP address and the stenograf version to PyPI and
  nothing else. It is documented and can be switched off.
- Trust is the same as at install time: HTTPS to PyPI, where only this
  project's release workflow can publish, and uv verifies each file against the
  index's checksums. No extra signature scheme; its key handling would cost
  more than it protects for a tool of this size.
- Because every update needs a click, a bad release does not spread by itself;
  withdrawing it on PyPI stops the offer.

## 7. Where the code goes

- `stenograf/update.py` — the library: the check, the "standard install" test,
  stored state, starting prepare, hand-over, reading the result. Standard
  library only. Both front-ends call it; neither contains update logic.
- `stenograf/_updater.py` — the script of U6 and U7: `prepare` (may import
  stenograf) and `apply` (may not). Its name and its two commands are the
  contract.
- `stenograf/cli/` — taking the shared lock at the group's entry; the notice
  of U20 and the `steno doctor` row.
- `stenograf/gui/` — the line on the home screen, the menu-bar entry, the
  "started" marker, a "Check for updates" row in Settings.
- `install.sh`, `install.ps1` — the `--force` retry (U18).
- `tests/fake_uv.py` — a stand-in for uv that can succeed, fail early, or stop
  halfway leaving a folder without a receipt, in the manner of
  `tests/fake_stenocap.py`.

## 8. Steps

0. **Measure on Windows before building** — the assumptions that would change
   the design (Linux is done, §2):
   - M1 to M11 there; the script is POSIX shell, so it needs a port or Git
     Bash;
   - how long a fresh environment takes to create (the length of the hole in
     §5);
   - whether the environment folder can be renamed once the app and its
     launcher have exited, and whether a just-installed folder can be renamed
     away again, with Defender active;
   - whether a detached process started from `pythonw.exe` survives the app's
     exit, without a console window (M13's Windows form);
   - whether a shared/exclusive lock behaves as in M12 (Windows has no `flock`;
     it needs `LockFileEx`).
   One Linux item is left for whoever has such a machine: the lock on a network
   home directory. Until then U8's fallback applies there.
1. Installers: the `--force` retry, with a CI test that starts from both broken
   states of M10. Useful on its own, so it goes first.
2. Library: check, version comparison, standard-install test (one fixture per
   receipt shape from M3, M11 and U3), stored state. Tests. No UI.
3. The lock at every entry point (U8). Tests, including a crashed holder and a
   file system that refuses locks.
4. The updater script against `fake_uv`: success, every row of §5 that can be
   simulated, and the folder checks of U17. Run on Python 3.12 as well.
5. Prepare and hand-over in the library; the home-screen line, menu-bar entry,
   marker and Settings row. Offscreen tests and screenshots.
6. The terminal notice (U20): the line at the end of a command, the background
   check in the long-running commands, the `steno doctor` row. Tests for each
   rule — a pipe stays silent, a second run the same day stays silent, the
   switch silences everything and sends no request.
7. CI, on all three systems, and as a gate before a release is published:
   - the last published release, installed with the installer, updates itself
     to the freshly built package (served from a local index);
   - the freshly built package updates itself to the same code with a higher
     version number. Without this second half a release whose own check or
     hand-over is broken would strand its users silently.
8. README: the check, the switch, the manual command. The first release with
   this feature cannot be reached by self-update; announce it with the install
   command.
9. Gates on real machines (§9).
10. Later, only if wanted: `steno update`.

## 9. Gates (real machines, by observation)

- **macOS**: after an update started from the app in `~/Applications`, the
  first meeting records without a new microphone prompt; no alert appears when
  the old app exits; the same with the bundle moved to another folder.
- **Windows**: no console window flashes; a pinned taskbar icon still starts
  the app afterwards; the tray icon keeps its place; Defender active
  throughout.
- **Linux**: the menu entry still starts the app; the restarted window carries
  its icon.
- **All**: three deliberately broken releases — one that fails its self-check,
  one that crashes at start, one whose install is killed halfway — each end in
  the old version with the failure line; and after the third, with the updater
  killed too, the install command repairs the machine.

## 10. Decided by Daniel (2026-11-19)

- **The daily check is on by default**, documented, with the off switch (U10).
- **One click does everything**: "Update" prepares and, if the app is still
  idle when that has finished, restarts by itself (§4, step 4).
- **No waiting period beyond the hour** that covers a release still uploading
  (U10).

- **Terminal users get the notice, not an update command** (U20), and the one
  setting governs it: with the check switched off, the terminal neither checks
  nor says anything.

## 11. Built 2026-11-19 — what was verified, and where it departs from the plan

**Verified by running it** (not by reading it):

- The whole update on a real uv, twice: `scripts/update_e2e.py` installs a
  wheel as a standard uv tool install in a throwaway home directory, lets the
  installed program find, prepare and hand over an update to a second release,
  and then to a third that exits at start. macOS (uv 0.13.7): updated,
  started, nothing left parked; then rolled back, the old version started
  again, the launcher files as before, the broken version remembered as
  failed — 64 s in all. Linux (gpdmini, uv 0.13.8): the same, 48 s. In both
  runs the updater was never told where uv's tool directory is and took it
  from the receipt (U13), and the machine's real install was untouched.
  One earlier Linux run passed on the wrong code: uv served the previous
  run's build of the same made-up version from its cache. The script now
  uses new version numbers on every run and compares the installed updater
  with the wheel's, byte for byte.
- The installer's repair (U18) on Linux: `install.sh` run against an install
  whose receipt was deleted, and against one whose folder was deleted, printed
  uv's "Executable already exists", reinstalled, and ended with a working
  `steno` both times.
- The home-screen line in every phase, by screenshot of the offscreen app.
- 1 250-odd unit tests, among them one per row of §5 that can be simulated
  against `tests/fake_uv.py`.

**Not verified:** everything on Windows (lock, detached start, rename, the
installer's new retry and the new step in `install.yml`); the restart through
`open Stenograf.app` and the microphone permission after it (the first gate of
§9 — it needs the real launcher in `~/Applications`); the Settings row by eye
(it lies below the fold of a long report; its bindings are covered by the QML
warning test); the two new jobs in `release.yml` have not run on GitHub yet.

**Departures from the plan, and additions:**

- **Windows offers the notice, not the button** (`update.SWAP_MEASURED`),
  until step 0 is done. `STENOGRAF_UPDATE_UNMEASURED=1` switches the button on
  for the measurement itself; `release.yml` has a Windows job that runs the
  whole update with it and is allowed to fail.
- **U9: the speech-recognition weights are not fetched during prepare**, only
  the plain model files (voice-activity, diarization, speaker embedding).
  Fetching the weights means loading that backend's libraries beside the
  running app; they still download on first use.
- **Only a version that was installed and taken out again is marked as
  failed** (U16 says any failure is). A failed download is usually the
  network, and an update that never began — a meeting running in a terminal, a
  file in use — says nothing against the version. The result file carries
  `rolled_back` for this; such a version is withheld until the next release or
  a click on "Check for updates".
- **A launcher refresh that fails is logged, not fatal**: the environment's
  path has not changed, so the old launchers still start the app.
- **Added: a second lock, held by the updater for its whole life**
  (`updater.alive`). The main lock is free while the updater waits for the new
  app's first start — exactly when the previous version is still parked. Found
  while preparing the end-to-end run: the freshly started app's clean-up of
  "leftovers" would otherwise have deleted the parked version under a waiting
  updater. The same lock makes a second updater leave the first one to it.
- **Added: a crash inside the swap rolls back** instead of leaving the
  machine without an install at the path every launcher names — and does so
  while still holding the lock, so that no app can start from an environment
  that is about to be moved away. A crash before anything was moved starts the
  untouched app again.
- **Added: a start counts as dead only after the lock has been free for ten
  seconds.** A `steno` command in a terminal holds the same lock for a moment;
  without the pause its exit read as the app's, and a good version was rolled
  back while it was still starting (reproduced by the code review, then by a
  test that fails with the pause set to zero).
- **Added: deleting the parked or failed folder can be finished later.** The
  receipt is what marks such a folder as ours, and a plain recursive delete
  removes it first on APFS; a logout in the middle left a folder no later
  update dared remove, which blocked every later update. The marks now go
  last.
- **Added: an app started directly (no launcher in between) that exits with an
  error is rolled back at once**, not after the 45 s the lock-based detection
  needs (measured on Linux: 45 s before, under a second after). Through a
  launcher — macOS and Windows — it is still the 45 s, plus the ten-second
  pause. The request says which of the two it is (`direct`), because a
  launcher's exit code means nothing: `open` returns 0 whatever happens, and
  `explorer.exe` returns 1 when all went well.
- **Step 7, first half, is missing**: "the last published release updates
  itself to the freshly built package" needs a published release that contains
  the updater. Add it to `release.yml` with the first release after that one.
  The second half (the fresh build updates itself to the same code under a
  higher version, and survives a broken one) gates the publish on macOS and
  Linux.

**Learned on the way (measured 2026-11-19, uv 0.13.7, macOS):**

- uv writes a `find-links` setting into the receipt's `[tool.options]` whether
  it came from `UV_FIND_LINKS` or from a config file. So an install made on a
  machine with a package mirror configured counts as customised and gets the
  notice, not the button. That is the safe side — the updater's fresh install
  would not repeat options that came from the environment — but it means such
  users never get one click. Reading the recorded options back into the
  updater's install would lift that; not built.
- **Inferred, not observed:** on macOS a new version that dies at start is
  started through the bundle's launcher, which shows its own alert for a
  failed start and stays alive until that is dismissed. The updater then rolls
  back and runs `open` again, which may only bring the alert forward; the
  restored version would then start on the user's next click and report the
  failure. To be looked at with the second gate of §9.

**The code was reviewed independently after it was built** (2026-11-19). Four
defects were reproduced and nine more reported; all but the following are
fixed and have a test. Left as they are, knowingly:

- A new app that needs more than 45 s to get from its launcher to its first
  line of Python is judged dead and rolled back while it starts. It then finds
  the lock taken, says "being updated" and exits; the old version is back and
  the next click starts it. Not observed; the 45 s can be raised if a slow
  machine shows it.
- What two releases' *apps* must agree on (the state folder, its per-install
  subfolder, the lock and marker file names) is now written into
  `_updater.py`'s contract and pinned by a test, not passed explicitly. Passing
  it would need a channel from the updater to an app started through `open`,
  which carries no environment.
- An install made with `uvx` or pipx is told it "was not installed with the
  install command", which is true and not specific.
- A terminal-only user never runs the app's clean-up of a parked folder. Only
  the app's updater ever parks one.
