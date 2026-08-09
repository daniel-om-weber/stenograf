# stenograf — remaining work

**This file holds only what is not built yet.** Everything shipped —
phases 0–7, the architecture and model-choice research, the AEC design, the
code-cleanup backlog, the biasing evaluation — was removed on 2026-07-25 and
lives in this file's git history (`git log --follow -p PLAN.md`, plus the
deleted `PLAN-AEC.md` and `PLAN-CLEANUP.md`). Locked product scope and
platform decisions are in `CLAUDE.md`; measured evidence for the shipped
defaults is in `eval/README.md` and in the code's own docstrings.

**No side-plan is live.** One draft awaits its scoping session:
`PLAN-DIARIZATION-SPEED.md` (2026-08-09, discussion notes — finalize
diarization measured at RTF 0.049 with 84 % in sequential batch-1 embedding
calls; parallelize/batch first, online precompute decided on real-meeting
finalize-split + watt numbers). `PLAN-DIARIZATION.md` (2026-08-09, all five steps
measured to a verdict — the declined list below carries the survivors; read
`eval/diarization-sota-2026.md` and `eval/README.md`'s corpus-harness
sections before touching `diarization/`, `voiceprints.py`, or evaluating any
diarization or speaker-embedding model), `PLAN-CAPTURE-HELPER.md` (2026-08-02, built
— both halves shipped and the arrival-stamped transports are gone),
`PLAN-LINUX.md` (2026-07-26), `PLAN-WINDOWS.md` and `PLAN-ASR-CHALLENGER.md`
(both 2026-07-27), and `PLAN-NOTES-MARKDOWN.md` + `PLAN-MEETING-PRESETS.md`
(both 2026-07-30, built) were deleted; each one's evidence, decisions and
rejected alternatives survive under `git log --follow -p <file>`, and what
stayed open out of them is folded into the sections below. Read the Windows
history before re-measuring anything on that platform — in particular its last
section, which is the observation recipe for a real Windows desktop session
(screenshot DPI, SAPI voice selection, the German-locale traps, how to drive
the TUI without a pty). Read the capture-helper history before touching
capture, `aec.py` or the wheel matrix — its evidence section
(`eval/wasapi_timestamps.py` re-runs in twelve seconds) and its
decided/not-candidates lists are the part not to re-derive.

What ships today: `steno start` (live captions → diarized transcript → notes)
on macOS, Linux and Windows, published to PyPI as `stenograf`, driven either by
the CLI or by the Qt desktop app (bare `steno`, `steno --gui`, or the app
icon). The Textual launcher was retired 2026-07-30 with step 7's flip.

---

## Phase 8 — native GUI (Qt Quick). CLOSED 2026-07-30: all seven steps shipped; only step 1's deferred watt measurement remains open (below).

**Decision: the launcher becomes a real desktop application built on Qt Quick
(PySide6), and `Stenograf.app` is installed locally by `steno setup`.** This
absorbed the old Phase 7 Tier 2 (tray + packaged installers). **Step 7 shipped
2026-07-30: PySide6 is a base dependency, bare `steno` in an interactive
terminal with a display opens the window, and the Textual front-end is retired
entirely** — Daniel declared the use-gate met (the desktop app / tray icon is
the default way of using the tool) and picked retirement over a no-display
fallback: one real UI (Qt), one thin line-oriented CLI. The full design,
review findings and accepted costs are in the deleted `PLAN-GUI-DEFAULT.md`
(`git log --follow -p PLAN-GUI-DEFAULT.md`); in one breath: a double dispatch
gate (TTY + display seam) instead of a graceful-error promise, because a Qt
platform-plugin failure is an uncatchable C++ abort; `--plain` a hidden no-op;
`--gui` gate-free forever (the frozen stub); the glibc-2.39 arm64 install
floor accepted; setup's terminal-launcher arms deleted with the retirement
calls kept.

**The gate list is green (observed 2026-07-30).** The plain-mode replay
meeting, the pipe/script help check, and Daniel's three launch gestures on
this Mac — bare `steno` from a terminal, the app icon, and bare `steno` with
the tray app already running (it printed its one line) — all passed. The one
unobserved case stays optional and blocked on enabling Remote Login: an SSH
bare-`steno` to see the macOS arm of the display heuristic on the case it
bets on.

**Built 2026-07-25 (steps 3 + 4): `stenograf/gui/` — all six screens, opt-in
behind `steno --gui`,** with PySide6 as the optional `[gui]` extra; step 5 put
them behind a double-clickable `~/Applications/Stenograf.app`. Getting
there moved the shared work into the library, which is what keeps two
front-ends from drifting: `stenograf/flow.py` (meeting-request resolution, the
meeting run, transcribe, notes, the settings report) and
`stenograf/captions.py` (the live-caption line rules). The Textual screens were
rewritten onto both and are now as thin as the Qt ones. Tests are headless
(`tests/test_gui.py`, `QT_QPA_PLATFORM=offscreen`): every page is instantiated
with its real controller and the Qt message handler must stay silent — a QML
binding error is a warning, not an exception, so an unwatched app "works" while
rendering nothing.

Landmines paid for while building it, beyond the spike's (they outlive the
retired Textual screens — the Qt app keeps all three):

- **A QML property may not be named `onSomething`** — `readonly property color
  onAccent` is parsed as a signal handler and kills the whole singleton with
  "Cannot assign a value to a signal".
- **Screen objects must outlive the engine.** Parent them to the shell and the
  shell to the `QGuiApplication`; the reverse teardown order re-evaluates every
  binding against a null object and floods stderr with TypeErrors at exit.
- **`ComboBox.textRole`/`valueRole` against a Python list of maps** is a
  silent-`undefined` risk; `Combo.qml` reads the selected entry out of the model
  by index instead.

### Why Qt, and why not the alternatives

A `textual-serve` + `pywebview` route was costed first — by far the cheapest,
since every existing screen would work unchanged in a native window — and
rejected: it renders our TUI inside a browser engine, the wrong side of the
no-web-UI lock, and its Linux GTK backend needs PyGObject anyway. A second
route, our own terminal renderer (custom Textual `Driver` → `pyte` → a
canvas), was rejected for making us the owner of a terminal emulator's bug
surface (unicode widths, IME, HiDPI, copy/paste) in exchange for no new
capability.

Toolkit comparison (2026-07-25, all facts checked against PyPI):

| | install cost | rendering | licence | verdict |
|---|---|---|---|---|
| Qt Quick (PySide6) | ~110 MB (`PySide6-Essentials`) | GPU scene graph | LGPL-3.0 ✓ | **chosen** |
| Tk 9 + sv-ttk | nominally 0 | CPU | MIT ✓ | runner-up, blocked (below) |
| Dear PyGui | 3 MB | GPU (ImGui) | MIT ✓ | tooling aesthetic; immediate mode redraws continuously |
| Slint | 16 MB | GPU | **GPLv3 or commercial** | disqualified — MIT project; also *no stable Python release* (latest `1.17.1b2`) |
| Toga | small | native widgets | BSD ✓ | disqualified — `toga-gtk` needs `pygobject`+`pycairo` (sdist-only), `toga-winforms` needs pythonnet |

Qt wins on more than looks: **`QSystemTrayIcon` speaks StatusNotifierItem on
Linux natively**, which deletes the 300–500 lines of hand-rolled D-Bus
(`StatusNotifierItem` + `com.canonical.dbusmenu` over `jeepney`) that a
pip-only tray backend would otherwise cost — PyGObject is sdist-only, so
pystray's Linux backends are unreachable from a wheel-only install. The
menu-bar/taskbar mode therefore stops being a separate project and becomes a
mode of the same app. Coverage caveat: KDE/XFCE/Cinnamon/Ubuntu-GNOME have
SNI, but stock GNOME needs the AppIndicator extension — so the tray stays an
enhancement, with the desktop launcher as the guaranteed entry point.

**LGPL, deliberately:** PySide6 is LGPL-3.0 (PyQt6 is GPL and would relicense
this MIT project). A normal pip dependency satisfies the relinking obligation;
a future PyInstaller-bundled build would not, and needs real attention.

### Distribution: no Developer ID required

macOS quarantine is applied by the *downloading* program, so a file written
locally is never assessed by Gatekeeper. `steno setup` exploited this first for
`Stenograf.command` and now (step 5) for the full `.app` bundle — custom icon,
Dock presence, menu-bar mode — with no Developer ID, no notarization, not even
a real signature (ad-hoc suffices). **The $99/yr gates only a *downloadable*
artifact.**

Consequence, **measured** in step 2 below: TCC grants move off Terminal.app onto
our bundle. That is better hygiene ("Stenograf would like to access the
microphone"); `setup` now says so in as many words, because the grant it takes
for the terminal is not the app's. An ad-hoc bundle's TCC identity is weak in
one specific, load-bearing way — TCC stores
**`cdhash` and nothing else** as the requirement, no identifier and no anchor.
So the bundle must be a thin wrapper that is **byte-identical forever**, and it
must **spawn** the installed `steno` as a child; a wrapper that `exec`s it loses
the bundle identity altogether. Windows and Linux need no equivalent trick: a
locally written `.lnk` carries no Mark-of-the-Web, and Linux signs nothing.

### Prototype results (2026-07-25)

Both toolkits were built against the real launcher information architecture
(home menu + live meeting screen with committed captions, dim interim tail,
phase-coloured header) sharing one fake feed, ~860 lines total, in
`prototypes/gui/` (untracked; `prototypes/gui/README.md` has the full landmine
list). Qt produced the intended design directly. Tk's *meeting* screen came
out presentable — colour-tagged `Text`, tab-stop alignment, clean type — but
its *home* screen renders flat, because `highlightthickness` /
`highlightbackground` does not reliably draw a 1 px card border on macOS and
surface-on-background alone is too weak to read as a card; Qt got the same
look from `radius` + `border.color`.

**The blocker that settled it:** *uv-managed CPython cannot open a Tk window.*
python-build-standalone ships Tcl/Tk but bakes in the build machine's prefix
(`/tools/deps/lib/tcl9.0`), so `import tkinter` succeeds and `tkinter.Tk()`
dies with "Cannot find a usable init.tcl". Homebrew's `python3` fails
differently (no `_tkinter`; separate formula), as do the distros that split out
`python3-tk`. Since `uv tool install` *is* the distribution channel, Tk's whole
advantage — free and already present — is false as stated. It is recoverable
(set `TCL_LIBRARY`/`TK_LIBRARY` from `sys.base_prefix` before importing
tkinter) but that is a startup hack shipped forever, with a different failure
per platform and per Python source. Qt needed no equivalent.

**PySide landmines paid for in the spike:** `qmlRegisterSingletonInstance()`
breaks the `ApplicationWindow` root outright — every QML-declared child is then
rejected from `contentData` ("Cannot assign QQmlConnections_QML_6 … expected
QObject"); `setContextProperty()` works but resolves late, so each
`Component`'s first binding pass reads `null` and logs a TypeError. **Use
`engine.setInitialProperties({...})`** — set before component completion, every
binding correct on the first pass, zero warnings. `QQuickWindow.grabWindow()`
screenshots the app with no Screen Recording permission, which makes UI
regression tests possible in CI (`screencapture` from inside the app instead
*blocks* on the TCC prompt).

### Rules the port inherits

1. The GUI stays a thin client of the library — the same rule the CLI
   follows: screens gather inputs and call library entry points; logic a
   screen needs that the library lacks goes into the library.
2. Slow work goes to a worker thread with signals — the Qt equivalent of
   `@work(thread=True)`. Blocking the GUI thread freezes rendering.
3. Notes generation still goes through the existing entry point on the thread
   that imported `mlx_lm`. A Qt worker thread naively re-importing it dies
   with "no Stream(gpu, 0)".
4. **The redraw budget carries over, but it cannot reach Qt's floor.** The TUI's
   `TEXTUAL_FPS` pin and `animation_level = "none"` become "bind the view to
   model updates, no idle animations": a QML `Timer { running: true }`, a
   spinner, or an easing transition on every caption holds the compositor awake
   for the whole meeting, against a live pipeline tuned to ~0.6 W. The app keeps
   exactly one 1 Hz timer (`gui/meeting.py`, the meeting clock).

   That discipline is necessary but **not sufficient, and step 1 measured why**:
   a *visible, unoccluded* Qt Quick window with scene-graph content is woken at
   the display's refresh rate — ~120/s on the Liquid Retina XDR panel — whatever
   the QML does. The lever that actually works is **occlusion**, and it is ~100×
   rather than ~2×: covered by another window the same app drops to ~1 wakeup/s.
   So the realistic in-meeting cost is the occluded case (the window normally
   sits behind the video-call window), and a tray-only mode idles at the floor
   permanently — a power argument for step 6 in its own right.
5. The notes screen stays a dumb file picker, never a meeting list with
   metadata — that would be the meeting browser the product philosophy forbids.

### Sequencing

Each step shipped working; step 1's watt remainder is deferred and is the
one thing still open in this phase.

1. **Half done 2026-07-25; the watt half deliberately deferred.** The
   per-process half is measured and the app is clean: on the idle home screen
   `steno --gui` is indistinguishable from a 40-line standalone replica of
   `Home.qml`'s structure with no Python object behind it — both ~120 wakeups/s,
   ~2.0 CPU ms/s, **GPU 0.00 ms/s**, Energy Impact ~0.16. The wakeups are
   therefore not our timers, not the `Behavior`/`ColorAnimation` pairs and not
   the `app` bridge; they are the display-refresh floor described in rule 4
   above, and `QSG_RENDER_LOOP=basic` does not remove them.

   Two measurement traps, both paid for: an **empty `ApplicationWindow` is not a
   valid control** — with no scene graph it never exercises the render loop, so
   comparing against it makes Qt's floor look like our own bug; and `sample`
   showing all six threads parked does **not** mean nothing wakes the process,
   because each wake is ~17 µs and invisible at 1 ms sampling. Use
   `powermetrics --samplers tasks` (per-process wakeups survive a busy machine;
   absolute watts do not).

   **Deferred:** the absolute watt delta against the pipeline's ~0.6 W. It needs
   a quiet machine — Spotlight/SeaDrive indexing had the noise floor at
   446–795 mW, 350 mW peak-to-peak, which is the order of the quantity being
   measured — and the wakeups' true cost is masked meanwhile, since "Pkg idle"
   wakeups read ~0 while `mds_stores` keeps the package out of idle anyway. Not
   a blocker for anything below; pick it up on an idle machine, on battery,
   several minutes per arm, app-open vs app-closed interleaved.
2. **Done 2026-07-25 — passed.** A locally written, ad-hoc-signed
   `Stenograf.app` was granted mic + system audio, then
   `uv tool install --reinstall stenograf` replaced the venv, the python symlink
   and stenocap itself (helper cdhash `d431bc81…` → `f8246bfc…`, a harsher
   change than a version upgrade makes); the app relaunched with a mic frame, no
   prompt, no new TCC row and an untouched grant. Evidence read straight out of
   `~/Library/Application Support/com.apple.TCC/TCC.db`. Two findings step 5
   must honour, both learned the hard way:

   - **The main executable may not be a script, and may not `exec`.** A
     `#!/bin/sh` wrapper that execs python makes the process launchd started
     *become* the interpreter — a Mach-O outside the bundle — so TCC drops the
     bundle identity and path-keys the grant to the shared uv interpreter
     (`client_type=1`, and the prompt is titled "python3.13"). That is worse
     than the status quo: every uv-managed tool shares that interpreter, and a
     `uv python` upgrade moves the path and re-prompts. A compiled wrapper that
     **spawns** `steno` as a child and stays alive as its parent gets
     `client=dev.stenograf.app, client_type=0` for both
     `kTCCServiceMicrophone` and `kTCCServiceAudioCapture` — children inherit
     the responsible process, so stenocap's request is attributed to the app.
     stenocap's own signature is never a TCC client, which is why rebuilding it
     has never re-prompted.
   - **The stored requirement is `cdhash H"…"` alone** (decode with
     `csreq -r <blob> -t`). The grant survives upgrades precisely because the
     pinned identity lives entirely inside the bundle while the venv is outside
     it — and by the same token *any* change to the bundle silently revokes it.
3. **Done** — `stenograf/gui/` + shell + home, opt-in behind `steno --gui`.
4. **Done** — meeting screen (live captions on a worker thread), transcribe,
   notes, settings, doctor. Native file dialogs replace the TUI's tree pickers;
   the notes picker stays a folder picker, never a meeting list.
5. **Done 2026-07-25.** `steno setup` installs `~/Applications/Stenograf.app`
   (with the `gui` extra; without it the Desktop `.command` stays, since an
   `.app` cannot host a TUI). Sources in `native/appbundle/`, the built and
   signed bundle committed at `src/stenograf/assets/Stenograf.app` and shipped
   in the wheel — `shortcut.py` only ever copies it, and
   `tests/test_shortcut.py` pins a sha256 over the tree so a casual rebuild
   fails CI. Built as step 2 demanded: a universal Mach-O stub that
   `posix_spawn`s `steno --gui` and waits on it, an icon (`icon.svg` →
   `.icns` + the `icon.png` Qt now uses for its window/Dock/taskbar tile), and
   the launch target outside the bundle at `~/Library/Application
   Support/stenograf/launch-target` — one argv element per line, falling back
   to `~/.local/bin/steno --gui`, which is why the CLI must keep accepting
   `--gui` forever. That path is `data_dir()`'s macOS default rather than the
   `~/.config` this plan first named, but with `$STENOGRAF_DATA` deliberately
   *not* honoured: launchd gives the stub no environment, so both ends have to
   agree on one fixed path.

   Everything the stub does beyond spawning exists because a Dock launch has
   nowhere to complain to: the child's output is teed to
   `~/Library/Logs/Stenograf.log`, a missing target or a non-zero exit raises a
   `CFUserNotification` alert, signals are forwarded so logout doesn't orphan
   the window, and `PATH` is widened past launchd's bare four entries. The
   signal handler restores the default disposition once the child is gone —
   without that the stub survives SIGTERM while an alert is up, which cost a
   rebuild to find. Verified end to end on this machine: the wheel round-trips
   the bundle byte-identically with its signature intact, the copy in
   `~/Applications` still satisfies `codesign --verify --strict`, and capture
   started from the app wrote the grant this whole design is for —
   `kTCCServiceMicrophone` and `kTCCServiceAudioCapture` against
   `dev.stenograf.app`, `client_type=0`, requirement `cdhash H"1651c78f…" or
   cdhash H"b0516786…"`, i.e. **both slices of the universal binary**, so the
   grant even survives an Intel↔Apple-Silicon migration. No `python3.13` row
   appeared. The step-2 test bundle's grant was replaced rather than migrated,
   which is the freeze rule demonstrated from the other side.
6. **Done 2026-07-25.** `stenograf/gui/tray.py`: a `QSystemTrayIcon` carrying
   the app icon's two commas without their tile (`assets/tray.svg`, re-inked at
   runtime — a macOS template image when idle, red while recording, amber while
   finishing), a menu that can open the window, start, stop, open the meetings
   folder and quit, and `steno --gui --tray` to start with no window at all.
   Where `isSystemTrayAvailable()` is false — stock GNOME without the
   AppIndicator extension — `install()` returns `None` and nothing below
   happens: the window is the app and closing it quits, exactly as before.

   Two behaviour changes ride along, both worth more than the icon. **Closing
   the window hides it and a running meeting carries on**, which is what makes
   rule 4's occlusion lever reachable — and **quitting mid-meeting now waits for
   the finalize** instead of relying on the after-the-loop `join_meetings`,
   because with no window in front of it Quit becomes the normal way a meeting
   ends rather than an accident. The menu bar says so while it waits, and a
   second Quit abandons it (the checkpoint survives; a wait with no way out is
   the worse bug).

   The feared plist coupling never existed (measured in step 5,
   `native/appbundle/README.md`): LaunchServices moves the bundle's registration
   onto the spawned Qt child, and the child flips its own activation policy —
   `setActivationPolicy(accessory)` via `objc_msgSend`, since AppKit is already
   loaded and one selector does not justify pyobjc. Measured through the real
   bundle on 2026-07-25, and the two findings that shaped the code:

   - **A second `open` on an accessory app starts no second instance** —
     LaunchServices still tracks it — but AppKit's default reopen handling has
     no window to order front, so the gesture does nothing. It arrives as
     `QEvent::ApplicationActivate`, which is now what restores the window.
   - **The launch activation looks identical**, and obeying it would put a
     window on screen in `--tray` mode. The first one is therefore ignored,
     which is also why `LSUIElement` stays rejected: it additionally stops
     `open` from bringing the window forward at all.

   Landmine paid for: **`QSystemTrayIcon` needs a `QApplication`.** It is a
   QtWidgets class and its menu is a real QWidget; under the `QGuiApplication` a
   pure Qt Quick app would use, the process dies on `qFatal` before any Python
   exception can be raised. `run()` and the test fixture now build a
   `QApplication`.

   **Not measured:** the wakeup claim itself. Step 1 showed a visible window at
   the ~120/s display floor and an occluded one at ~1/s, and a hidden window
   should be at least as good, but the tray-mode number was not taken — it
   belongs with step 1's deferred watt half, on the same quiet machine.
7. **Done 2026-07-30 — shipped in four commits, each CI-green.** PySide6
   into `dependencies` (`gui = []` stays an empty extra so old install
   commands resolve), the double-gated dispatch (`_interactive_terminal()` +
   the new `_display_available()` seam), `src/stenograf/ui/` deleted,
   `steno setup`'s terminal-launcher arms collapsed with the retirement calls
   kept, docs and the `verify` skill rewritten. The decision record is
   `git log --follow -p PLAN-GUI-DEFAULT.md`; the observation gates were run
   the same day and are green (top of this section). `--gui` keeps
   working forever (`cli/__init__.py`): the frozen stub compiles it in as the
   fallback argv, and that stub cannot change without revoking every macOS
   user's microphone grant.

---

## Notes and presets — both built 2026-07-30, the presets UI 2026-07-31; what is open is gates

`PLAN-NOTES-MARKDOWN.md` and `PLAN-MEETING-PRESETS.md` were deleted with this
section; their design records, adversarial-review findings and rejected
alternatives are in `git log --follow -p <file>`. Read the notes plan before
touching `notes/` — in particular the four jobs the deleted JSON schema was
doing (structure, sanitizing, truncation detection, refusal detection) and
which replacement covers each, because three of the four are non-obvious.

**Notes are markdown now: the template is the schema** (`d5963dd`). Headings
are matched verbatim against the template actually used, zero matches is a hard
fail, missing or empty sections are warnings that travel on
`NotesProvenance.warnings`, and truncation is read off the backend's own
completion signal (`finish_reason` / `done_reason`) rather than a text
heuristic. `.notes.json` is no longer written. **Three gates remain, and two of
them need a machine this desk does not have:**

- **Gate 0 — the Ollama probe. Not run; needs the CachyOS notebook.** One
  `/api/chat` call to qwen3:8b, no `format`, oversized prompt, answering three
  things at once: whether `prompt_eval_count` shows the server silently
  truncating the head (`ollama.py` sends no `options.num_ctx`, so its char
  ceiling may be a client-side fiction — if so, derive `num_ctx` from
  `max_input_chars`), whether `done_reason` is present (the truncation check
  above depends on it), and whether reasoning arrives in `message.thinking` or
  leaks into `message.content` (the latter needs the shared
  `strip_reasoning()`). Do not write the conclusion up as "`format=` was doing
  this": the repo reads neither field today, so the mechanism is unknown.
- **Gate A — equivalence on real meetings. Captured; Daniel's read pending.**
  Before/after pairs on the same transcripts live in `gate-a/` (gitignored,
  real meeting content). `small-claude` and `mid-claude` succeeded on both
  paths, all headings matched, zero warnings. **mlx has no before-note to
  read** — the JSON path failed 2 of 2 attempts on the small meeting ("missing
  a usable 'title'", from the pinned `4fe5c76` worktree; mechanism not
  investigated, the path is deleted) where the markdown path succeeded first
  try. n=2 against n=1, so the direction is the claim, not the size: the
  shipped macOS default could not produce notes for that real meeting on the
  old path and can on the new one.
- **Gate B — template adherence on Ollama. Not run; needs CachyOS or
  Windows.** Run twice on the same box — first with `format=` still in place
  (check out `4fe5c76`, the last commit that has it), then without — because a
  single unconstrained run cannot attribute a failure between "removing
  `format=` broke it" and "this path never worked" (a real Ollama notes e2e has
  never been run at all; it is in the declined list below). **It does not block
  the macOS default; it blocks declaring the Ollama path healthy** — which is
  worth remembering before a release, since Ollama is the notes default on both
  non-macOS platforms.

Accepted regressions, decided rather than discovered later, and all of them
release-notes material: `.notes.json` is gone; **owner-grouped action items are
gone** (the vault note shows action items as the model wrote them); the vault
note gains Highlights and inline `[h:mm:ss]` timestamps it used to strip; and a
decision-free meeting now gains an empty `## Decisions` section, since a
template makes the model emit every heading — the per-section emptiness warning
is that one's trace.

**Meeting presets shipped too** (`b00ef0b`, after `4fe5c76`'s per-run
baseline): `[meetings.<name>]` sections, the sparse overlay with the `""`
off-marker, `--preset` on `start`/`transcribe`/`notes`, per-preset `template` +
`instructions` + `[vocab]` (which reaches TurboBias, so a transcript recorded
under a preset is not reproducible without it), `steno presets`, `cwd` +
`STENOGRAF_MEETING_DIR`/`STENOGRAF_OUTPUT_HOME` on the command backend, and
per-preset doctor checks that resolve each `argv[0]` under the effective PATH.
Three things stayed open:

- ~~**The UI half.**~~ **Built 2026-07-31, ahead of the evidence question
  below** — Daniel asked for the picker directly. "Meeting type" is the first
  control in the Qt setup form (hidden entirely when settings.toml defines no
  `[meetings.*]`, so a machine without presets sees the form it always had),
  each entry carrying the one-line summary `steno presets` prints —
  `MeetingPreset.summary()` is now the single source both render. The Settings
  screen got the same picker: it re-renders the read-only report under
  `settings_report(preset=…)`, which is `steno settings show --preset NAME`'s
  library seam. Both reports **attribute** rows now — a key the preset set reads
  `([meetings.NAME])`, one it switched off with `""` reads
  `([meetings.NAME] switched it off)`, and an env var that beats a preset key
  says so — because the overlay is sparse, so reading the section tells you what
  it sets but not what wins. Vocabulary gets a sentence rather than a row: it
  merges, so there is no single effective value to print.
- **The evidence question is now moot for the picker but still open for the
  layer.** The per-run flags (`--notes-backend`, `--notes-model`,
  `--instructions`) were the baseline the preset layer had to beat on real use;
  persistence, per-kind `[vocab]` and a GUI picker are the three things that
  baseline structurally cannot do. The picker exists ahead of that verdict, so
  what real use decides now is whether presets *get written at all* — if
  `[meetings.*]` sections stay empty on this machine after a few weeks, the
  layer is the thing to reconsider, not the control.
- **One known asymmetry, documented in `resolve_meeting_request`'s docstring —
  and now user-visible, which it was not when it was accepted:** a user picking
  a meeting type whose summary reads "notes via mlx" gets a different backend if
  `STENOGRAF_NOTES_BACKEND` is set. Still judged not worth an override channel
  through `MeetingRequest` (the var is a developer escape hatch, and the
  Settings screen's report names it as beating the preset), but the reason it
  was cheap has changed. The mechanism: `flow.MeetingRun` calls
  `create_backend(None, …)`, so the env var wins inside `default_backend_name`,
  while the CLI passes the preset's backend explicitly and holds
  flag > preset > env.

Two designs from those plans are **rejected and should not be re-opened**: a
directory of preset *files* (a shared checkout is arbitrary code execution —
`[notes] command` argv runs unattended after every meeting, and `git pull` must
never change which commands run, triggered by recording a meeting; if sharing
becomes real it returns as a directory loaded *without* executable keys unless
settings.toml allowlists them), and any **preset-name stamp in the
`MeetingProfile`** (profiles serialize into every transcript and must record
resolved values, not machine-local late-bound references — regenerating notes
for an old meeting takes `--preset` explicitly).

---

## Platform work still open

**Windows — the capture helper shipped 2026-08-02; one AEC-quality run is what
is left.** `native/stenocap/` (Rust, WASAPI, device-stamped) replaced the
in-process `soundcard` streams, and `FAR_END_LAG_S` went with them — the
capture-helper plan, closed and deleted 2026-08-02 with the Linux half built
(`git log --follow -p PLAN-CAPTURE-HELPER.md`). A status icon the shell files
under Stenograf rather than under `pythonw.exe` landed 2026-08-01
(`gui/wintray.py`; the design notes are in CLAUDE.md, the GUID is permanent).
The rest of this section is the history that produced both.

**The pre-helper state — closed 2026-07-26 except for one optional AEC run.**
`PLAN-WINDOWS.md` was deleted on 2026-07-27 with five of its six items green;
`git log --follow -p PLAN-WINDOWS.md` holds the evidence. Shipped: a real `.lnk`
launcher (Start Menu + Desktop) with a multi-size `.ico`, written through
`IShellLink` COM in `winlink.py`, and the
`SetCurrentProcessExplicitAppUserModelID` call that makes the taskbar recognize
its own window — together they are why a finish notification arrives as a toast
titled *Stenograf* instead of *pythonw.exe*. Measured green on the GPD notebook
(Radeon 890M, Win 11 26200): the desktop app on a real session (tray, toast,
taskbar identity, 125/150/200 % scale, close-during-meeting, and the named-pipe
single instance including the crashed-instance case no other platform
exercises), the TUI by eye in Windows Terminal, and DirectML on the AMD tier —
byte-identical to CPU, with the speed win landing at 1.3–1.5× on an iGPU against
6.6× on a discrete card.

**W4 found a real bug and fixed it.** With the speakers at 90 % instead of 40 %
(the volume knob, not the driver's audio enhancements, is what establishes an
echo path) 80 seconds was enough: 2.6 dB ERLE and two lines of far-end speech
attributed to the local speaker, against macOS's 37.6 dB. The cause was not the
canceller but the timestamps it pairs channels by — WASAPI's loopback tap is the
longer transport and both channels are stamped on arrival, so AEC3 was handed a
reference labelled ~60 ms *after* its own echo, and its delay estimator only
searches backwards. `CaptureProvider.far_end_lag_s` now carries the per-provider
correction (`capture.windows.FAR_END_LAG_S = 0.15`, 2.5× measured, because the
error is one-sided): **13.7 dB ERLE, 0 leaked lines** on a re-run of the same
script.

**W4's remainder is now the capture helper's acceptance test, and it is the one
Windows thing still open.** Its alignment half was superseded and then settled:
the helper shipped 2026-08-02 and deleted the constant rather than tuning it.
But **nothing has yet scored the helper against what it replaced** — 13.7 dB
ERLE with zero leaked lines, `eval/README.md` — because this desk is on
headphones, so there is no echo path and `eval/aec_alignment.py` correctly
refuses to grade one. So the run below stopped being optional: it is what turns
"the timeline is right" (measured, and end-to-end meetings work) into "the
canceller is at least as good as before". It also still carries the two
questions device timestamps never touched: double-talk, which nothing on this
machine has covered, and the residual gap to macOS (13.7 dB against 37.6 dB —
chassis, driver processing, or headroom). It is one command —
`uv run python eval/aec_rig.py far-only --seconds 1800 --volume 90`, **plain
`uv run`, never `--group eval`** (that group cannot resolve on Windows: `mlx`
has no `win_amd64` wheel) — plus `eval/aec_alignment.py` on the dump. It is half
an hour of speech out loud, so it wants an empty room, and the rig pins an
en-US SAPI voice because the system default is German.

**DirectML fail-loudly (2026-08-02) — verify on the real Windows box before the
next release.** An explicit `[asr] provider = "dml"` that can't deliver now
raises `BackendUnavailableError` with the canary's reason instead of silently
running on CPU; the fallback branch and the `provider_fallback`/`active_provider`
plumbing are deleted. To confirm on the GPD notebook: a healthy `dml` run still
loads and transcribes, and a forced failure (`STENOGRAF_ASR_PROVIDER=cuda`
against the DirectML build) exits with the clean error naming
`[asr] provider = "cpu"`.

**`.lnk`-only launcher (2026-08-02) — verify on the real Windows box before the
next release.** `steno setup` now writes only the two `Stenograf.lnk` shell
links (Start Menu + Desktop); the batch-file fallback is deleted, and a COM
refusal is a clear setup error instead. To confirm on the GPD notebook:
`steno setup` still lands both links and they launch the app.

**Linux — the same root cause, closed by construction 2026-08-02.** The
never-measured Linux echo-cancellation risk (`parec`, one subprocess per
channel, both arrival-stamped by `SessionClock`) ended the way the
capture-helper plan prescribed: the Rust helper grew a PulseAudio-protocol
backend (`native/stenocap/src/pulse.rs`) that stamps both taps
`CLOCK_MONOTONIC − server latency` on one clock, `parec` and `SessionClock`
left with it, and **no arrival-stamped transport exists on any platform now**
(`far_end_lag_s` is deleted outright, not zeroed — `EchoCanceller` keeps the
parameter only as `eval/aec_alignment.py`'s verification knob). Measured on
PipeWire 1.6 the day it shipped: stamp error against a sample-count line holds
a bounded ±10 ms band over 35 s, a played tone lands at its wall-clock instant,
and a real live meeting through the helper transcribed word-perfectly. Real
PulseAudio is exercised by ci.yml's `capture-linux` job. The wheel matrix grew
the low-floor `manylinux_2_28` capture wheel so Ubuntu 22.04 / Debian 12 /
RHEL 9 keep live capture instead of silently falling to the pure wheel. What
was *not* run on Linux is an ERLE-scored AEC session (this machine's desk has
no speaker echo path either); the alignment premise it would confirm is the
one now guaranteed by construction, so it rides along as a nice-to-have with
the Windows gate above, not as its own gate.

**Capture fold stage (b) — macOS-gated (queued 2026-08-02).** Stage (a) folded
`capture/macos.py` and `capture/linux.py` into `capture/helper.py`; the one
platform difference left is the stop gesture (`_stop_signal`: SIGINT for the
Swift helper, stdin-EOF for the Rust one). Stage (b): teach the Swift helper to
exit on stdin EOF (mirroring the Rust helper's stdin watcher in
`native/stenocap/src/main.rs`), then delete `_stop_signal` and
`_request_exit`'s signal arm — one stop gesture everywhere. Legal (the helper
binary is not under the frozen app bundle), but it recompiles and re-signs a
signed binary, so it waits for a real macOS session and a live-capture check
afterwards.

**The desktop app on Linux — measured, fixed and closed 2026-07-25.** The app
ran on a real session (KDE Plasma 6.7.3, Wayland, 150 % scale) including a live
meeting captured through PipeWire from the GUI; seven problems were found and
fixed the same day — the single-instance claim (now a `QLocalServer` in
`gui/app.py`, unconditional on **every** platform), the localised documents dir
(resolved from `user-dirs.dirs`, no legacy branch), the desktop entry, the
doubled window title, the X11 `WM_CLASS`, the window default, and *Open
Stenograf* greyed for a merely **buried** window (`isVisible()` means mapped,
not looked at — it now keys off focus, as does the "Meeting finished"
notification the same predicate had been suppressing). `PLAN-LINUX.md` carried
the evidence, the decisions and the container ladder for the desktops it has
never run on, and was deleted on 2026-07-26 with nothing open:
`git log --follow -p PLAN-LINUX.md`.

**stenodiar off macOS — done 2026-07-26, nothing open.** The
release-attachment plan was dropped for platform wheels; `release.yml` cut all
four artifacts and passed every smoke leg on a `workflow_dispatch` dry run,
which is everything a tag does except publish. Why the floor is
manylinux_2_39 and not lower — it is upstream onnxruntime's, not ours — is in
`native/README.md` with the rest of the design.

---

## Deferred and declined

Kept here so future sessions don't re-open them.

- **Challenger ASR models — declined 2026-07-27.** The recurring "the
  leaderboard has a new leader" question (most recently Cohere Transcribe
  03-2026) is closed, and `PLAN-ASR-CHALLENGER.md` was deleted with it
  (`git log --follow -p PLAN-ASR-CHALLENGER.md` for the candidate analysis and
  the adjudication design). Two reasons, and neither is about that model.
  **The contract:** `asr/base.py:13` makes word-level timestamps mandatory —
  speaker assignment intersects them with diarization turns — so a
  timestamp-free model can only ever be a finalize-only second pass with a merge
  layer, on top of Parakeet, at roughly double the finalize cost. **The prior:**
  every model that has beaten Parakeet on a read-speech benchmark has lost to it
  on our meeting audio. Voxtral Small 24B led by a *wider* paper margin than
  Cohere's and lost 32:38 head-to-head; Canary-1B-v2 lost every pairing ~1:2
  (Phase 0, 161 blind adjudication sites, de+en). Parakeet holds the default on
  being ~10× faster and 5× smaller at Whisper-large-v3-equal quality.
  **What would reopen it:** a leaderboard leader that already emits word
  timestamps — that one is not a challenger at all, it is a `BackendSpec` in
  `asr/registry.py`, which is what the seam exists for. A model without them
  does not get evaluated again.
- **Hand-labelled RTTM references — declined.** Without them DER, word
  attribution, the 0.5 re-ID threshold and far-field speaker-count
  over-splitting (a small group measured as 8) stay unmeasured. The scorer is
  built and tested; only labels are missing, and labelling was ruled out.
  The AMI/ICSI harness (`eval/ami.py`, 2026-08-02, from the deleted
  diarization plan's step 0) sidesteps the ruling-out rather
  than reversing it: AMI/ICSI are CC-BY-4.0 *with per-speaker headset
  channels*, so references in our exact two-channel topology are built from
  public corpora — labelling our own audio stays ruled out.
- **The diarization + speaker re-ID program — closed 2026-08-09, every
  candidate measured.** `PLAN-DIARIZATION.md` deleted
  (`git log --follow -p PLAN-DIARIZATION.md` for the step record); evidence
  is in `eval/README.md`'s corpus-harness sections, the research record in
  `eval/diarization-sota-2026.md`. What ships: the owned ward+gated-fold
  loop, overlap-clean embeddings, the solo-channel bypass with
  `Diarizer.channel_embedding` for 1:1 profiles, profile store v2 +
  rename-once enrollment, threshold 0.56. The heavy clustering fallback
  stays scoped, deferred not closed: two-covariance PLDA via WeSpeaker +
  stock VBx (`eval/diarization-loop-spec.md` §3, ranked option A, with
  starting hyperparameters); trigger = a Phase-B-style clustering residual
  the fold-gate audit can't explain.
  Declined with numbers, and what would reopen each:
  - **DiariZen meeting-base (2026-08-09):** loses the fair k+1+production-fold
    frame on loops (+1.3 mean / +1.2 median paired ΔDER, wins 4/20, sign
    p≈0.012, attribution −1.1) and the near-field duos (+2.0/+1.3, 3/20);
    its own hyperparameters swept and exonerated. The ONNX-export program
    never opens. Trigger: the meeting mix shifts toward many-speaker rooms —
    it won all four ICSI k=4–8 loops while their k=2 duos flipped back.
  - **Embedding upgrade (2026-08-07):** ERes2NetV2-zh and ResNet34-LM both
    regress clustering (6.6/7.6 pts attribution) and *collapse* the
    short-turn axis (FAR0 thresholds 0.82/0.88 vs 0.605); English ERes2NetV2
    was never released. Watch: ReDimNet2 with a CC-BY vox2 checkpoint + working ONNX;
    an export attempt of `eres2net_large_sv_en_voxceleb_16k` only if a step
    is actually limited by full-duration embedding accuracy — today none is.
  - **Stride >1 s (2026-08-07):** 2.1×/3.1× speedups cost naming purity (the
    FAR0 threshold jumps 0.592→~0.78, silently invalidating the calibration).
    Trigger: finalize latency becomes binding — then per-chunk embedding
    (mask at stat-pooling) first, and any stride change re-runs the
    threshold calibration in its gate.
  - **Automatic profile updates (2026-08-07, reaffirmed post-ward):**
    ungateable wrong updates above every implementable bar while the oracle
    ties no-update everywhere; growth stays user-confirmed. Re-run
    `eval/auto_update.py` as part of any future diarization change's gate
    (likewise `eval/threshold_pick.py`).
  - **Sortformer** (broken ONNX+CPU path, 4-speaker cap, can't consume a
    known count — no trigger); **joint/LLM SA-ASR** (trigger: a permissive,
    CPU-practical joint model beating the modular cascade on meeting
    benchmarks with real non-oracle diarization); **TS-VAD** (trigger: an
    Apache/MIT ONNX implementation, or Personal-VAD-class open weights);
    **AS-Norm** (degraded the strongest open-set models; trigger: an
    unsatisfying threshold curve, and QMF-style duration-aware calibration
    goes first); **EEND-TA** (watch — no code or weights; trigger: a
    release); **pyannote.audio as a dependency** (opt-out telemetry, ORT
    dropped, paid-SDK pull — weights keep getting extracted, the package
    never imports here).
- **Repair-only overhang — the one open accuracy idea.** Default decode stays
  the exact current slice; overhang fires only on a detected ≥1.5 s speech
  hole. It is the salvage from the reverted cut-overlap work, structurally
  immune to the language-flip sites that killed the always-on version. The
  **VAD threshold 0.4 re-attempt is blocked on it** (its known cost is more
  forced mid-speech cuts, unrepaired since the revert); a ~0.5 s always-on left
  slice overhang stands as a separate candidate. Any of these re-runs the full
  battery against a control worktree.
- **Never move VAD window bounds.** Both bound-moving fixes were tried and
  reverted; only the decode slice may change.
- **livekit — re-ask deferred from the capture-helper plan, and its trigger
  has now fired.** 25.6 MB installed for one class (`rtc.AudioProcessingModule`,
  AEC3, imported in exactly one place — `aec.py`). It was never a
  remove-the-wrapper candidate on its own: it discards nothing and costs no
  correctness, and replacing it risks a canceller that measures 37.6 dB on
  macOS. The trigger was the helper holding both channels with device
  timestamps on every platform — true since 2026-08-02. Running AEC3 *inside*
  the helper (Rust bindings to the same WebRTC APM) would mean channel pairing
  never crosses a language boundary: `aec.py`'s timestamp machinery would stop
  existing rather than get a better input. Revisit on all three platforms at
  once, against the shipped ERLE numbers; the full reasoning is in the deleted
  plan (`git log --follow -p PLAN-CAPTURE-HELPER.md`).
- **A cancel button for an in-flight notes call.** Quitting the app no longer
  waits one out (`5a6b902`: `shutdown()` abandons it, the transcript is already
  persisted at `finalized`, and `steno notes --last` regenerates), but actually
  *interrupting* a blocking `subprocess.run` / mlx generate needs the Popen
  restructure priced in the deleted presets plan. Deferred, not declined.
- **On-device notes backends.** Windows and Linux CPU stay Ollama-default; the
  in-process fallback, if ever needed, is llama-cpp-python with off-PyPI
  wheels. onnxruntime-genai-directml was evaluated and rejected (DX12-GPU-only
  on the EOL DirectML EP).
- **Packaged signed installers / any downloadable artifact.** Needs the $99
  Developer ID plus notarization; revisit only if non-terminal users ask to
  double-click an installer.
- **Lower-priority, independent:** greedy re-ID → Hungarian assignment;
  SRT/VTT dropping text not covered by `words` (latent — Parakeet emits
  full-or-none); meeting-mode auto-detect; hybrid cross-channel dedup;
  acoustic first-segment LID for the live pass; a real Ollama notes e2e (needs
  a machine with Ollama installed — now also the reason Gates 0 and B above are
  unrun); re-running the winning biasing config over the full test set.
