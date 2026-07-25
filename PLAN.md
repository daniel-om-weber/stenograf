# stenograf — remaining work

**This file holds only what is not built yet.** Everything shipped —
phases 0–7, the architecture and model-choice research, the AEC design, the
code-cleanup backlog, the biasing evaluation — was removed on 2026-07-25 and
lives in this file's git history (`git log --follow -p PLAN.md`, plus the
deleted `PLAN-AEC.md` and `PLAN-CLEANUP.md`). Locked product scope and
platform decisions are in `CLAUDE.md`; measured evidence for the shipped
defaults is in `eval/README.md` and in the code's own docstrings.

What ships today: `steno start` (live captions → diarized transcript → notes)
on macOS, Linux and Windows, published to PyPI as `stenograf`, driven either by
the CLI or by the Textual launcher (`steno` with no arguments).

---

## Phase 8 — native GUI (Qt Quick). Decided 2026-07-25; app built and installable, **step 6 is next**, then 7.

**Decision: the launcher becomes a real desktop application built on Qt Quick
(PySide6), and `Stenograf.app` is installed locally by `steno setup`.** This
absorbs the old Phase 7 Tier 2 (tray + packaged installers). The CLI
subcommands and the whole library stay untouched; the Textual launcher stays
the default until the Qt screens reach parity.

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

Landmines paid for while building it, beyond the spike's:

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

1. The GUI stays a thin client of the library — the same rule the CLI and the
   Textual UI follow: screens gather inputs and call library entry points;
   logic a screen needs that the library lacks goes into the library.
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

Each step ships working; none blocks the platform work below. Steps 2–5 are
done and step 1's remainder is deferred, so **the next thing to build is step
6** — which step 5 unblocked rather than constrained: the Info.plist coupling
the plan expected turned out not to exist (see step 6).

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
6. Menu-bar / taskbar mode via `QSystemTrayIcon`, degrading to the launcher
   where no tray host exists. Rule 4 gives this a power argument the toolkit
   comparison did not: a tray-only app has no visible window, so it idles at ~1
   wakeup/s instead of the display-refresh floor.

   **Still independently shippable — the feared plist coupling does not exist**
   (measured 2026-07-25, `native/appbundle/README.md`). LaunchServices moves the
   bundle's registration onto the spawned Qt child when it checks in (one Dock
   tile, our icon, `originalPid` still the stub), and the child owns its own
   activation policy at runtime: `setActivationPolicy(accessory)` flips the
   record from `Foreground` to `UIElement` live. So menu-bar mode is a call in
   the child, not an `LSUIElement` key in the frozen Info.plist — which was
   tried and rejected anyway, because a UIElement app cannot be activated by
   `open` and the window then opens *behind* the frontmost one.
7. Flip the default (`[gui]` moves into `dependencies`, bare `steno` opens the
   window); retire the Textual launcher once parity is reached. Not before the
   app has been *used* for real meetings — nothing in the automated tests can
   tell whether the live screen reads well over half an hour.

---

## Platform work still open

**Windows real-hardware validation.** Everything automatable passes on the
desktop (transcribe, `--replay`, live WASAPI capture, DML byte-identity). Three
items need the Windows notebook with real speakers: a ≥30-min
**speakers-not-headphones** meeting exercising AEC across WASAPI loopback
silence gaps (livekit's APM is installed but never exercised on Windows
hardware, and the 0.5 s re-anchor moves far-end alignment, so re-convergence
after long system silence is unverified); a by-eye TUI check in Windows
Terminal (captions, resize, clean Ctrl-C); and re-running the three automatable
items on the notebook's GPU (different DML vendor tier). This gates the
"supported" claim, and there is nothing to code until it finds something.

**stenodiar distribution off macOS.** Throughput is measured and passes (8.6× RT
on Windows, 8.2× on Linux), `find_stenodiar` handles the `.exe` suffix and a
`target/release` dev fallback, and `build.ps1`/`build.sh` exist — but the
shipping decision (GitHub Releases attachment + a documented drop location,
deliberately not wheel-bundled: 40 MB for an optional sherpa-fallback feature
does not belong in every platform wheel) has no implementation. Today a
Windows/Linux user has to build it with cargo.

---

## Deferred and declined

Kept here so future sessions don't re-open them.

- **Hand-labelled RTTM references — declined.** Without them DER, word
  attribution, the 0.5 re-ID threshold and far-field speaker-count
  over-splitting (a small group measured as 8) stay unmeasured. The scorer is
  built and tested; only labels are missing, and labelling was ruled out.
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
  a machine with Ollama installed); re-running the winning biasing config over
  the full test set.
