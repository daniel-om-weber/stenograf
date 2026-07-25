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

## Phase 8 — native GUI (Qt Quick). Decided 2026-07-25; app built, steps 1–2 and 5–7 open.

**Decision: the launcher becomes a real desktop application built on Qt Quick
(PySide6), and `Stenograf.app` is generated locally by `steno setup`.** This
absorbs the old Phase 7 Tier 2 (tray + packaged installers). The CLI
subcommands and the whole library stay untouched; the Textual launcher stays
the default until the Qt screens reach parity.

**Built 2026-07-25 (steps 3 + 4): `stenograf/gui/` — all six screens, opt-in
behind `steno --gui`,** with PySide6 as the optional `[gui]` extra. Getting
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
locally is never assessed by Gatekeeper. `steno setup` already exploits this
for `Stenograf.command`; the same trick carries a full `.app` bundle — custom
icon, Dock presence, `LSUIElement` for menu-bar mode — with no Developer ID,
no notarization, not even a real signature (ad-hoc suffices). **The $99/yr
gates only a *downloadable* artifact.**

Consequence to design for: TCC grants move off Terminal.app onto our bundle.
That is better hygiene ("Stenograf would like to access the microphone") but
reworks the `setup`/`doctor` permission flow, and an unsigned bundle's TCC
identity is weak — so the bundle must stay a **thin wrapper that never
changes** (it execs the installed `steno`), and the grant's survival across
`uv tool upgrade` must be tested before anything is built on it. Windows and
Linux need no equivalent trick: a locally written `.lnk` carries no
Mark-of-the-Web, and Linux signs nothing.

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
4. **The redraw budget carries over verbatim.** The TUI's `TEXTUAL_FPS` pin and
   `animation_level = "none"` become "bind the view to model updates, no idle
   animations": a QML `Timer { running: true }`, a spinner, or an easing
   transition on every caption holds the compositor awake for the whole
   meeting, against a live pipeline tuned to ~0.6 W. The prototype keeps
   exactly one 1 Hz timer and ships `pulseEnabled: false` to make the cost of
   a pulsing REC dot visible.
5. The notes screen stays a dumb file picker, never a meeting list with
   metadata — that would be the meeting browser the product philosophy forbids.

### Sequencing

Each step ships working; none blocks the platform work below.

1. **Open.** Idle-power measurement of the built app under `powermetrics` — the
   number that must not regress the live pipeline's power profile. The app ships
   the prototype's discipline (one 1 Hz timer, no page transitions, no idle
   animation; only hover/toggle feedback animates, event-driven and ≤90 ms), but
   the discipline is asserted by review, not measured.
2. **Open.** The TCC-survival test: locally written `.app`, grant the mic,
   `uv tool upgrade`, confirm no re-prompt. Steps 5–6 assume it passes.
3. **Done** — `stenograf/gui/` + shell + home, opt-in behind `steno --gui`.
4. **Done** — meeting screen (live captions on a worker thread), transcribe,
   notes, settings, doctor. Native file dialogs replace the TUI's tree pickers;
   the notes picker stays a folder picker, never a meeting list.
5. `shortcut.py` writes `Stenograf.app` instead of `Stenograf.command`; icon
   assets (`.icns`/`.ico`/`.png`) land here — none exist yet.
6. Menu-bar / taskbar mode via `QSystemTrayIcon`, degrading to the launcher
   where no tray host exists.
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
