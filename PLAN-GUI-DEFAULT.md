# stenograf — the GUI becomes the default, the Textual UI retires

Decided 2026-07-30 (Daniel): the default way of using stenograf is the desktop
app / tray icon, so Phase 8 step 7's use-gate is declared met — and the
Textual front-end is **retired entirely**, not kept as a no-display fallback.
This is the explicit answer to PLAN.md step 7's "pick one" and closes the
two-front-end era: one real UI (Qt), one thin line-oriented CLI. **Not
built.** Adversarially reviewed 2026-07-30 (Opus, code-grounded — it *ran*
the Qt failure case); this is the post-review draft, findings baked in.

---

## Why retirement is clean (measured against the code)

- Textual is imported by exactly one package: `src/stenograf/ui/` (**12**
  modules — the launcher's screens plus the full-screen meeting view).
  Verified by grep over `src/`, `tests/`, `scripts/`, `eval/`, `install.sh`,
  `install.ps1`: outside `ui/` every `textual` hit is a docstring mention,
  never an import, and nothing imports `rich` anywhere. `captions.py` (the
  shared line rules) is imported by `ui/meeting.py` and `gui/meeting.py` —
  it stays.
- The CLI already has a complete non-Textual path for everything:
  `PlainLiveView` (`view.py:113`) streams committed captions line-by-line and
  is *already* what runs for `--plain` and for a non-TTY stdout.
- **Ctrl-C in plain mode is a clean stop, not a death** (review-verified):
  `session.py:1013-1017` catches KeyboardInterrupt, stops the provider and
  joins capture; `_shield_interrupt` (`:359-377`) makes the finalize immune
  to a second Ctrl-C; notes still run via `_finish_run`. `--max-seconds` is
  the non-interactive stop.

What is genuinely lost, stated: the terminal live view's cosmetics (interim
tail, REC/elapsed header, done screen), the launcher menu, and — see step 1 —
installability on old-glibc Linux arm64.

## Step 1 — the dependency flip

- `PySide6-Essentials>=6.8` moves from the `gui` extra (`pyproject.toml:109`)
  into `dependencies`; `textual>=8.2.8` (`:87`) is deleted; the dev group's
  duplicate PySide6 pin (`:134`) collapses. `gui = []` stays as an empty
  extra (precedent: `ollama = []`, `:99`) so existing
  `uv tool install 'stenograf[gui]'` invocations keep resolving.
  **`uv.lock` is regenerated in the same commit** (it currently pins textual
  and marks pyside6 `extra == 'gui'`).
- **The install matrix narrows, precisely** (measured from `uv.lock`:
  PySide6-Essentials 6.11.1 wheels are `macosx_13_0_universal2`,
  `manylinux_2_34_x86_64`, `manylinux_2_39_aarch64`, `win_amd64`,
  `win_arm64`): **Linux arm64 needs glibc ≥ 2.39 (Ubuntu 24.04)** — Debian
  12 / Ubuntu 22.04 / Raspberry Pi OS arm64 can no longer install stenograf
  at all, though live capture *does* run there today. Accepted with eyes
  open: no machine in this project's fleet is old-glibc arm64, none of those
  configurations were ever tested or claimed, and `release.yml:44-47`'s
  `build-pure` comment ("musl, Linux arm64") must be rewritten to say the
  pure wheel no longer buys those platforms an install. ~110 MB of Qt joins
  every install, including terminal-only use — PLAN.md pre-decided the
  mechanism, this plan spends it.

## Step 2 — the dispatch

`cli/__init__.py:76-85` today: `--gui` → app, TTY → Textual launcher,
else → help. Becomes:

- `--gui` → app, **unchanged forever** (the frozen `Stenograf.app` stub
  compiles it in; `cli/__init__.py:34-37`'s comment stays; `shortcut.py`'s
  `launch-target` bakes it in too).
- bare `steno` → the app only when **both** hold: `_interactive_terminal()`
  (the existing seam — stdin *and* stdout are TTYs) **and**
  `_display_available()` (new seam: on Linux, `DISPLAY` or `WAYLAND_DISPLAY`
  set; on macOS/Windows, no `SSH_CONNECTION`/`SSH_TTY`). Everything else →
  help. Keeping the TTY gate (the review's correction — the first draft
  dropped it silently) removes the cron / launchd / pipe / `make` / CI
  misfire class wholesale: a script invoking bare `steno` gets help text,
  exactly as today. The icon path never touches either gate (the stub passes
  `--gui`).
- **The honest failure story (review-measured):** a Qt platform-plugin init
  failure is a C++ `qFatal` → `abort()` — exit 134, Qt's own two-line
  stderr, and **no Python `except` can catch it**; the review demonstrated
  this by running it. So there is no "degrade to a clear error" for a
  misjudged display: the double gate shrinks the misfire class to
  "interactive terminal, display env present, display actually broken"
  (e.g. a stale `DISPLAY` in tmux), and that residue aborts with Qt's
  message. Recoverable — `steno start` still works — and not worth a
  pre-flight probe that would cost a subprocess PySide6 import on every
  launch. The *missing-PySide6* case stays a catchable ClickException
  (`gui/__init__.py:29-36`), whose message must be rewritten: it currently
  recommends `stenograf[gui]`, which becomes an empty extra
  (`tests/test_cli.py:1704-1717` pins the old text).
- **`run_gui` gets a `try/finally` around `app.exec()`** so
  `join_meetings()` runs on every exit (`gui/app.py:530-532` has none): after
  the flip, "typed `steno`, then Ctrl-C" becomes an ordinary terminal
  gesture, and today an exception out of `exec()` would skip the one thing
  that stops capture and awaits the finalize.
- **Already-running instance:** the single-instance claim currently returns 0
  *silently* after asking the running app to show its window
  (`gui/app.py:500-502`). Bare `steno` in a terminal must say so in one line
  ("stenograf is already running — brought its window to front") — a
  terminal user staring at silence is a bug, and the show-the-window behavior
  is a genuine upside of the flip worth surfacing.
- `--tray` semantics unchanged. Help text (`main`'s docstring, `--gui` help)
  rewritten: no launcher to mention.
- **Windows, named decision:** there is no `[project.gui-scripts]` entry and
  none is added — bare `steno` in cmd/PowerShell runs the console
  `steno.exe`, opens the window, and blocks that console for the app's
  lifetime; closing the console kills the app. Accepted and documented: this
  is normal for CLI-launched GUI apps, a `gui-scripts` entry would make
  `steno transcribe`'s output invisible, and the supported icon path (the
  `.lnk` via `pythonw`, `shortcut.py:401-409`) is untouched.

## Step 3 — delete the Textual front-end

- `src/stenograf/ui/` (all 12 modules) and `tests/test_ui.py` deleted.
- `cli/start.py`: the TUI arm dies — `use_tui` (`:299`), the routing at
  `:543`, the `notes_in_tui` closure (`:394-412`) and the Textual import;
  live runs use `PlainLiveView` unconditionally, notes run through
  `_finish_run` exactly as today's no-TTY runs do.
- **`--plain` becomes a hidden accepted no-op**, not a deleted option — six
  live callers were found: `.github/workflows/ci.yml:109` (the
  `capture-linux` job would go red on a deleted option), `eval/aec_rig.py:88`,
  and recipes in `eval/README.md:270`, `eval/aec_alignment.py:13`,
  `eval/aec_echo_present.py:14`, `README.md:186`. The repo-local callers are
  cleaned up in the same commit; the hidden no-op keeps any external script
  working and can go whenever.
- **`CaptureLog` stays** — `flow.py:251` builds it for the *Qt* meeting path;
  only `start.py`'s TUI uses die (`:300`, `:439-443`). Named behavior change:
  a plain live run's capture-crash `ClickException` loses the buffered
  `FATAL` detail (`capture/macos.py:130-133` populates `_stderr_tail` only
  under `on_log`); the FATAL line still reaches the terminal directly on
  inherited stderr, so the information survives — the *exception message*
  gets generic. Accepted.
- `tests/test_cli.py`: `test_bare_invocation_on_a_tty_opens_the_launcher`
  (`:1637-1646`) is rewritten **in commit 1** (the dispatch change breaks it
  there); `test_plain_forces_the_stream_even_on_a_tty` (`:786`) and
  `test_start_tui_generates_notes_while_the_screen_is_still_up` (`:856`)
  legitimately survive until this commit and are rewritten against the plain
  path here. The intermediate state after commit 1 is coherent: `steno
  start` on a TTY still opens the TUI until this commit removes it.
- Docstring sweep — places that name the TUI as *live context*, not history
  (delete-or-reword, carefully where the seam survives for Qt):
  `gui/__init__.py:4-8` (says Textual stays the default — now false),
  `loaders.py:13-14, 56, 238` (the announce/`CaptureLog` seam is still used
  by `flow.py` — reword, don't delete), `session.py:366`,
  `capture/macos.py:101, 136-137`, `view.py:10, 18, 54, 116`,
  `captions.py:9`, `gui/screens.py:3, 11`, `gui/meeting.py:3`,
  `pyproject.toml:85-86` (already misnames the package `stenograf.tui`).
- The declined-list TUI re-wrap bug and the `_running`/`_render` landmine
  die with the code.

## Step 4 — `steno setup` collapses its launcher branches

Larger than one branch (review-enumerated): in `shortcut.py` the docstring's
"without the extra" bullets (`:4-33`), `install_shortcut`'s split
(`:194-209`), `gui_installed` (`:223-232`), `_install_desktop_entry`'s `gui`
param and `Terminal=` split (`:235-255`), `_install_windows_launcher`'s
non-gui arm (`:284-286`), `_install_cmd_file` / `_install_command_file`, and
the `_WINDOWS_CMD` / `_LINUX_DESKTOP` terminal templates; in
`cli/doctor_cmd.py` the fallback block (`:59-88`) including the
"install the extra and re-run" offer. `--models-only` stays.

**Keep the retirement calls** (`_retire_command_file`, `_retire_cmd_file`,
`_retire_windows_links`): a pre-flip install upgrading past this change would
otherwise keep a stale second "Stenograf" launcher that opens a dead TUI.
They run on every `steno setup` as today; only the *install* arms of the
fallbacks are deleted. `tests/test_shortcut.py`: the `qt` fixture's
parametrization (`:56-68`) collapses and ~8 fallback-only tests die; the
retirement tests stay; the frozen-bundle sha256 pin is untouched.

## Step 5 — docs and plan closure

- README: bare `steno` opens the app; delete/rewrite `:105-121` (the
  launcher), `:123-145` (the "desktop app (preview)" install recipe —
  no longer an extra), `:179-190` (`--plain`), `:319-320` (the launcher's
  notes switch), `:445-453` ("Three front-ends", "skip entirely where
  PySide6 is not installed").
- CLAUDE.md: "Three front-ends, one library" becomes two (Qt + thin CLI);
  the drift rule's reason rescopes to "the Qt app and the CLI must not
  drift"; current-focus section flips to step 7 SHIPPED; the Textual-retire
  question is answered.
- PLAN.md: step 7 closed with this file's evidence; `release.yml`'s
  build-pure comment updated (step 1); step 1's watt half and the platform
  items untouched.
- **The `verify` skill needs a rewrite, not a touch-up**: its primary
  recipe is the headless *Textual* smoke (Textual's `run_test` pilot, the
  pty/pyte recipe, "the TUI waits on `q`"), which dies with the TUI. Its
  replacement net for "CLI-green does not imply the UI works" is
  `tests/test_gui.py`'s offscreen Qt harness — the skill should say so and
  keep the plain-path live recipe.
- Release notes: the flip, the retirement, `--plain` now a no-op, the
  bigger install, the arm64 glibc floor. Version bump is a release decision.

## Sequencing (every commit keeps the CI triple green)

1. Step 1 + step 2 (dependency + dispatch + `try/finally` + the rewritten
   bare-invocation test + `gui/__init__` message).
2. Step 3 (the deletion + `--plain` no-op + the six caller cleanups +
   docstring sweep).
3. Step 4 (setup collapse, retirement calls kept).
4. Step 5 (docs, skill, plan closure).

## Gates

- CI triple locally at every commit, including the `capture-linux` CI job's
  `--plain` invocation surviving commit 2.
- One real `steno start` plain-mode meeting in a terminal (the `verify`
  skill's no-hardware recipe) — plain becomes load-bearing as the *only*
  terminal live mode.
- One bare-`steno` launch from a terminal, one from the app icon, and one
  bare `steno` with the tray app already running (must print its one line),
  on this Mac.
- `steno` from a pipe/script and over SSH prints help, exit 0 — the
  strand-nobody guarantee, testable through the two seams.
- Optional (blocked on enabling Remote Login): a real SSH-into-this-Mac
  bare-`steno`, to observe the heuristic's macOS arm on the case it bets on.
