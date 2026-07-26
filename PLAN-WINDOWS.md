# stenograf on Windows — what is missing, and where to measure it

The Linux side-plan closed with no work open and was deleted on 2026-07-26
(`git log --follow -p PLAN-LINUX.md` has the evidence, the decisions and the
container ladder). **This one is the opposite: it holds open work**, and it
exists because PLAN.md's platform section named four Windows items without ever
saying which machine could close them or what the code gaps actually are. Read
in order: the machine, then what is already at parity (do not re-derive it),
then the two code items — neither needs hardware — then the four that do.

---

## The machine (measured 2026-07-26)

**The session that wrote this ran on the Windows notebook, not on the desktop.**
That matters: every "automatable" Windows number on record (transcribe,
`--replay`, live WASAPI capture, DirectML byte-identity, 2026-07-11) was taken
on a machine with an RTX 4080 SUPER, and PLAN.md's three open items were all
waiting on *this* one.

- GPD `G1617-02`, Windows 11 Home 10.0.26100, on battery.
- **AMD Radeon 890M** (integrated, DX12) — the different DirectML vendor tier
  the desktop cannot stand in for, and an iGPU sharing system RAM, so the open
  question here is not only "byte-identical?" but "faster than CPU at all?"
- A 3840×2160 panel driven at **150 % scale** (2560×1440 logical) — the HiDPI
  case the app has never been rendered on.
- Realtek HD Audio with **real speakers**, i.e. the AEC scenario.
- Windows Terminal (`wt.exe`) present; the microphone consent store reads
  `Allow`, so `steno setup`'s privacy-toggle check passes here.

State of *this checkout*, so a session does not discover it mid-task:

- **No `.venv`.** Start with `uv sync --extra gui` (uv 0.11.32 is installed).
  The GUI extra is not optional for anything below — the app is still opt-in.
- **No Rust** (`cargo` absent), so `hatch_build.py` skips `stenodiar.exe` and a
  checkout install has **no diarization helper**: estimated speaker counts fall
  back to sherpa's threshold clustering, which over-splits badly. Explicit
  `--speakers N` is unaffected. To exercise the real helper here, install the
  published `win_amd64` wheel (`uv tool install stenograf`) instead of running
  from the checkout, or install Rust and rebuild.
- **No Ollama**, which on Windows is the default *and only* supported notes
  backend — so `steno notes` and `--notes` cannot run on this machine until it
  is installed. `steno doctor` reports that as an optional check, correctly.

---

## Already at parity — do not re-derive

The pipeline itself is not the gap. Windows has, and CI proves on every push
and every tag:

- **Live WASAPI capture**, mic + loopback-of-default-output, through
  `soundcard` (`capture/windows.py`); the queue/pump/clock machinery is shared
  with Linux (`capture/streaming.py`).
- **ONNX Parakeet** with the DirectML flavor declared per-platform
  (`pyproject.toml:61-62`) and `dml` opt-in through `[asr] provider`
  (`asr/providers.py`). CPU is the default here as everywhere.
- **`stenodiar.exe` in the `win_amd64` wheel** — built by `release.yml`'s
  `build-windows`, asserted present by `smoke-windows`, which also installs the
  wheel the way a user does and runs the pipeline end to end from a file.
- **`steno setup`**: the mic privacy toggle is read up front and named
  (`cli/doctor_cmd.py:97`) — the honest Windows counterpart to macOS's TCC
  prompts, since no prompt exists here — then the launcher, then the models.
- **`install.ps1`**, the one-command install, mirroring `install.sh`.
- **The Qt app compiles and drives headlessly** (`tests/test_gui.py`,
  `QT_QPA_PLATFORM=offscreen`) and the single-instance claim is deliberately
  unconditional (`gui/app.py:375`), so Windows is covered *in code* — as a named
  pipe, which is the one half no Linux or macOS run exercises.

---

## Open — code, and neither item needs hardware

These two are one change in practice: the shortcut has to carry the same id the
process sets, or neither buys anything. Do them together.

### W1 — the launcher is a batch file where macOS gets an app

`steno setup` writes `~/Desktop/Stenograf.cmd` (`shortcut.py:244`), and with the
`gui` extra that file is a `start "Stenograf" pythonw.exe -m stenograf --gui`
wrapper. Compare what the other two platforms get: macOS installs
`~/Applications/Stenograf.app` — real icon, Dock tile, Spotlight entry — and
Linux installs an application-menu entry plus a themed icon plus
`StartupWMClass` and `SingleMainWindow`. Windows gets the generic batch-file
icon, **no Start-menu entry at all**, nothing meaningfully pinnable, a console
flash on every launch, and a launch failure that goes nowhere (all three costs
are already written down in `_install_cmd_file`'s docstring).

`shortcut.py:22-26` declined the `.lnk` for two reasons, and both have expired:
it "needs COM (or a PowerShell detour) plus an `.ico` this project does not
ship". The art exists — `native/appbundle/icon.svg` is what the `.icns` and
`icon.png` were rendered from — and a new sibling in `src/stenograf/assets/` is
explicitly allowed (CLAUDE.md; it is the frozen `.app` *tree* that may not be
touched, not the directory).

What to build:

- A multi-size `src/stenograf/assets/icon.ico` (16/24/32/48/64/128/256) from
  that SVG, committed like the other rendered assets.
- A `.lnk` writer. **Use ctypes `IShellLink`/`IPersistFile`, not the
  `WScript.Shell` PowerShell detour** — the detour cannot set the shortcut's
  `System.AppUserModel.ID`, which needs `IPropertyStore`, and without that
  property W2 buys nothing (Windows matches a pinned shortcut to a running
  window by that id, and toast notifications from a desktop app require a
  Start-menu shortcut carrying it). VBScript stays rejected: Windows is removing
  it.
- Two targets when the `gui` extra is present: the Start Menu
  (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Stenograf.lnk`) and the
  Desktop — resolved through the existing `_windows_desktop()`, which already
  handles OneDrive's redirected Desktop. Target `pythonw.exe`, no console at
  all, so the flash goes away with the `.cmd`.
- The same convert-in-place discipline the other platforms have: installing or
  removing the extra must leave *one* launcher, and a `Stenograf.cmd` may only
  be deleted when it looks like one we wrote (`_retire_command_file` is the
  pattern).
- **The TUI variant keeps the `.cmd`.** A console app wants a console, and that
  is where a crash stays readable — the same reasoning that keeps
  `Stenograf.command` on macOS without the extra.
- A test beside the existing Windows-only legs in `tests/test_shortcut.py`
  (:373, :414): write the shortcut, read it back through `IShellLink::GetPath`
  and the property store, assert target and app id.

### W2 — the process never claims an AppUserModelID

`gui/app.py:459-470` sets `applicationName`, `setDesktopFileName` (the Linux
half of window↔launcher identity) and the window icon — and nothing at all for
Windows. Qt does not set an AUMID on its own, so the window groups under
`pythonw.exe` in the taskbar, a pinned shortcut does not match the running app,
and tray balloons are attributed to Python.

This is the same class of defect the Linux session found and fixed in Wayland
app_id / X11 `WM_CLASS`; only the API differs:

```python
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("dev.stenograf.app")
```

Called in `run()` next to `setDesktopFileName`, **before the first window
exists**. Use `dev.stenograf.app` — already the macOS bundle identifier — and
put it in `shortcut.py` beside `APPLICATION_NAME` and `DESKTOP_FILE_NAME`, for
the reason stated there: two strings that must be equal will otherwise drift.
W1's shortcut writes the same constant.

---

## Open — needs this machine, in this order

Nothing below is worth coding against blind; each one is an observation that
may or may not produce a work item.

### W3 — the desktop app has never been on a real Windows session

Every Windows run so far was `QT_QPA_PLATFORM=offscreen` in CI. Do this first:
it is now cheap (the machine is here), it is what W1/W2 have to be judged
against, and it is the item most likely to find something.

- Does the tray icon appear, and does it re-ink red/amber? Expect it to land in
  the **overflow flyout** — Windows 11 hides new notification-area icons by
  default, and the user must drag it onto the taskbar. Worth knowing before it
  is reported as "the tray does not work". `isSystemTrayAvailable()` is
  effectively always true here, so the no-tray degrade branch cannot be staged
  on Windows the way `dbus-run-session` staged it on Linux — it stays a Linux
  fact.
- Does `showMessage` on finish render as a toast, with our name and icon?
  (This is the half that may depend on W1's Start-menu shortcut.)
- Does the window match its launcher in the taskbar, and can it be pinned?
  (After W2; the point of measuring it.)
- Fonts and layout at **150 %** — the live setting on this panel — then 125 %
  and 200 % from Settings → Display → Scale. Qt reads the scale at startup, so
  restart the app after each change rather than expecting it to follow.
- **Closing the window during a live meeting**: it must hide, the meeting must
  keep recording, and *Stop & finalize* from the tray must write the transcript
  with no window on screen. This is the case the Linux session found two real
  bugs in.
- **The named-pipe single instance** — the one half no other platform exercises
  (`tests/test_gui.py:752` skips its file assertion here). A second launch must
  exit 0 silently and bring the first app's hidden window back; a killed
  instance must not block the next launch.

### W4 — AEC over real speakers, ≥30 minutes

The livekit APM ships on every platform but has never run against Windows
hardware. Two Windows-specific reasons this is not a formality: WASAPI loopback
**delivers no packets while nothing renders**, so `soundcard` synthesizes zeros
from wall-clock time, and the session clock re-anchors when a channel drifts
past `_REANCHOR_TOLERANCE_S = 0.5` (`capture/windows.py`) — which moves far-end
alignment, and re-convergence after a long system silence is unverified.

Run a real call on **speakers, not headphones**, ≥30 min, with deliberate
multi-second silences in it, and take `--aec-dump <dir>` (mic/lpb/enh triples)
so a failure is diagnosable afterwards. The failure to look for: remote voices
transcribed as a local speaker after a silence gap.

### W5 — the TUI by eye in Windows Terminal

`wt.exe` is installed. Captions rendering, a resize mid-meeting, and a clean
Ctrl-C. Cheap, and the one thing headless pilots cannot judge — note that the
win32 meeting-start `EBADF` crash (2026-07-12) shipped past a fully green CLI
suite, which is why the `verify` skill insists the launcher be driven under a
live app rather than in CLI mode.

### W6 — DirectML on an AMD tier

Re-run the three automatable items here and compare against the desktop's
2026-07-11 numbers: `[asr] provider = "dml"` (or `STENOGRAF_ASR_PROVIDER=dml`)
must transcribe **byte-identically to CPU**, and `unavailable_reason` must fire
rather than silently falling back if the EP is missing. On an iGPU sharing
system memory the speed claim is genuinely open — record whatever it is, and if
DML is not a win here, that is a settings-documentation finding, not a bug.

---

## Decided — do not re-litigate

- **Notes on Windows are Ollama or nothing.** No in-process backend unless
  Ollama proves insufficient, and then it is llama-cpp-python CPU GGUF
  (Phi-4-mini) with off-PyPI wheels. `onnxruntime-genai-directml` was evaluated
  and rejected (DX12-GPU-only on the EOL DirectML EP). macOS's zero-setup mlx-lm
  default has no Windows counterpart by decision, not by omission.
- **Exactly one onnxruntime flavor may be installed**, so DirectML is a platform
  marker, not an extra (`pyproject.toml:50-62`). Do not "add" CUDA on Windows.
- **Windows needs no launcher-identity trick.** A locally written file carries
  no Mark-of-the-Web, and there is no TCC equivalent to pin an identity to — so
  the whole frozen-bundle discipline that governs `Stenograf.app` does *not*
  apply to anything W1 writes. Regenerate freely; `steno setup` is expected to
  self-heal the embedded interpreter path.
- **`soundcard` over `pyaudiowpatch`** (spiked on real hardware 2026-07-11): one
  API for both channels, and server-side resampling via `AUTOCONVERTPCM`.
- **The diarizer runs `cpu` here** (`diarization/speakrs.py:44`); the `cuda`
  feature type-checks in CI but is not in the shipped wheel, and nothing asks
  for it yet.

---

## Observing the app on this machine

The `verify` skill covers CLI/TUI headlessly and is still right; none of it
reaches a desktop session. What transfers from the Linux recipe, and what does
not:

**Transfers unchanged.** Drive the real app from a harness rather than by hand:
construct the `QApplication` first, arm `QTimer.singleShot` chains that navigate
(`gui.open("Doctor")`) and submit the setup form with the same map QML sends,
and save `window.grabWindow()` per screen — it needs no screen-capture
permission and captures at the real scale factor. **Call `run()`, not `build()`,
whenever identity matters**: the application name, the icon, the AUMID (after
W2) and the single-instance claim are all set inside `run()`, so a `build()`
harness photographs a nameless, generic-icon app. Keep test meetings out of
`~/Documents\Meetings` with `STENOGRAF_DATA=<scratch>` plus a `settings.toml`
holding `[output] dir`. Give the GUI a meeting without a microphone by wrapping
`loaders.make_provider` (the hardware boundary) to substitute a wav — `flow.py`
has no replay flag by design; a previous meeting's `audio.wav` is the best
source.

**Does not transfer.** There is no KWin scripting and no session bus, so every
window/tray probe the Linux work leaned on is gone. On Windows:

- Window facts: PowerShell + UIAutomation
  (`System.Windows.Automation`) or plain Win32 through ctypes
  (`FindWindow`/`GetWindowText`/`GetWindowThreadProcessId`) to prove which
  process owns a window. Taskbar grouping and the pinned-shortcut match are an
  **eyes-on-screen check** — screenshot with .NET
  (`[Drawing.Graphics]::CopyFromScreen`) and crop, the way `spectacle` + crop
  was used on Plasma.
- Tray facts: no DBus, no `GetLayout`, no `AboutToShow` to call — the icon and
  its menu are a screenshot and a mouse. Remember the overflow flyout (W3).
- Test speech: no `say`. Use `System.Speech.Synthesis.SpeechSynthesizer` from
  PowerShell to a WAV, or piper the way `capture-linux` does in `ci.yml`.
- Killing a stuck run: `pkill -f` matched the agent's own command line on Linux
  and killed the session; here use `Get-Process`/`Stop-Process -Id` with a
  resolved pid, one at a time.

---

## Docs that change when the above lands

`README.md:13` still says "Windows support is in progress", and the launcher
section (`README.md:96-127`) names only the macOS Desktop icon and the Linux
menu entry — W1 gives Windows a Start-menu entry to name, and W3–W6 are what
the status line is actually waiting on.
