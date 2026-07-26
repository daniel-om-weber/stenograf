# stenograf on Windows — what is missing, and where to measure it

The Linux side-plan closed with no work open and was deleted on 2026-07-26
(`git log --follow -p PLAN-LINUX.md` has the evidence, the decisions and the
container ladder). **This one still holds open work**, but much less than it
did: the 2026-07-26 session ran on the notebook and closed five of the six
items. Read in order: the machine, then what is already at parity (do not
re-derive it), then the one open item, then how to observe any of it again.

---

## The machine (measured 2026-07-26)

**Every "automatable" Windows number before 2026-07-26** (transcribe,
`--replay`, live WASAPI capture, DirectML byte-identity, 2026-07-11) was taken
on a desktop with an RTX 4080 SUPER. Everything below was taken on the
notebook, which is the machine the open item still needs.

- GPD `G1617-02`, Windows 11 Home 10.0.**26200**, on battery.
- **AMD Radeon 890M** (integrated, DX12) — the second DirectML vendor tier, and
  an iGPU sharing system RAM.
- The panel runs **1920×1080 at 150 % scale** (1280×720 logical). It was
  3840×2160 when this file was first written; the *scale* is what the app sees
  and that has not moved.
- Realtek HD Audio, real speakers and a built-in mic. **The capture endpoint
  has driver audio-processing objects installed and enabled**, which turns out
  to matter more than the speakers do — see W4.
- Windows Terminal (`wt.exe`) present; the microphone consent store reads
  `Allow`, so `steno setup`'s privacy-toggle check passes here.
- **The default SAPI voice is `Microsoft Hedda Desktop` (de-DE).** Left alone it
  reads English test text with German phonetics, and the ASR output is then so
  mangled ("throughput" → "Troput", "peak hours" → "Pika Oaz") that it reads as
  a model failure and is nothing of the kind. `Microsoft David Desktop` and
  `Microsoft Zira Desktop` (en-US) are installed — **always `SelectVoice`
  explicitly**. With David the same passage comes back near-verbatim.

State of *this checkout*, so a session does not discover it mid-task:

- `uv sync --extra gui` is the setup line (uv 0.11.32). The GUI extra is not
  optional for anything below.
- **No Rust** (`cargo` absent), so `hatch_build.py` skips `stenodiar.exe` and a
  checkout install has **no diarization helper**: estimated speaker counts fall
  back to sherpa's threshold clustering, which over-splits badly. Explicit
  `--speakers N` (and `--local`/`--remote`) is unaffected. To exercise the real
  helper, install the published `win_amd64` wheel instead of running from the
  checkout, or install Rust and rebuild.
- **No Ollama**, which on Windows is the default *and only* supported notes
  backend — so `steno notes` and `--notes` cannot run here. `steno doctor`
  reports that as an optional check, correctly.
- The ASR weights were half-downloaded (a zero-byte `.incomplete` blob, and
  `snapshot_download` then reporting success without fetching
  `encoder-model.onnx.data`). `steno setup --models-only` does **not** repair
  that; a direct `hf_hub_download` of the missing file does.

---

## Already at parity — do not re-derive

CI proves the first group on every push and every tag; the second group was
measured by hand on this notebook on 2026-07-26.

**Proven by CI**

- **Live WASAPI capture**, mic + loopback-of-default-output, through
  `soundcard` (`capture/windows.py`); the queue/pump/clock machinery is shared
  with Linux (`capture/streaming.py`).
- **ONNX Parakeet** with the DirectML flavor declared per-platform and `dml`
  opt-in through `[asr] provider` (`asr/providers.py`). CPU is the default.
- **`stenodiar.exe` in the `win_amd64` wheel** — built by `release.yml`'s
  `build-windows`, asserted present by `smoke-windows`, which installs the wheel
  the way a user does and runs the pipeline end to end from a file.
- **`steno setup`**: the mic privacy toggle is read up front and named
  (`cli/doctor_cmd.py`) — the honest Windows counterpart to macOS's TCC prompts,
  since no prompt exists here — then the launcher, then the models.
- **`install.ps1`**, the one-command install, mirroring `install.sh`.

**Measured on the notebook, 2026-07-26** — the four items this file was written
to gate, plus the two code items it specified.

- **W1 — the launcher is a real shortcut.** `steno setup` with the `gui` extra
  now writes `Stenograf.lnk` to both the Start Menu and the Desktop through COM
  (`winlink.py`: `IShellLink` + `IPersistFile` + `IPropertyStore`, ctypes only,
  no pywin32), carrying `assets/icon.ico` and :data:`APP_USER_MODEL_ID`, and
  targeting `pythonw.exe` so no console exists to flash. Without the extra the
  `Stenograf.cmd` stays — a console app wants a console. Converting either way
  leaves exactly one launcher, and only ever deletes a file that looks like ours
  (the `.cmd`'s `@echo off` sniff; the `.lnk`'s app id). If COM refuses, setup
  degrades to the batch file rather than failing.
- **W2 — the process claims its identity.** `gui/app.py`'s `run()` calls
  `SetCurrentProcessExplicitAppUserModelID` before the first window exists.
  Verified live: `GetCurrentProcessExplicitAppUserModelID` reports
  `dev.stenograf.app`, the same string the `.lnk` declares.
- **W3 — the desktop app on a real session.** All green.
  - The taskbar button carries the app's own icon, not a Python one.
  - `showMessage` on finish renders as a **real Windows toast titled
    "Stenograf"** with our mark — which is the payoff for W1+W2 together, since
    Windows attributes a desktop app's toast by matching a Start-menu shortcut's
    AppUserModelID to the process's.
  - The tray icon lands in the **notification-area overflow flyout**, exactly as
    this file predicted, and re-inks red while recording (photographed).
  - Layout holds at **125 %, 150 % and 200 %** (`dpr` 1.25/1.5/2.0): text crisp,
    long paths wrapped, cards scrolled rather than clipped.
  - **Closing the window mid-meeting hides it and recording continues**, and
    *Stop & finalize* from the tray wrote the transcript with no window on
    screen.
  - **The named-pipe single instance works completely** — the half no other
    platform exercises. `\\.\pipe\stenograf-<user>` is claimed; a second launch
    exits 0 silently and restores the hidden window; a `Stop-Process -Force`
    kill leaves no pipe behind (the kernel reclaims it) and the next launch
    starts normally. `tests/test_gui.py`'s crashed-instance test still skips
    here because it asserts on a *file*, which a named pipe is not — the
    behaviour it protects is nonetheless confirmed.
  - Found and fixed on the way: `tests/test_gui.py` was writing real meeting
    folders into the developer's own `~/Documents/Meetings`, **on Windows
    only** — a daemon meeting thread outliving its monkeypatch, reaching the
    real `make_provider` after teardown, where Windows (alone) has a capture
    stack that needs no helper and no permission and therefore says yes.
- **W5 — the TUI in Windows Terminal.** Captions render correctly; Ctrl-C stops,
  finalizes, swaps in the diarized transcript and waits on `q` as designed, with
  no orphan process. The Textual launcher also drives cleanly into a real
  meeting under `run_test` here, which is the path the 2026-07-12 win32
  meeting-start `EBADF` crash slipped through. **One finding, and it is not a
  Windows bug:** narrowing the terminal mid-meeting leaves already-written
  caption lines at their old wrap behind a horizontal scrollbar instead of
  re-wrapping — `RichLog` renders strips at write time. Widening restores them,
  and the on-stop finalize re-renders at the current width. Filed in PLAN.md.
- **W6 — DirectML on the AMD tier.** Byte-identical to CPU on both a 99 s and a
  594 s sample, across `transcript.{txt,json,md}`. `unavailable_reason` fires
  rather than falling back silently: asking for `cuda` against the DirectML
  build prints *"acceleration unavailable (cuda: CUDAExecutionProvider is not in
  this onnxruntime build (available: DmlExecutionProvider, CPUExecutionProvider))
  — using CPU"*. **The speed claim is where the tiers part.** On 594 s of audio
  the ASR pass goes 18.7× → 27.4× realtime (**1.47×**) and end-to-end 36.6 s →
  27.5 s (**1.33×**), against the desktop's 16× → 107× (**6.6×**) on an RTX 4080
  SUPER. So DML on an iGPU sharing system memory is a real but modest win, not
  the order-of-magnitude one — a settings-documentation fact, not a bug, and the
  reason CPU stays the default.

---

## Open — needs this machine

### W4 — AEC over real speakers, ≥30 minutes

**Attempted 2026-07-26 and inconclusive, for a reason worth more than the
attempt.** A 33-minute far-end track (84 s of en-US speech, 15 s of silence, ×20)
was played over the Realtek speakers at 40 % while `steno start --local 1
--remote 1 --aec-dump` captured mic + loopback. It was stopped at 6 minutes once
the dump had already settled the question:

    duration            366.0 s  (far end active 252.2 s)
    mic during far      -40.5 dBFS
    residual (enh)      -41.2 dBFS
    ERLE                  0.7 dB

0.7 dB reads as a dead canceller. It is not. **No echo ever reached the
microphone**: the mic sits at −50.7 dBFS while the speakers play and −49.7 dBFS
while they are silent — the same level, i.e. its own noise floor either way —
and the loopback↔mic envelope correlation is ~0.07 with no stable lag (the
per-minute best lag wanders between −1360 ms and +2000 ms, which is what
correlating noise against noise looks like). ERLE is undefined without an echo
path, so **this run did not exercise the canceller at all**. The most likely
cause is the driver audio-processing on the capture endpoint (`FxProperties` is
populated and `Disable_SysFx` is unset) doing its own echo suppression upstream
of us; the 40 % volume on a small chassis is the other candidate.

One thing the run *did* show, and it is the user-visible failure this whole
feature exists to prevent: in six minutes, one caption of far-end text —
`"comes down to operation."`, a fragment of *"the decision therefore comes down
to operational cost"* — was attributed to the **local** channel. Faint residual
is still occasionally decodable by Parakeet even when it is 50 dB down.

**Before the next attempt, establish the echo path** — `eval/aec_echo_present.py`
is the one-minute check that exists because this run skipped it:

    steno start --local 1 --remote 1 --max-seconds 60 --aec-dump probe \
        --out probe-meeting --plain     # …with speech over the speakers…
    uv run python eval/aec_echo_present.py probe

It passes when the mic is ≥ 6 dB louder while the speakers play than while they
rest. To get there: turn the microphone's **audio enhancements off** (Settings →
System → Sound → the input device → Audio enhancements → Off) and raise the
output volume. Only then is the ≥30-minute run worth anyone's speakers — and it
wants a machine nobody is sitting at, since it is half an hour of audio out loud.
The failure to look for is unchanged: remote voices transcribed as a local
speaker, especially after a multi-second silence, because WASAPI loopback
delivers no packets while nothing renders (`soundcard` synthesizes zeros from
wall-clock time) and the session clock re-anchors past
`_REANCHOR_TOLERANCE_S = 0.5`, which moves far-end alignment.

---

## Decided — do not re-litigate

- **Notes on Windows are Ollama or nothing.** No in-process backend unless
  Ollama proves insufficient, and then it is llama-cpp-python CPU GGUF
  (Phi-4-mini) with off-PyPI wheels. `onnxruntime-genai-directml` was evaluated
  and rejected (DX12-GPU-only on the EOL DirectML EP). macOS's zero-setup mlx-lm
  default has no Windows counterpart by decision, not by omission.
- **Exactly one onnxruntime flavor may be installed**, so DirectML is a platform
  marker, not an extra (`pyproject.toml`). Do not "add" CUDA on Windows.
- **CPU stays the default provider**, now with a second data point: DML buys
  1.3–1.5× on an iGPU against 6.6× on a discrete card (W6). It is worth
  documenting, not worth defaulting to.
- **Windows needs no launcher-identity trick.** A locally written file carries
  no Mark-of-the-Web, and there is no TCC equivalent to pin an identity to — so
  the frozen-bundle discipline that governs `Stenograf.app` does *not* apply to
  anything W1 writes. Regenerate freely; `steno setup` self-heals the embedded
  interpreter path, and both `.lnk`s are rewritten on every run.
- **`IShellLink` through ctypes, not `WScript.Shell`.** The PowerShell detour
  cannot reach `IPropertyStore`, so it cannot set the AppUserModelID, and a link
  without one matches nothing. VBScript would reach it and stays rejected —
  Windows is removing it. `pywin32`/`comtypes` are not worth a dependency for
  one file written once per install.
- **`soundcard` over `pyaudiowpatch`** (spiked on real hardware 2026-07-11): one
  API for both channels, and server-side resampling via `AUTOCONVERTPCM`.
- **The diarizer runs `cpu` here** (`diarization/speakrs.py`); the `cuda`
  feature type-checks in CI but is not in the shipped wheel, and nothing asks
  for it yet.

---

## Observing the app on this machine

The `verify` skill covers CLI/TUI headlessly and is still right; none of it
reaches a desktop session. What follows is what actually worked on 2026-07-26 —
the harnesses live in git history, not in the repo.

**The recipe.** Drive the real app from a Python harness: wrap
`stenograf.gui.tray.install` to capture the shell object, arm `QTimer.singleShot`
chains that navigate (`shell.open("Doctor")`) and submit the setup form with the
same map QML sends, and save `window.grabWindow()` per screen — it needs no
screen-capture permission and captures at the real scale factor. **Call `run()`,
not `build()`, whenever identity matters**: the application name, the icon, the
AUMID and the single-instance claim are all set inside `run()`. Keep meetings out
of `~/Documents\Meetings` with `STENOGRAF_DATA=<scratch>` plus a `settings.toml`
holding `[output] dir`. Give the GUI a meeting without a microphone by wrapping
`loaders.make_provider` to force `--replay`'s own file provider (`paced=True`, so
it runs at meeting cadence) — that is the documented replay seam, not a stub of
the orchestration.

Five traps, all paid for:

- **A non-zero `QTimer.singleShot` armed before the `QApplication` exists is
  silently dropped** — there is no event dispatcher to start it on, and `run()`
  builds the application itself. Arm a zero-delay one first and set up the real
  timeline from inside it.
- **`SHGetPropertyStoreForWindow` returns an empty store for our window**, and
  that is correct, not a bug: a window with no *explicit* AppUserModelID inherits
  the process's, and the inheritance is not visible in the store. Check
  `GetCurrentProcessExplicitAppUserModelID` instead, and read the taskbar with
  your eyes.
- **`EnumWindows` through a PowerShell delegate found nothing** while
  `Get-Process ... | MainWindowHandle` found the window immediately. Use the
  latter.
- **Screenshots:** `[Drawing.Graphics]::CopyFromScreen` in a DPI-*unaware*
  process silently captures an upscaled 1280×720 of a 1920×1080 screen. Call
  `SetProcessDPIAware()` first. Capture the smallest region that answers the
  question — a window rect via `GetWindowRect`, or a horizontal band — rather
  than the whole desktop, which is someone's private screen.
- **Test speech:** `System.Speech.Synthesis`, and **name the voice** (see the
  machine notes above). There is no `say` and no `script`/pty, so the TUI is
  driven by launching `wt.exe --title <marker> -- <python> -m stenograf start …`
  and photographing the window; `SendKeys` `^c` into the foreground terminal is a
  real Ctrl-C to it.
- Killing a stuck run: `Get-CimInstance Win32_Process` filtered on `CommandLine`,
  then `Stop-Process -Id` one at a time. `pkill -f` does not exist, and matching
  too loosely kills the agent's own shell.
