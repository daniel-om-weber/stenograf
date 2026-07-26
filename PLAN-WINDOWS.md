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
- Realtek HD Audio, real speakers and a built-in mic. The capture endpoint has
  driver audio-processing objects installed and enabled (`FxProperties`
  populated, `Disable_SysFx` unset) — **which turned out not to matter**: at
  90 % output volume the mic hears the speakers 22 dB above its own noise floor
  with the enhancements left on. **Volume is the knob, not the APO.** At 40 % no
  echo reaches the mic at all, which cost one 33-minute run (W4).
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

## W4 — AEC over real speakers

### The canceller was broken on Windows, and it is fixed (2026-07-26, second session)

Two attempts, and the second one found a real bug. Read this before running
anything: the shape of the failure is now known and the open work is much
narrower than "run it for 30 minutes".

**Attempt 1 was inconclusive** — 33 minutes of far-end audio at 40 % volume,
scored 0.7 dB ERLE, and the reason was that **no echo ever reached the
microphone**: it sat at its own noise floor (−50.7 dBFS playing, −49.7 dBFS
silent) either way. ERLE is undefined without an echo path, so nothing about the
canceller was measured. `eval/aec_echo_present.py` exists because of that run.

**Attempt 2 established the echo path with the volume knob alone.** 90 % instead
of 40 % put the mic 22.5 dB above its noise floor while the speakers played — so
the driver audio-processing (`FxProperties` populated, `Disable_SysFx` unset)
was **not** the blocker, and nothing in Windows Settings had to be touched. That
hypothesis is now retired.

And with a real echo path, 80 seconds was enough to expose the defect:

| | attempt 2, as shipped | after the fix |
|---|---|---|
| ERLE | **2.6 dB** | **13.7 dB** |
| residual | −29.6 dBFS | −42.1 dBFS |
| far-end lines attributed to `Local-1` | **2** | **0** |

(macOS, for scale: 37.6 dB, −65 dBFS, 0 leaked lines.) The two leaked lines were
`"numbers from left one."` and `"we have all cast."` — garbled echoes of *"…the
throughput numbers from last week"* and *"…comes down to operational cost"*,
which is exactly the user-visible failure this feature exists to prevent.

**The cause was timestamps, not the canceller.** `EchoCanceller` pairs the two
channels *by timestamp*, and `SessionClock` stamps every channel when its frames
**arrive**, so each channel's timeline carries its own transport latency as a
constant offset. WASAPI's loopback tap is the longer path (render buffer →
endpoint mix → loopback capture → `AUTOCONVERTPCM` resampler → us), so the
reference was labelled **~60 ms later than its own echo** — measured off one dump
as −44 ms at sample resolution and −60 ms on the tick grid, stable across all
four quarters of the run, i.e. an offset and not drift. AEC3 aligns the near end
against far-end *history*: a reference that arrives after the echo is unusable at
any `set_stream_delay_ms` value, which is why the hint measured irrelevant back
in the design work and why this looked like a dead canceller.

Proven before any code changed, by re-running the real AEC3 over the captured
dump with the reference advanced by a sweep of offsets:

    0 ms → 4.7 dB    40 ms → 4.7 dB    60 ms → 15.1 dB    80 ms → 15.8 dB
    100 ms → 15.2 dB    150 ms → 14.5 dB    250 ms → 14.7 dB

A cliff at 60 ms and a flat plateau after it. **The error is one-sided** — early
is what the estimator searches, late is fatal — so the fix declares a *generous*
correction rather than a precise one.

**And the lag is not a per-machine number to look up: it is re-rolled at every
meeting start.** Each channel anchors on its own first frame and the two pump
threads open their recorders independently, so the same Realtek endpoint measured
60 ms on one 80-second run and 10–25 ms on the next, minutes later. A constant
fitted to the second run would have been 25 ms — dead on the first. That kills any
"measure it once per machine at install" idea, and it is the strongest argument
for the safety factor. `eval/aec_alignment.py` carries a 0.1 s floor for exactly
this reason.

The fix: `CaptureProvider.far_end_lag_s` (0.0 for
everyone whose channels share a clock) with `capture.windows.FAR_END_LAG_S =
0.15`, 2.5× the larger measurement, subtracted when the canceller files the
reference. Only the canceller's copy moves; the forwarded frame keeps the
provider's timeline, so the transcript, the dump and the merge are untouched.
`_MAX_HOLD_S` grows by the same amount, or the extra wait for a corrected
reference would be charged to the stalled-tap budget and every healthy meeting
would report reference loss.

**Do not judge the fix by re-measuring the dump.** `--aec-dump` records frames as
the provider stamped them, so `lpb.wav` still trails `mic.wav` afterwards. ERLE
and leaked `Local-N` lines are the measurements that moved.
`eval/aec_alignment.py` does the whole diagnosis in one command — measure the lag,
replay the dump through the real canceller at a sweep of corrections, print the
constant to ship — and it compares against what the provider already declares, so
a fixed machine reads PASS instead of shouting about the raw timeline.

### Still open — needs a machine nobody is sitting at

The ≥30-minute run, now against a canceller that works. What is left to learn is
specifically what 80 seconds cannot show:

- Whether alignment holds over half an hour. `soundcard` synthesizes zeros from
  wall-clock time while nothing renders, and the session clock re-anchors forward
  past `_REANCHOR_TOLERANCE_S = 0.5` — which moves far-end alignment, and is
  exactly what the 150 ms of headroom may or may not absorb. Multi-second
  silences before speech are the case to watch.
- Whether the residual gap to macOS (13.7 dB against 37.6 dB) is the chassis, the
  driver's processing, or a lag constant that could be tighter.
- Double-talk, which no run on this machine has covered at all.

Recipe: `eval/aec_echo_present.py` first (volume at 90 %), then the long run with
`--aec-dump`, then `eval/aec_score.py` and a count of `Local-N` lines in the
transcript. It is half an hour of speech out loud, so it wants an empty room.

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
- **Device-side timestamps — the real fix — are deferred, with a trigger.** macOS
  has no far-end lag because its helper does not use arrival time at all: CoreAudio
  hands it each buffer's Mach host time, for both taps, in one clock domain. Both
  other platforms expose the same thing — Windows in
  `IAudioCaptureClient::GetBuffer`'s `pu64QPCPosition`/`pu64DevicePosition` (QPC is
  machine-wide, so a mic stream and a loopback stream are directly comparable),
  Linux in `pa_timing_info`/`pa_stream_get_time` or PipeWire's `pw_time`. Neither
  of our *transports* carries them: `soundcard` returns bare sample arrays and drops
  the packet metadata, and `parec` is a pipe of raw PCM where every timestamp dies
  at the process boundary (`--latency-msec` only requests a target). So this is a
  transport rewrite on both — owning WASAPI capture through ctypes COM, and taking
  the libpulse/PipeWire dependency that the parec decision exists to avoid — for
  one number that a generous constant already gets right. **The trigger: a constant
  can only fix a constant.** If the ≥30-minute run shows the offset moving
  mid-meeting — plausible, since the forward re-anchor deliberately jumps a
  channel's timeline when `soundcard`'s zero-fill under-estimates a silence gap —
  then no declared value works and this becomes the fix rather than a refinement.
  It would retire the re-anchor heuristic at the same time, which is the other
  thing arrival stamping made necessary.
- **The far-end lag is a declared constant, generously set, not a measured
  runtime value.** A per-meeting delay estimator would be a second delay
  estimator in front of AEC3's own, and AEC3's works fine once the sign is right:
  the sweep is flat from 60 ms to 250 ms, so precision buys nothing and the only
  real risk is a driver whose tap is slower than 150 ms. Nor is it a setting —
  it is a property of a transport, not a preference, and no user can measure it.
  Revisit only if a machine turns up where the mic hears the speakers (the probe
  passes) and ERLE stays near zero anyway.
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
