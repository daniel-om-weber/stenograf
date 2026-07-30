# stenograf — instructions for Claude Code

Meeting transcription pipeline: capture → live captions (TUI) → finalize
(diarized transcript) → notes. Shipped on PyPI as `stenograf`.

**Two front-ends, one library.** The CLI (`cli/`) and the Qt desktop app
(`gui/` — bare `steno`, `steno --gui`, or the app icon; the Textual launcher
was retired 2026-07-30) are both thin: they gather inputs and call library
entry points. The workflows they share live in `flow.py` (meeting request →
run, transcribe, notes, settings report) and `captions.py` (live-caption line
rules). Logic a screen needs that the library lacks goes into the library —
never into one front-end, or the Qt app and the CLI drift.

`PLAN.md` holds **only unbuilt work** — Phase 8 closed 2026-07-30 (all seven
steps shipped; the app is the default UI, installed as
`~/Applications/Stenograf.app` by `steno setup`, and lives in the menu bar).
What remains there: step 1's watt half (deferred to a quiet machine), the
step-7 observation gates still to run on this Mac, the open platform items
and the declined list. Step 5's bundle is FROZEN: its bytes are every user's
microphone grant, so read `native/appbundle/README.md` before touching
anything under it or under `src/stenograf/assets/` — new sibling files there
are fine, the `.app` tree is not. Everything
shipped, including the architecture and model-choice research, the AEC design
and the code-cleanup backlog, was pruned on 2026-07-25 and lives in git
history (`git log --follow -p PLAN.md`, and the deleted `PLAN-AEC.md` /
`PLAN-CLEANUP.md`).
**The capture helper is whole (2026-08-02): every platform streams
device-stamped audio from a `stenocap` helper, and no arrival-stamped
transport exists.** Arrival-stamped audio was the root cause behind both the
Windows far-end lag and the never-measured Linux one; the fix is one Rust
helper (`native/stenocap/` — WASAPI/`pu64QPCPosition` on Windows,
PulseAudio-protocol/`CLOCK_MONOTONIC − server latency` on Linux, serving
PipeWire through pipewire-pulse) speaking the frame format
`capture/helper.py` defines — one clock for both taps, as the macOS
`stenocap` has always done. `capture/helper.py` holds the transport all three
platforms share (the per-platform provider modules folded into it 2026-08-02;
only `capture/windows.py` remains, for the consent store and the loopback
suffix); `soundcard`, `parec`, `FAR_END_LAG_S`, `SessionClock` and
`far_end_lag_s` itself are all deleted. The wheel matrix carries it: win_amd64
and manylinux_2_39 bundle both helpers, and a low-floor **manylinux_2_28 wheel
carries stenocap alone** so Ubuntu 22.04 / Debian 12 / RHEL 9 keep live
capture (stenodiar's glibc-2.39 floor is onnxruntime's, not capture's).
`PLAN-CAPTURE-HELPER.md` closed and was deleted that day
(`git log --follow -p PLAN-CAPTURE-HELPER.md`) — **read its history before
touching capture, `aec.py`, or `hatch_build.py` on any platform**: vendoring
or monkeypatching soundcard and PortAudio are rejected with reasons, win_arm64
was struck on measurement (five base dependencies ship no wheel for it), and
the dependency-hygiene test its not-candidates table applies is not guessable
from code. Still owed, in PLAN.md: the AEC-quality run scoring the Windows
helper against the deleted constant's 13.7 dB (needs speakers in an empty
room), and the livekit re-ask whose trigger — the helper everywhere — has now
fired. `eval/wasapi_timestamps.py` is the original evidence and re-runs in
twelve seconds.

**`PLAN-DIARIZATION.md` is the one live side-plan (opened 2026-08-02):** the
diarization + speaker re-ID accuracy program, with its research record in
`eval/diarization-sota-2026.md`. Read both before touching `diarization/`,
`voiceprints.py`, `eval/der.py`, or evaluating any diarization or
speaker-embedding model — the declined list there (Sortformer, joint SA-ASR,
TS-VAD, AS-Norm-by-default, pyannote.audio-as-dependency) carries measured
reasons and re-open triggers.

**Six side-plans closed and were deleted; their evidence is in git history,
and none of it should be re-derived from scratch.** `PLAN-LINUX.md`
(2026-07-26): evidence, decisions and the container ladder. `PLAN-WINDOWS.md`
(2026-07-27): five of six items green — the `.lnk` launcher, the
AppUserModelID, the app on a real session, the TUI in Windows Terminal, DirectML
on the AMD tier — plus the AEC bug it found (the loopback tap's arrival stamps
run ~60 ms behind the mic's, AEC3 only searches backwards, so
`CaptureProvider.far_end_lag_s` corrected it: 2.6 → 13.7 dB ERLE, two leaked
lines → none; that constant is gone since the capture helper, and 13.7 dB is now
the number the helper must beat). Its last section is the observation recipe for
a real Windows desktop session (screenshot DPI, SAPI voice selection,
German-locale traps, driving the TUI without a pty) — read it before observing
anything there. Its one leftover, the AEC-quality run, is in PLAN.md and now
gates the helper rather than nothing.
`PLAN-ASR-CHALLENGER.md` (2026-07-27): the recurring "leaderboard has a new
leader" question is **declined**, not gated — see PLAN.md's declined list before
evaluating any ASR model. `PLAN-NOTES-MARKDOWN.md` and
`PLAN-MEETING-PRESETS.md` (both 2026-07-30, both built): notes are markdown
against a template that *is* the schema, and a meeting kind is a
`[meetings.<name>]` section selected with `--preset` (its UI half — the
"Meeting type" picker in the Qt setup form and the Settings screen, plus
`steno settings show --preset` — shipped 2026-07-31). Read the notes plan's
history before touching `notes/` — the deleted JSON schema was quietly doing
four jobs (structure, sanitizing, truncation detection, refusal detection) and
which replacement covers each is not guessable from the code. Their leftovers —
two Ollama gates that need a non-macOS box and Gate A's read — are in PLAN.md.
Retrieve any of them with `git log --follow -p <file>`. Measured evidence for the shipped defaults is in
`eval/README.md`; design rationale lives in the code's own docstrings. Use the
`verify` skill to run/observe the tool without live capture hardware.

## Product philosophy (LOCKED — don't re-litigate)

The tool ends at **transcript + notes in a visible folder**
(`~/Documents/Meetings`). No meeting management, browsing, playback, index,
or web UI — ever. Obsidian/Finder/audio players do that; the filesystem is
the index.

## Workflow conventions

- Commit straight to `main` for planned work; branch only for large
  unplanned multi-commit refactors.
- Before committing, the CI triple must pass locally: `uv run ruff check .`,
  `uv run --with pyright pyright` (macOS only — the mlx deps install there
  only, so it is the only place types resolve as CI sees them), and
  `uv run pytest -q`. `scripts/hooks/pre-commit` runs all three; enable it
  per clone with `git config core.hooksPath scripts/hooks`.
- Never mention Claude or AI assistance in commit messages or PR bodies —
  no `Co-Authored-By: Claude`, no "Generated with" trailers.
- Delegate token-heavy research fan-outs to cheaper Opus subagents rather
  than running them in the main loop.
- Release = version bump + tag (CI publishes to PyPI).

## Comment policy

1. Comments answer WHY. If a comment can be inferred from the line, delete it.
2. A measured claim keeps its number and its date; the date says when to
   re-measure.
3. Never name a phase, stage, task, step, or plan file. State the fact; git
   has the history.
4. Never describe what a file used to contain. Tombstones only where they
   stop a revert, in the present tense.
5. A fact lives at exactly ONE address — code, `eval/README.md`, or
   `PLAN.md`. Link, don't copy.
6. Module-docstring length is bounded by the cost of the mistake it prevents.

## Phase 8 step 7 SHIPPED and OBSERVED 2026-07-30 — the GUI is the default UI

The Textual front-end is retired — Daniel answered step 7's "pick one" with
full retirement, not a no-display fallback: bare `steno` opens the Qt window
only from an interactive terminal with a display (`_interactive_terminal()` +
`_display_available()` in `cli/__init__.py`); everywhere else it prints help,
and headless/SSH use is the line-oriented CLI (`steno start` streams plain
captions; `--plain` is a hidden accepted no-op). The observation gates are
green (Daniel ran the three launch gestures the same day). The decision
record and review findings: `git log --follow -p PLAN-GUI-DEFAULT.md`. The
shipped flip has not been released yet — release = version bump + tag.

**`--gui` must keep working forever** (`cli/__init__.py`): `Stenograf.app`'s
launcher stub is the frozen binary holding every macOS user's microphone
grant, and `--gui` is compiled into it as the fallback argv — it bypasses
both dispatch gates by design.

## Platform decisions

- **Notes backends per platform** (researched 2026-07-10): macOS = mlx-lm,
  pinned to the 0.29 line (0.30+ requires transformers>=5 which is
  import-broken, and Voxtral needs 4.x); generation must stay bound to the
  thread that imported `mlx_lm` (guarded in code). **Linux CPU fallback =
  llama-cpp-python — its wheels are OFF-PyPI** (extra index URL required);
  Ollama stays the default when a GPU box is available. **Windows (decided
  2026-07-12): Ollama-default, no in-process backend unless Ollama proves
  insufficient — then llama-cpp-python CPU GGUF (Phi-4-mini), same off-PyPI
  wheel friction as Linux.** onnxruntime-genai-directml was rejected
  (DX12-GPU-only on the EOL DirectML EP; coexistence with our
  onnxruntime-directml would actually have been fine — it depends on the
  same flavor).
- **Windows status icon** (built 2026-08-01, `gui/wintray.py`): the notification
  area is Windows' menu bar, and Qt's `QSystemTrayIcon` cannot reach the one
  field that makes it work — `NOTIFYICONDATA.guidItem`. Without it the shell
  files the icon under the *interpreter's* path, so the user's show/hide choice
  belongs to `pythonw.exe` and dies on the next Python bump. `gui/tray.py`
  therefore picks a hand-rolled `Shell_NotifyIcon` implementation on a real
  Windows session and Qt's everywhere else (offscreen included, so the tests
  never register a real icon). Windows 11 also hides every *new* tray icon and
  offers no API to promote one; `IsPromoted` under
  `HKCU\Control Panel\NotifyIconSettings` is honoured live, and is written once
  and only when absent. **Read the module docstring before touching it** — the
  GUID is permanent, and the path-binding caveat, the async key creation and
  the shared-WNDPROC rule are all measured, not guessed.
- **MLX on background threads**: materialize weights on the load thread or
  inference dies with "no Stream(gpu, 0)"; verify MLX-threading changes
  against the real backend, not mocks.
- **Diarization licensing**: DiariZen's multi-domain checkpoints are
  CC-BY-NC — unshippable; its `meeting-base` checkpoint is MIT (trained only
  on AMI/AISHELL-4/AliMeeting) and is `PLAN-DIARIZATION.md`'s step-5
  candidate.
  speakrs (cross-platform: CoreML on mac, ORT elsewhere) is the chosen
  auto-count estimator, still immature at v0.5.0 — **vendored with two
  CPU-throughput patches** in `native/stenodiar/vendor/` (see VENDOR.md;
  upstream candidates, re-check on every speakrs release).
