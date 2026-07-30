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
**`PLAN-CAPTURE-HELPER.md` is the one live side-plan and the live design work**
(2026-07-26, evidenced, not built): arrival-stamped audio is the root cause
behind both the Windows far-end lag and the unmeasured Linux one, and the fix is
a native Rust capture helper per platform emitting the frame format
`capture/macos.py:9-16` already defines — one clock for both taps, as `stenocap`
has always done. It deletes `far_end_lag_s` rather than tuning it, drops
`soundcard` and `parec`, and expands the wheel matrix (capture becomes
mandatory, so today's untagged platforms would otherwise lose it). Vendoring or
monkeypatching soundcard, and PortAudio, are rejected there with reasons; the
livekit question is deferred with a trigger. `eval/wasapi_timestamps.py` is its
evidence and re-runs in twelve seconds. **Read it before touching capture,
`aec.py`, or `hatch_build.py` on any platform.**

**Three side-plans closed and were deleted; their evidence is in git history,
and none of it should be re-derived from scratch.** `PLAN-LINUX.md`
(2026-07-26): evidence, decisions and the container ladder. `PLAN-WINDOWS.md`
(2026-07-27): five of six items green — the `.lnk` launcher, the
AppUserModelID, the app on a real session, the TUI in Windows Terminal, DirectML
on the AMD tier — plus the AEC bug it found (the loopback tap's arrival stamps
run ~60 ms behind the mic's, AEC3 only searches backwards, so
`CaptureProvider.far_end_lag_s` now corrects it: 2.6 → 13.7 dB ERLE, two leaked
lines → none). Its last section is the observation recipe for a real Windows
desktop session (screenshot DPI, SAPI voice selection, German-locale traps,
driving the TUI without a pty) — read it before observing anything there. Its
one leftover, an optional AEC-quality run, is in PLAN.md and gates nothing.
`PLAN-ASR-CHALLENGER.md` (2026-07-27): the recurring "leaderboard has a new
leader" question is **declined**, not gated — see PLAN.md's declined list before
evaluating any ASR model. Retrieve any of them with
`git log --follow -p <file>`. Measured evidence for the shipped defaults is in
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
- **MLX on background threads**: materialize weights on the load thread or
  inference dies with "no Stream(gpu, 0)"; verify MLX-threading changes
  against the real backend, not mocks.
- **Diarization licensing**: DiariZen is CC-BY-NC — unshippable here.
  speakrs (cross-platform: CoreML on mac, ORT elsewhere) is the chosen
  auto-count estimator, still immature at v0.5.0 — **vendored with two
  CPU-throughput patches** in `native/stenodiar/vendor/` (see VENDOR.md;
  upstream candidates, re-check on every speakrs release).
