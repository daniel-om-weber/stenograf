# stenograf — instructions for Claude Code

Meeting transcription pipeline: capture → live captions (TUI) → finalize
(diarized transcript) → notes. Shipped on PyPI as `stenograf`.

**Three front-ends, one library.** The CLI (`cli/`), the Textual launcher
(`ui/`) and the Qt desktop app (`gui/`, `steno --gui`) are all thin: they gather
inputs and call library entry points. The workflows they share live in
`flow.py` (meeting request → run, transcribe, notes, settings report) and
`captions.py` (live-caption line rules). Logic a screen needs that the library
lacks goes into the library — never into one front-end, or the two UIs drift.

`PLAN.md` holds **only unbuilt work** — the active plan is Phase 8's remainder
(the app is built, opt-in, installed as `~/Applications/Stenograf.app` by
`steno setup`, and lives in the menu bar; **only step 7 — the default flip — is
left, and it is gated on real use, not on code**. Step 5's bundle is FROZEN:
its bytes are every user's microphone grant, so read
`native/appbundle/README.md` before touching anything under it or under
`src/stenograf/assets/` — new sibling files there are fine, the `.app` tree is
not. Step 1's per-process profile is done and its watt half is deferred to a
quiet machine), plus the open platform items and the declined list. Everything
shipped, including the architecture and model-choice research, the AEC design
and the code-cleanup backlog, was pruned on 2026-07-25 and lives in git
history (`git log --follow -p PLAN.md`, and the deleted `PLAN-AEC.md` /
`PLAN-CLEANUP.md`). Measured evidence for the shipped defaults is in
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

## Current focus: Phase 5 — Linux (ACTIVE since 2026-07-10)

Two machines, sequenced: ONNX ASR backend on the Mac first (only place the MLX↔ONNX parity
harness runs), then capture work moves to the CachyOS notebook (x86_64,
real PipeWire); GitHub Actions Ubuntu is the stable-distro CI reference.

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
