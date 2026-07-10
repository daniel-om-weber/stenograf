# stenograf — instructions for Claude Code

Meeting transcription pipeline: capture → live captions (TUI) → finalize
(diarized transcript) → notes. Shipped on PyPI as `stenograf` 0.1.0. Full
history, current state, and the active plan live in `PLAN.md` (§5 = phase
plan); echo-cancellation deep-dive in `PLAN-AEC.md`. Use the `verify` skill
to run/observe the tool without live capture hardware.

## Product philosophy (LOCKED — don't re-litigate)

The tool ends at **transcript + notes in a visible folder**
(`~/Documents/Meetings`). No meeting management, browsing, playback, index,
or web UI — ever. Obsidian/Finder/audio players do that; the filesystem is
the index.

## Workflow conventions

- Commit straight to `main` for planned work; branch only for large
  unplanned multi-commit refactors.
- Never mention Claude or AI assistance in commit messages or PR bodies —
  no `Co-Authored-By: Claude`, no "Generated with" trailers.
- Delegate token-heavy research fan-outs to cheaper Opus subagents rather
  than running them in the main loop.
- Release = version bump + tag (CI publishes to PyPI).

## Current focus: Phase 5 — Linux (ACTIVE since 2026-07-10)

Two machines, sequenced (details in PLAN.md §5 "Development-environment
plan"): ONNX ASR backend on the Mac first (only place the MLX↔ONNX parity
harness runs), then capture work moves to the CachyOS notebook (x86_64,
real PipeWire); GitHub Actions Ubuntu is the stable-distro CI reference.

## Platform decisions not recorded in PLAN.md

- **Notes backends per platform** (researched 2026-07-10): macOS = mlx-lm,
  pinned to the 0.29 line (0.30+ requires transformers>=5 which is
  import-broken, and Voxtral needs 4.x); generation must stay bound to the
  thread that imported `mlx_lm` (guarded in code). **Linux CPU fallback =
  llama-cpp-python — its wheels are OFF-PyPI** (extra index URL required);
  Ollama stays the default when a GPU box is available. Windows (future) =
  onnxruntime-genai-directml + Phi-4-mini.
- **MLX on background threads**: materialize weights on the load thread or
  inference dies with "no Stream(gpu, 0)"; verify MLX-threading changes
  against the real backend, not mocks.
- **Diarization licensing**: DiariZen is CC-BY-NC — unshippable here.
  speakrs (cross-platform: CoreML on mac, ORT elsewhere) is the chosen
  auto-count estimator, still immature at v0.5.0.
