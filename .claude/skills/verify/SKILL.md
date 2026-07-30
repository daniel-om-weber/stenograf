---
name: verify
description: How to run and observe stenograf end-to-end (CLI live pass, finalize, Qt app) without live capture hardware.
---

# Verifying stenograf changes at the surface

The product surface is the Qt desktop app (bare `steno` / `steno --gui`) and
the `steno` CLI. Everything is drivable headlessly via `--replay` — no mic, no
system tap, no native helper needed.

## The UI net — CLI-green does not imply the app works

The CLI does its loader work with click on its own stdio; the app does it
under a live GUI, through the `announce` seam and a worker thread — a code
path no `--replay` CLI run touches (this class of gap is how the win32 EBADF
meeting-start crash shipped despite a fully validated CLI, 2026-07-12, back
when the Textual launcher existed). The replacement net since the TUI's
retirement is `tests/test_gui.py`'s offscreen Qt harness
(`QT_QPA_PLATFORM=offscreen`): every page is instantiated with its real
controller, driven exactly as QML drives it (`opened()`, `start()`,
`stop()`), and the Qt message handler must stay silent — a QML binding error
is a warning, not an exception, so an unwatched app "works" while rendering
nothing. Extend that harness, not a mock layer.

Patch `stenograf.output.default_output_home` to a scratch dir first. On a
machine with audio devices the harness can run the REAL provider + models end
to end. When a test must fake (CI has no audio), fake at the hardware
boundary (`capture.windows.default_devices`, the provider class), never at
`loaders.make_provider`/`load_backends` — faking the orchestration seam is
exactly what hid the EBADF crash from the UI suite.

Rendering itself still needs eyes — nothing headless can tell whether the
live caption screen reads well over half an hour. `QQuickWindow.grabWindow()`
screenshots the app without any Screen Recording permission if a visual
record is needed.

## Build & launch — the plain live pass

No build step; run from the repo with uv:

```bash
uv run steno start --local 1 --remote 1 --no-aec \
  --out <tmpdir> --replay mic.wav,remote.wav
```

- `--out DIR` is the meeting's own folder — files land directly in it as
  `transcript.{md,json,txt}`. Without it, a `meeting-YYYYMMDD-HHMMSS/` folder
  is created under the output home (`[output] dir` in settings.toml, else
  `~/Documents/Meetings`) — always pass `--out` when testing to stay out of
  the user's real meetings. Re-running into the same `--out` needs `--force`
  (an existing transcript is refused); fresh tmpdir per run avoids it.
- `--replay MIC[,SYSTEM]` replays wav files as the two channels. With the live
  pass on (default), replay is **paced to wall-clock**, so a 2-minute file takes
  2 minutes — that is the point: it exercises the LiveWorker at meeting cadence.
- The live view is the plain line stream on every stdout, TTY or not — the
  only terminal live mode since the TUI retired. Committed captions print
  line by line ("You:"/"Remote:"), so latency is measurable straight off the
  stream: timestamp first occurrences of sentinel words against wall clock.
- The process exits on its own after finalize (and after notes, with
  `--notes`); transcript files are written before the notes step runs.
- Bare `steno` (no subcommand) opens the Qt window only from an interactive
  terminal with a display; from a pipe or script it prints help — so scripted
  checks can safely invoke it to assert the dispatch, but use `steno start`
  to run a meeting.

## Test audio

Synthesize speech with macOS TTS; Silero VAD and parakeet handle it fine:

```bash
say -o mono.aiff "<long text>"
afconvert -f WAVE -d LEI16@16000 -c 1 mono.aiff mono.wav
```

Stitch scenarios (silence gaps, multi-utterance) with numpy + the stdlib `wave`
module at 16 kHz mono int16. A silent same-length `mic.wav` makes a
remote-only scenario. TTS reads continuously — good stand-in for a monologue.

## Gotchas

- `--out` isolates the transcript but NOT the notes export: the user's
  standing settings.toml can set `[notes] auto = true` plus a
  `[notes.export] dir` inside the real Obsidian vault, so a test run then
  writes a combined note there (it happened 2026-07-30). Pass `--no-notes`
  on every scripted run unless notes are what you are testing — and if they
  are, check `steno settings show` for an export dir first.
- ASR mishears TTS a little ("caching"→"cucking") — irrelevant to plumbing
  checks; pick distinctive sentinel words (e.g. "provisioning", "October").
- `--no-aec` avoids the echo canceller wrapping the provider (one less
  variable when the mic channel is silent anyway).
- Model load adds ~4–8 s before capture starts; account for it when mapping
  audio time to wall time (capture itself starts immediately and buffers).
- A user's real `steno start` (installed via uv tool) may be running — check
  `pgrep -fl steno` before killing anything.
- Backgrounded runs cannot be stopped with `kill -INT`: POSIX non-interactive
  shells start `&` children with SIGINT *ignored*, and Python inherits that
  disposition — the signal is a silent no-op and the meeting runs forever
  (SIGTERM kills without finalizing). Use `--max-seconds N` to end a scripted
  run; it stops capture by itself and runs the full finalize.
