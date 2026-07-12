---
name: verify
description: How to run and observe stenograf end-to-end (TUI, live pass, finalize) without live capture hardware.
---

# Verifying stenograf changes at the surface

The product surface is the `steno` CLI / Textual TUI. Everything is drivable
headlessly via `--replay` — no mic, no system tap, no native helper needed.

## Headless TUI smoke — real stack, no TTY, any platform

CLI-mode green does NOT imply the launcher works: the CLI does its loader
work *before* Textual owns stdio, the launcher does it *under* a live app —
a code path no `--plain`/`--replay` run touches (this is how the win32
EBADF meeting-start crash shipped despite a fully validated CLI,
2026-07-12). Textual's `run_test` pilot drives the real launcher without a
terminal, so "the TUI needs eyes" is true only for rendering — plumbing is
verifiable headlessly:

```python
app = StenografApp()
async with app.run_test(size=(100, 40)) as pilot:
    await pilot.click("#start")          # Home -> setup form
    await pilot.pause()
    app.screen._submit()                 # form defaults -> meeting screen
    # ... pilot.pause() until MeetingScreen._phase is CAPTURING (model load
    # takes seconds), press "q" to stop+finalize, then inspect
    # app._notifications — a "Meeting failed" toast is the failure signal.
```

Patch `stenograf.output.default_output_home` to a scratch dir first. On a
machine with audio devices this runs the REAL provider + models end to end.
When a test must fake (CI has no audio), fake at the hardware boundary
(`capture.windows.default_devices`, the provider class), never at
`loaders.make_provider`/`load_backends` — faking the orchestration seam is
exactly what hid the EBADF crash from the UI suite.

## Build & launch

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
- The TUI only engages on a TTY. No tmux on this machine; use
  `/usr/bin/script -q ts.bin <command>` for a real pty and record the ANSI
  stream. Render frames from `ts.bin` with pyte (`uv run --with pyte python3`),
  feeding bytes up to a marker word to reconstruct the screen at that moment.
- Textual paints only *visible* rows, so "when does a word first appear in the
  stripped ANSI stream" ≈ "when did the user first see it". Timestamping first
  occurrences of sentinel words against wall clock measures caption latency.
- After finalize the TUI waits on `q` — the process does NOT exit on its own,
  and the transcript file is written only after dismissal. Drive with stdin
  DEVNULL → plan a timeout, and don't expect files in `--out` if you kill it.

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

- ASR mishears TTS a little ("caching"→"cucking") — irrelevant to plumbing
  checks; pick distinctive sentinel words (e.g. "provisioning", "October").
- `--no-aec` avoids the echo canceller wrapping the provider (one less
  variable when the mic channel is silent anyway).
- Model load adds ~4–8 s before capture starts; account for it when mapping
  audio time to wall time.
- A user's real `steno start` (installed via uv tool) may be running — check
  `pgrep -fl steno` before killing anything.
