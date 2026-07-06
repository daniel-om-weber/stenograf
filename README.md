# stenograf

Accuracy-first, fully local meeting transcription for **German** and **English**
(one language per meeting), with speaker labels. Audio is processed entirely
**in memory** — it never touches disk; only the transcript is persisted.

Built for Apple Silicon (M-series) first; Linux and Windows support is designed
in from the start.

> **Status: pre-alpha.** The transcription pipeline works today — batch
> transcription of recorded files (`steno transcribe`) and the two-pass live
> captions (`steno start --replay`, driving the real streaming + finalize
> passes). Native macOS system-audio capture still needs its Core Audio helper,
> so live capture of a real meeting isn't wired up yet. See [PLAN.md](PLAN.md).

## Why another transcription tool?

- **No audio on disk, ever.** Live transcription of a meeting has far lighter
  legal requirements than recording it. stenograf keeps the session's audio in
  RAM only and writes nothing but text.
- **Accuracy first.** A two-pass design: fast live captions while the meeting
  runs, then a high-accuracy re-transcription of the full in-memory buffer the
  moment it ends. German is a first-class citizen, not an afterthought.
- **Channel-aware speakers.** Microphone and system audio are captured as
  separate streams, so local and remote voices never get confused; diarization
  handles the rest (2–8 speakers).

## Usage

```sh
uv tool install stenograf

steno doctor                                # first-run checks & model download
steno start                                 # live captions, everything auto-detected
steno start --lang de --local 3 --remote 2  # hybrid meeting, German
steno transcribe recording.mov              # batch-transcribe an existing file
```

`steno start` streams **live captions** while the meeting runs — a full-screen
TUI on a terminal, a plain line-by-line stream when piped — and replaces them
with the high-accuracy, speaker-labelled transcript the moment you stop
(Ctrl-C). The audio stays in RAM throughout; only the transcript is written.

Useful flags:

```sh
steno start --plain                 # plain caption stream instead of the TUI
steno start --no-live               # skip live captions; just finalize on stop
steno start --flush-interval 60     # crash-checkpoint the captions every 60s
steno start --replay mic.wav        # dev: drive the live pass from a file
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```sh
uv sync
uv run pytest
uv run steno doctor
```

See [PLAN.md](PLAN.md) for the full architecture, model choices, and roadmap.

## License

[MIT](LICENSE)
