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

### Naming speakers across meetings

Enroll a voice once and every later meeting relabels that speaker automatically
(cross-meeting re-identification):

```sh
steno profiles enroll Daniel daniel-sample.wav   # a short clip of one speaker
steno profiles list                              # show enrolled voiceprints
steno profiles rename Daniel "Daniel W."
steno profiles remove Daniel
```

To name one person from a multi-speaker recording (e.g. a meeting saved with
`--record-audio`), diarize it and pick their cluster:

```sh
steno profiles enroll Anna meeting.wav --speakers 4          # lists the clusters
steno profiles enroll Anna meeting.wav --speakers 4 --speaker S2
```

Matching is on by default in `steno start`/`transcribe` and does nothing until
you enroll someone; disable it with `--no-reid`, or adjust the match strictness
with `--reid-threshold` (0–1, default 0.5). Voiceprints live in the platform data
dir (not the model cache) and are never uploaded.

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
