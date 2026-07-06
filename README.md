# stenograf

Accuracy-first, fully local meeting transcription for **German** and **English**
(one language per meeting), with speaker labels. Audio is processed entirely
**in memory** — it never touches disk; only the transcript is persisted.

Built for Apple Silicon (M-series) first; Linux and Windows support is designed
in from the start.

> **Status: pre-alpha.** The architecture is settled (see [PLAN.md](PLAN.md)),
> the pipeline is under construction. Nothing transcribes yet.

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

## Planned usage

```sh
uv tool install stenograf

steno doctor                                # first-run checks & model download
steno start --lang de --local 3 --remote 2  # hybrid meeting, German
steno start                                 # everything auto-detected
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
