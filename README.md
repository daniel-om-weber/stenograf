# stenograf

Accuracy-first, fully local meeting transcription for **German** and **English**
(one language per meeting), with speaker labels. Audio is processed entirely
**in memory** — it never touches disk; only the transcript is persisted.

Built for Apple Silicon (M-series) first; Linux and Windows support is designed
in from the start.

> **Status: pre-alpha, macOS only.** The pipeline is complete end to end: live
> system-audio + microphone capture, live captions, and the high-accuracy
> speaker-labelled finalize pass. The local web UI, meeting notes, and Linux
> capture are not built yet. See [PLAN.md](PLAN.md).

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
- **Speakers, not headphones.** Remote voices leaving your laptop speakers and
  re-entering the mic are cancelled in the audio domain (WebRTC AEC3, with the
  system channel as the far-end reference), so they are never transcribed as a
  local speaker.

## Install

Requires macOS 14.4+ on Apple Silicon and [uv](https://docs.astral.sh/uv/).
The wheel ships the signed capture helper — no toolchain needed.

```sh
uv tool install stenograf
steno doctor    # environment checks
steno setup     # one-time: mic + system-audio permission prompts, model downloads
```

macOS scopes the permission grant to the app the prompt came from, so run
`steno setup` once from each terminal app (or IDE) you'll start meetings from.

Pre-release channel: `uv tool install git+https://github.com/daniel-om-weber/stenograf`
installs the current main branch; building from the repository compiles the
capture helper on your machine, which needs the Xcode command-line tools
(`xcode-select --install`).

### From a checkout

```sh
git clone https://github.com/daniel-om-weber/stenograf
cd stenograf
uv sync
sh native/helper/build.sh     # builds + ad-hoc signs native/helper/stenocap
uv run steno doctor
uv run steno setup
```

Every command below is then `uv run steno …` from the repo.

## Usage

```sh
uv run steno start                                 # live captions, everything auto-detected
uv run steno start --lang de --local 3 --remote 2  # hybrid meeting, German
uv run steno transcribe recording.mov              # batch-transcribe an existing file
```

`steno start` streams **live captions** while the meeting runs — a full-screen
TUI on a terminal, a plain line-by-line stream when piped — and replaces them
with the high-accuracy, speaker-labelled transcript the moment you stop
(Ctrl-C). The audio stays in RAM throughout; only the transcript is written.

Useful flags:

```sh
steno start --plain                 # plain caption stream instead of the TUI
steno start --no-live               # skip live captions; just finalize on stop
steno start --title "Weekly sync"   # name the meeting in the archive
steno start --flush-interval 60     # crash-checkpoint the captions every 60s
steno start --no-aec                # disable echo cancellation (headphones)
steno start --record-audio          # opt in to keeping a WAV (off by default)
steno start --replay mic.wav        # dev: drive the live pass from a file
```

Both `start` and `transcribe` accept `--format md,json,txt,srt,vtt` (default
`md,json,txt` — `txt` is the plain prose without speaker labels or timestamps),
`--lang de|en`, and `--print` to echo the transcript to stdout.

`steno transcribe` recognizes 2-channel recordings whose channels are separate
voice feeds — a `--record-audio` tee (mic left, system right) or a
dual-channel call recording — and transcribes them per channel through the
meeting pipeline (`Local-N`/`Remote-N` labels, per-channel diarization with
`--local`/`--remote` counts) instead of downmixing; ordinary stereo still
downmixes to mono. Force either way with `--channels split|mix`.

## Your meeting archive

Transcripts are filed automatically into a managed archive at
`~/Library/Application Support/stenograf/meetings/<id>/`. Use `--out DIR` to
write somewhere else (still archived), or `--no-archive` to write loose files
next to the source and register nothing.

```sh
steno meetings list                 # every transcript, newest first
steno meetings show meeting-20260710-091500
steno meetings rm meeting-20260710-091500
```

Audio is stored only when you passed `--record-audio`; without it the archive
holds text alone.

## Naming speakers across meetings

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

## Vocabulary

Domain terms and attendee names are corrected in the finalized transcript
(the ASR has no decode-time biasing, so this is a post-correction pass):

```sh
steno transcribe rec.mov --attendee "Anja Müller" --glossary Kubernetes,gRPC
steno transcribe rec.mov --glossary-file terms.txt
```

A term and its transcription must share a word count — `gRPC` can fix `G R P C`
spoken as one word, but not a term split across word boundaries.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```sh
uv sync
uv run pytest
uv run steno doctor
```

The test suite is label-free and runs without a meeting: model-gated and
real-audio tests self-skip when their assets are absent.

See [PLAN.md](PLAN.md) for the full architecture, model choices, and roadmap;
[PLAN-AEC.md](PLAN-AEC.md) for the echo-cancellation design and its measurements;
`native/README.md` for the capture helper and its wire protocol; `eval/README.md`
for the model-evaluation and AEC-scoring harnesses.

## License

[MIT](LICENSE)
