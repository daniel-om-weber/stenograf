# stenograf

Accuracy-first, fully local meeting transcription for **German** and **English**
(one language per meeting), with speaker labels. Audio is processed entirely
**in memory** — nothing lands on disk but the transcript, unless you
explicitly opt into keeping a recording.

Built for Apple Silicon (M-series) first; Linux and Windows support is designed
in from the start.

> **Status: early release.** On macOS and Linux the pipeline is complete end
> to end: live system-audio + microphone capture, live captions, and the
> high-accuracy speaker-labelled finalize pass, plus meeting notes. Windows
> support is in progress. See [PLAN.md](PLAN.md).

## Why another transcription tool?

- **No audio on disk.** Live transcription of a meeting has far lighter
  legal requirements than recording it. stenograf keeps the session's audio in
  RAM only and writes nothing but text; keeping a WAV is a per-run opt-in
  (`--record-audio`).
- **Accuracy first.** A two-pass design: fast live captions while the meeting
  runs, then a high-accuracy re-transcription of the full in-memory buffer the
  moment it ends. German is a first-class citizen, not an afterthought.
- **Channel-aware speakers.** Microphone and system audio are captured as
  separate streams, so local and remote voices never get confused; optional
  diarization separates further speakers within a channel (2–8 speakers).
- **Speakers, not headphones.** Remote voices leaving your laptop speakers and
  re-entering the mic are cancelled in the audio domain (WebRTC AEC3, with the
  system channel as the far-end reference), so they are never transcribed as a
  local speaker.

## Install

One command sets up everything — [uv](https://docs.astral.sh/uv/) (installed
for you if missing), stenograf itself, the permission prompts, the model
downloads, and a desktop launcher:

```sh
curl -fsSL https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.sh | sh
```

That's the only command you have to type. Afterwards, double-click
**Stenograf** on your Desktop (macOS) or start **Stenograf** from the
application menu (Linux) to open the launcher — every workflow below is
reachable there with the mouse. Re-running the command upgrades stenograf.

Works on macOS 14.4+ on Apple Silicon (the wheel ships the signed capture
helper — no toolchain needed) and Linux with PipeWire or PulseAudio (capture
uses `parec`, shipped with pipewire-pulse / pulseaudio-utils on every desktop
distro; ASR runs ONNX on CPU).

### Manual install

With [uv](https://docs.astral.sh/uv/) already installed:

```sh
uv tool install stenograf
steno doctor    # environment checks
steno setup     # one-time: permission prompts, desktop launcher, model downloads
```

macOS scopes the permission grant to the app the prompt came from, so run
`steno setup` once from each terminal app (or IDE) you'll start meetings from.

Pre-release channel: `uv tool install git+https://github.com/daniel-om-weber/stenograf`
installs the current main branch; building from the repository compiles the
capture helper on your machine, which needs the Xcode command-line tools
(`xcode-select --install`).

### Windows

Same one command, run from PowerShell — installs uv if missing, then stenograf,
then the launcher:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.ps1 | iex"
```

Windows never prompts for the microphone, so capture stays silent until you flip
the toggle under Settings > Privacy & security > Microphone. Windows Terminal is
recommended for the live-caption TUI.

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

## The launcher

Double-click **Stenograf** (the Desktop icon on macOS, the application-menu
entry on Linux) — or run bare `steno` in a terminal — and a mouse-driven
launcher opens:

- **Start meeting** — capture this meeting with live captions; the
  speaker-labelled transcript replaces them the moment you stop.
- **Transcribe a recording** — turn an existing audio file into a transcript.
- **Generate notes** — summarize a finished meeting's transcript.
- **Settings** — show the active configuration.
- **Check setup** — verify models, permissions, and audio devices.

Everything the launcher does is also a plain CLI command — the rest of this
document — so terminal users and scripts lose nothing.

## Usage

```sh
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
steno start --title "Weekly sync"   # name the meeting (notes + export use it)
steno start --flush-interval 60     # crash-checkpoint the captions every 60s
steno start --no-aec                # disable echo cancellation (headphones)
steno start --diarization           # separate speakers within each channel (off by default)
steno start --record-audio          # opt in to keeping a WAV (off by default)
steno start --max-seconds 3600      # stop capture automatically after an hour
steno start --replay mic.wav        # dev: drive the live pass from a file
```

Both `start` and `transcribe` accept `--format md,json,txt,srt,vtt` (default
`md,json,txt` — `txt` is the plain prose without speaker labels or timestamps),
`--lang de|en`, `--diarization/--no-diarization` to run or skip speaker
separation for one run (off by default: each channel is one speaker and the
diarizer model is never loaded — it costs minutes on some machines;
`[speakers] diarization = true` in the settings makes running it the
default), and `--print` to echo the transcript to stdout.

If you know how many people spoke in a recording, tell `steno transcribe` with
`--speakers N` — a count above 1 turns diarization on, and a known count is
the biggest diarization accuracy lever (omitted, the count is estimated
whenever diarization runs).

`steno transcribe` recognizes 2-channel recordings whose channels are separate
voice feeds — a `--record-audio` tee (mic left, system right) or a
dual-channel call recording — and transcribes them per channel through the
meeting pipeline (`Local-N`/`Remote-N` labels, per-channel diarization with
`--local`/`--remote` counts) instead of downmixing; ordinary stereo still
downmixes to mono. Force either way with `--channels split|mix`.

## Where your files land

Every run writes its own date-named folder into `~/Documents/Meetings`:

```
~/Documents/Meetings/meeting-20260710-091500/
    transcript.md / .json / .txt        # the transcript (--format adds srt/vtt)
    transcript.notes.md / .notes.json   # if you generated notes
    audio.wav                           # only with --record-audio
```

That's it — there is no separate library or index to manage. Browse with
Finder or `ls`, read with anything that opens markdown, delete with `rm`.
Change the standing location with `[output] dir` in settings.toml, or give one
run its own folder with `--out DIR` (files land directly in it; if DIR already
holds a transcript, stenograf refuses to replace it unless you add `--force`).
Audio is stored only when you passed `--record-audio`; without it a meeting
folder holds text alone.

## Meeting notes (LLM summaries)

Turn any transcript into structured notes — summary, decisions, action items
per owner, open questions — with the LLM of your choice:

```sh
steno notes --last                    # notes for the newest meeting
steno notes ~/Documents/Meetings/meeting-20260710-091500
steno notes path/to/transcript.json   # …or any transcript file
steno start --notes                   # generate notes right after the meeting
```

Notes land as sibling `transcript.notes.md`/`.notes.json` files. On Apple
Silicon the default backend is `mlx` — a fully local in-process model, nothing
to set up. To use a different backend, configure it once in
`~/Library/Application Support/stenograf/settings.toml`:

```toml
[notes]
backend = "ollama"          # fully local via `ollama serve`
model = "qwen3:8b"
```

or drive any CLI you already have (prompt on stdin, JSON out):

```toml
[notes]
backend = "command"
command = ["claude", "-p"]

[notes.export]
dir = "~/Documents/Obsidian/Meetings"   # optional: also write one combined
                                        # "YYYY-MM-DD – Title.md" note here
```

With `[notes.export]` set, every summarized meeting also produces a single
self-contained markdown note (frontmatter, summary, action items, collapsible
transcript) — drop the dir inside an Obsidian vault and meetings file
themselves. Three more levers in `[notes]`: `instructions = "~/style.md"`
appends your house style to the built-in prompt, `thinking = false` skips the
mlx model's reasoning pass (faster, less careful), and `auto = true` makes
notes the default for every meeting — `steno start` summarizes without
`--notes`, and the launcher's "Generate notes after the meeting" switch starts
on. Out of the box notes never run unless you ask (`--notes`, the launcher
switch, or `steno notes` afterwards); with `auto = true`, `--no-notes` still
skips them for one run. A notes failure never touches the transcript.

## Naming speakers across meetings

Enroll a voice once and every later meeting relabels that speaker automatically
(cross-meeting re-identification):

```sh
steno profiles enroll Daniel daniel-sample.wav   # a short clip of one speaker
steno profiles enroll Daniel more.wav --reinforce  # fold in another sample
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

Domain terms and attendee names steer the transcript twice: they boost the
decoder toward those spellings *while* it transcribes, and then correct the
near-misses it still got wrong in the finalized text.

```sh
steno transcribe rec.mov --attendee "Anja Müller" --glossary Kubernetes,gRPC
steno transcribe rec.mov --glossary-file terms.txt
```

Write a term the way it appears in a sentence. A capitalized term (`Kubernetes`,
`iOS`) is imposed verbatim; an all-lowercase one asserts the spelling but leaves
the model's capitalization alone. In the correction pass a term and its
transcription must share a word count — `gRPC` can fix `G R P C` spoken as one
word, but not a term split across word boundaries.

Both layers are on whenever there are terms. `[asr] boost` scales the decoder
biasing (default 1.0; `0` turns it off, and much above ~3 it starts rewriting
words that were never in your list), and `[vocab] glossary_threshold` is the
similarity a word must reach before the correction pass touches it (default
0.95 — deliberately strict, because a loose threshold corrupts words the model
already had right).

## Settings

Standing preferences live in `settings.toml` in the platform data dir
(`~/Library/Application Support/stenograf/` on macOS) so you stop re-typing
them. A flag always beats an environment variable (`STENOGRAF_ASR_BACKEND`,
`STENOGRAF_NOTES_BACKEND`, …), which beats the file, which beats the built-in
default — `steno settings show` labels where every value came from.

```sh
steno settings show   # effective configuration + where each value comes from
steno settings edit   # open in $EDITOR (template on first run), validate on save
```

The first `steno settings edit` writes a fully commented template documenting
every key. All keys are optional; the ones you're most likely to want:

```toml
[transcript]
formats = ["md", "json", "txt"]   # default --format list (srt/vtt for subtitles)

[vocab]                           # standing vocabulary — MERGED with per-run
glossary_file = "~/steno/glossary.txt"     # --glossary/--attendee flags
attendees = ["Anja Müller"]
glossary_threshold = 0.95

[output]
dir = "~/Documents/Meetings"      # where meeting folders are created

[speakers]
diarization = true                # separate speakers within a channel (off by
                                  # default; a per-run flag or count also enables)
reid_threshold = 0.5              # cross-meeting voice match strictness (0–1)
profile_store = "~/steno/profiles.json"

[asr]
backend = "parakeet"
provider = "cpu"                  # ONNX execution provider for parakeet-onnx:
                                  # cpu | dml (DX12 GPUs, Windows) | cuda | auto

[notes]                           # see "Meeting notes" above
backend = "mlx"
```

Typos fail loudly: an unknown table or key is an error, and `steno doctor`
validates the whole file.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13 (3.14 is not
yet supported by the ASR stack).

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
