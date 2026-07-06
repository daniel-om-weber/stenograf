# Local Meeting Transcription Tool — Architecture Plan

Accuracy-first, fully local meeting transcription for German and English (one language
per meeting), 2–8 speakers, with optional speaker labels. Primary target: MacBook Pro
M4 Max; Linux portability planned as a later phase. A core design goal is an
**in-memory-only mode**: audio is never written to disk, only the transcript is.

*Based on deep research into the mid-2026 state of the art (four parallel research
tracks: ASR models, diarization, macOS audio capture, existing tools). Sources at the
bottom.*

---

## 1. Key research findings

### ASR: the German field has moved past Whisper

German WER on the multilingual Open ASR Leaderboard (FLEURS + MLS, read speech):

| Model | German WER | Size | License | Apple Silicon path |
|---|---|---|---|---|
| Voxtral Small 24B (Mistral) | **3.01%** | 24B | Apache-2.0 | mlx-voxtral / LM Studio (~14 GB at 4-bit) |
| NVIDIA Canary-1B-v2 | 4.10% | 1B | CC-BY-4.0 | none with word timestamps (NeMo/MPS only — see §2) |
| Qwen3-ASR-1.7B | 4.12% | 1.7B | Apache-2.0 | mlx-qwen3-asr |
| NVIDIA Parakeet-TDT-0.6B-v3 | 4.20% | 0.6B | CC-BY-4.0 | parakeet-mlx (~24× RT), FluidAudio CoreML (~110× RT) |
| Whisper large-v3 | 4.26% | 1.55B | MIT | mlx-whisper, whisper.cpp, WhisperKit |

Beyond raw WER, the CTC/RNNT-style models (Parakeet, Canary) have two structural
advantages over Whisper for meetings:
- **No hallucination on silence** — Whisper's autoregressive decoder invents text
  ("Vielen Dank." / "thanks for watching") during pauses; Parakeet/Canary don't.
- **Native word-level timestamps** — Whisper needs a separate forced-alignment step
  (WhisperX/wav2vec2) to get word timestamps good enough for speaker alignment.

Caveat: these numbers are read-speech benchmarks. Real meeting audio (far-field,
overlap, disfluencies) runs several points higher for every model, and there is no
public German *meeting* leaderboard — **the top candidates must be validated on our
own meeting recordings before committing** (Phase 0).

Streaming costs roughly +1.5–5 WER points vs batch across all models. Purpose-built
streaming options: Voxtral Mini 4B Realtime (<500 ms, German supported, Apache-2.0,
MLX support), Qwen3-ASR streaming mode, Lightning-SimulWhisper (CoreML+MLX,
large-v3-turbo real-time even on M2).

### Diarization

- **pyannote `speaker-diarization-community-1`** (CC-BY-4.0) is the open-source
  accuracy leader (AMI 17.0% DER, VoxConverse 11.2%). Raw pyannote is slow on Mac
  (poor MPS support), but native ports match its accuracy at absurd speed:
  - **speakrs** (Rust/CoreML, Apache-2.0): full community-1 pipeline, 7.1% DER on
    VoxConverse at **529× realtime** on M4 Pro. Mono 16 kHz f32 in → RTTM out.
    *(Phase 1 correction: speakrs is a Rust library only — no CLI, no prebuilt
    binaries. Using it means writing and building our own small Rust wrapper.)*
  - **FluidAudio** (Swift/CoreML, Apache-2.0): community-1 offline + LS-EEND streaming
    (up to 10 speakers) + Silero VAD + speaker embeddings + Parakeet ASR in one SDK.
- **Passing the known speaker count is the single biggest accuracy win**
  (representative: 7.9% DER with known count vs 16–22% without). The UI should ask.
- Realistic expectations: ~90–95% speaker attribution for 2–3 speakers, 80–88% for
  4–6, degrading further at 7–8. Overlapping speech stays lossy.
- **Cross-meeting speaker re-ID** ("this voice = Daniel") is feasible: average
  ECAPA-TDNN or CAM++ embeddings per cluster, cosine-match (~0.5 threshold) against a
  saved profile library; unmatched clusters become nameable new profiles.
- NVIDIA Streaming Sortformer is the streaming-quality leader but is CUDA-only and
  caps at 4 speakers — not usable natively on Mac (only via FluidAudio's CoreML port).

### macOS capture — fully in-memory

- **Core Audio process taps** (`AudioHardwareCreateProcessTap`, macOS 14.4+) are the
  right primitive: audio-only capture of the whole system or specific apps (just
  Zoom/Teams/browser), **no screen-recording permission, no menu-bar recording
  indicator** — only a dedicated "System Audio Recording" TCC prompt
  (`NSAudioCaptureUsageDescription`). Requires a **code-signed** helper (unsigned
  binaries silently get no prompt/no audio).
- ScreenCaptureKit can also capture audio but requires the scarier Screen Recording
  permission, shows the recording indicator, and fights you in audio-only mode.
  Virtual drivers (BlackHole) are a documented fallback for macOS < 14.4 only.
- **Keep mic and system audio as two separate streams end-to-end.** System audio =
  remote participants, mic = local user — this gives perfect local/remote speaker
  separation for free, and diarization only has to split remote speakers within the
  system channel.
- Echo: with headphones there is none. On speakers, apply macOS Voice Processing IO
  (AEC) on the mic path only (gotchas: emits 9 channels — extract one; disable
  auto-ducking via `voiceProcessingOtherAudioDuckingConfiguration`).
- Proven bridge pattern: Swift helper streams raw PCM chunks (~100–200 ms) over
  stdout/Unix socket → Python reads into NumPy → feeds the model directly. Reference
  implementations: **AudioTee** (tap → stdout streaming CLI), **insidegui/AudioCap**
  (canonical tap sample code).
- In-memory RAM math: 16 kHz mono int16 ≈ **115 MB/hour/channel** (float32: 230 MB).
  A 3-hour meeting on two channels ≈ 0.7–1.4 GB — trivial on an M4 Max. Nothing ever
  needs to touch disk.

### Existing tools — what to learn, where we win

| Project | Takeaway |
|---|---|
| Meetily (18k★, MIT, active) | Closest existing tool (Tauri, mic+system capture, whisper.cpp/Parakeet, Ollama summaries). **Has no diarization** — our main gap to beat. Mixes channels; we keep them separate. |
| Vibe | Reference for export formats (TXT/SRT/VTT/JSON/DOCX) and local HTTP API. |
| WhisperX | Canonical batch pipeline shape: VAD-segment → batch ASR → align → diarize → merge. |
| Lightning-SimulWhisper | Apple-native streaming engine (CoreML encoder + MLX decoder, AlignAtt policy). |
| UFAL SimulStreaming | LocalAgreement/AlignAtt commit policies — the academic basis for stable live captions. |
| noScribe | Accuracy-over-speed batch UX for researchers. |
| Granola (commercial) | UX lesson: no bot joins the call; user's rough notes + local LLM enhancement beats generic auto-summary. |

---

## 2. Architecture

Two-pass ("live + finalize") pipeline. The live pass gives immediate captions; the
finalize pass re-transcribes the full retained in-RAM audio with the most accurate
settings when the meeting ends. Every accuracy-critical decision lives in the
finalize pass, so live-pass compromises don't matter for the final transcript.

```
┌────────────────────────── Swift capture helper (signed) ──────────────────────────┐
│  Core Audio process tap (system audio)          Mic (AVAudioEngine, opt. AEC)     │
│         │ mono 16 kHz PCM, ~200 ms chunks              │                          │
└─────────┴──────────────── Unix socket, 2 channels ─────┴──────────────────────────┘
                                      │
┌──────────────────────────── Python core process ──────────────────────────────────┐
│  per-channel ring buffer  +  full-session in-RAM PCM store (int16, ~115 MB/h/ch)  │
│                                      │                                            │
│  LIVE PASS (optional view)           │  FINALIZE PASS (on stop — the real output) │
│  Silero VAD → streaming ASR          │  Silero VAD segmentation (~30 s windows)   │
│  → LocalAgreement commit             │  → batch ASR, language forced              │
│  → live captions                     │  → word timestamps (native or aligned)     │
│                                      │  → diarization on system channel           │
│                                      │    (community-1 via speakrs/FluidAudio,    │
│                                      │     num_speakers = N−1 remote)             │
│                                      │  → word↔speaker merge; mic channel = user  │
│                                      │  → optional speaker re-ID vs saved         │
│                                      │    embedding profiles                      │
│                                      ▼                                            │
│         transcript store (Markdown/JSON/SRT/VTT) — the ONLY thing written to disk │
│         optional: local LLM summary / note enhancement (Ollama)                   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Component decisions

**Capture (macOS):** Swift helper binary, code-signed, `NSAudioCaptureUsageDescription`
+ `NSMicrophoneUsageDescription`. Process tap for system audio (whole-system default,
per-app option), AVAudioEngine for mic with AEC toggled by output device
(speakers → on, headphones → off). Downmix to mono 16 kHz, stream length-prefixed
chunks over a Unix socket. Model on AudioTee/AudioCap. Fallback for macOS < 14.4:
BlackHole, documented as degraded.

**Finalize ASR (accuracy-critical):** pluggable backend interface. **Committed
default: Parakeet-TDT-0.6B-v3 via parakeet-mlx** — native word timestamps, no
silence hallucination, ~120× realtime, <1 GB RAM, same model as the live pass.
*(Canary-1B-v2 was the original planned default but was dropped in Phase 0
research, July 2026: no MLX/CoreML runtime emits its word timestamps — the
PyPI `canary-mlx` package is an abandoned template, mlx-audio's Canary port
returns placeholder timestamps, and onnx-asr supports timestamps only for
TDT/CTC/RNNT decoders. The sole working path, NeMo on PyTorch-MPS, is too
slow and heavy to ship; it remains an accuracy-ceiling reference in the eval
harness.)* **Opt-in max-accuracy: Voxtral Small 24B (mlx-voxtral, 4-bit,
~14 GB)** — best German WER (3.01%), slower, text only (no timestamps).
Fallback: Whisper large-v3 (mlx-whisper) + WhisperX-style alignment.

*Phase 0 result (July 2026, blind adjudication of 161 model-disagreement sites
on real meeting audio, de+en):* **Parakeet confirmed as default** — it tied
Whisper large-v3 exactly (42:42 head-to-head) while being ~10× faster and 5×
smaller; Whisper stays as fallback. **Voxtral's read-speech advantage did not
transfer** to meetings (lost 32:38 to Parakeet, 22:28 to Whisper) — demoted
from "opt-in max accuracy" to not-worth-shipping pending new evidence. Canary
was empirically the weakest (lost every pairing ~1:2) on top of having no
viable runtime. Methodology note: full hand-corrected references proved slow
and anchor-biased; the adjudication harness (eval/adjudicate.py) is the
recommended evaluation path going forward. Language is forced per meeting (user setting, or auto-detect once on
the first confident segment, then locked). `initial_prompt`/context seeded from a
user glossary and attendee names where the backend supports it.

**Live ASR (latency-critical, quality secondary):** **committed default:
Parakeet-TDT-0.6B-v3 via parakeet-mlx**, run with a **growing re-decode window**
(everything since the last long silence, capped at ~60–120 s, re-decoded every
1–2 s) and a LocalAgreement commit policy. Parakeet's ~120× realtime makes this
affordable (a 60 s window every 2 s needs only ~30× RT) and it largely removes
the fixed-chunk boundary artifacts that a 10 s window would cause — live quality
approaches finalize quality with the same model. Upgrade path if it still feels
laggy: Voxtral Mini 4B Realtime (true streaming, <500 ms) or Qwen3-ASR-1.7B
streaming. Interim text shown grey; finalize pass replaces the live transcript.

*Phase 2 spike verdict (July 2026, `StreamingParakeet` vs re-decode window on
real de meeting audio, M4 Max): the re-decode window is confirmed as the live
default and parakeet-mlx's incremental streaming API (`transcribe_stream`,
retained encoder/decoder state) is rejected. Measured — the incremental API at
small right-context (the "cheap linear" setting, e.g. `(256,8)`) produces
garbage (80–90% WER, code-switching into English); it is usable only at full
right-context `(256,256)` and even then is fragile (drifts badly below ~3 s feed
chunks) at ~13× RT / 13.7% WER-vs-ref over 300 s. The re-decode window over the
reliable full `generate()` path (~110× RT at any window size) is both more
accurate and simpler: re-decoding a **12–16 s trailing window** — uncommitted
tail + ~4 s left context, prefix-committed so committed audio drops out (NOT the
naive 60–120 s window, which would be ~27% duty) — every ~1–1.5 s costs only
**~7–10% of one accelerator during speech, ~0% in silence with VAD gating**, at
**finalize-grade accuracy (~10% WER, same `generate()` path as finalize)** and a
1–1.5 s caption cadence. LocalAgreement-2 over consecutive window decodes commits
the stable prefix; the last ~2–3 s stays grey. The incremental API and
Voxtral/Qwen streaming remain documented fallbacks only.*

**Diarization:** shipped baseline (July 2026): **sherpa-onnx** (pyannote
segmentation-3.0 + 3D-Speaker eres2net embeddings, ONNX/CPU) — pip-installable on
every platform, takes a known speaker count, and was planned for Linux/Windows
anyway. Embedding-model caveat from validation: sherpa's CAM++ VoxCeleb export
flips cluster identity between segmentation windows (one speaker shredded into
many); eres2net and titanet-small agree with each other and with the audio —
eres2net is the default. The community-1-accuracy upgrade on macOS needs a
wrapper binary we build ourselves, since **speakrs and FluidAudio are both
libraries, not CLIs**: either a small Rust CLI around speakrs or diarization in
the Swift helper via FluidAudio (evaluate when live capture lands; same
``Diarizer`` interface either way).

**Cross-platform accuracy path (no Mac-native models) — documented for later
(research July 2026):** the many-speaker weakness is sherpa's greedy
`FastClustering`, not the models. pyannote's own 3.1→community-1 gain ("marked
reductions in speaker confusion" at higher counts) was *only* a clustering swap
(AHC→VBx) on the *same* segmentation — so community-1-class accuracy is three
swappable ONNX pieces, and only the *runtime* is CoreML in the native ports:
pyannote segmentation-3.0 (have it) + **WeSpeaker ResNet293-LM** embedding (ONNX
in sherpa's zoo, VoxCeleb EER 0.447%, English — vs our current eres2net, the
lower-EER *zh-cn* export) + a ported **VBx** clustering step (the one missing
piece; BUT's `VBx` is the reference to lift). That reproduces speakrs/FluidAudio
in Python/ONNX — CPU everywhere, no PyTorch/CUDA/CoreML — behind the same
`Diarizer` interface. Staged: (a) cheap interim — swap the embedding to
ResNet293-LM (~1 line, strictly better for de/en); (b) least-code way to reach
the ceiling and measure the real gain — run `pyannote.audio` community-1 directly
(PyTorch, CC-BY-4.0, heavy + slow-ish on Mac MPS, but diarization is a small
slice of runtime); (c) the real target — the pure-ONNX VBx rebuild. **DiariZen**
(WavLM+Conformer+VBx, CC-BY-4.0) tops the open leaderboard (~13.3% DER overall,
7.1% at 5+ speakers) but is PyTorch/GPU-oriented with no ONNX export — skip
unless chasing the very top with a GPU. Dead end for our 2–8-speaker case: every
*end-to-end neural* diarizer is hard-capped (NVIDIA Sortformer at 4 speakers,
FluidAudio's LS-EEND streaming at 10) — only the clustering pipelines scale.
Lever order for many speakers: **known count** (done — biggest) > **VBx
clustering** > **better embedding** > the 3-speaker-per-window segmentation cap
(least important; it's a local per-window limit, not a global one).

Diarization is a
**per-channel** operation, configured by a meeting profile set at start
(`--local N --remote M`), covering three configurations:

| Configuration | Mic channel | System channel |
|---|---|---|
| Online (1 local, M remote) | attributed to user directly, no diarization | diarize with `num_speakers = M` |
| Hybrid (N local, M remote) | diarize with `num_speakers = N` | diarize with `num_speakers = M` |
| In-room only (N local, 0 remote) | diarize with `num_speakers = N` | not captured at all |

Speakers are labeled `Local-1..N` / `Remote-1..M` (renameable; re-ID can name them
automatically later). The channel prior still pays off in every mode: diarization
never has to separate local from remote voices — only voices *within* a channel —
and each run gets an exact speaker count, the biggest single accuracy lever.
In-room-only mode skips the system tap entirely (no system-audio permission needed,
single channel). Overlapped regions flagged as provisional in the transcript.

Hybrid-mode caveats: (a) remote audio played through room speakers bleeds into the
mic — AEC (Voice Processing IO) on the mic path is mandatory in hybrid mode, plus a
cross-channel dedup at merge time (near-identical text at the same timestamps on
both channels → keep the system-channel copy); (b) the reverse direction is safe —
meeting apps send only remote voices, so the system channel stays clean. In-room
mode is the acoustically hardest case (far-field mic, 2–8 speakers, more overlap):
transcription and diarization quality depend heavily on the mic — recommend an
external conference/boundary microphone for rooms with 4+ people; live-pass quality
will be noticeably lower and the finalize pass matters even more here.

**Speaker re-ID (optional, later):** per-cluster mean CAM++/ECAPA-TDNN embedding,
cosine-matched against a local profile store; UI lets the user name unmatched
clusters, which enrolls them.

**Meeting configuration & auto-detection:** every meeting parameter follows one
resolution order: **explicit user setting > auto-detected value > safe default** —
all settings optional, unspecified means auto. Auto-detected values are surfaced in
the UI as editable ("Detected: German, 2 remote speakers"), and because the full
audio is retained in RAM and the finalize pass is fast, a wrong detection is never
fatal: correct the value and re-run finalize in seconds.

| Parameter | Auto-detection mechanism | Reliability / phase |
|---|---|---|
| Language (de/en) | **Shipped (Phase 1, `stenograf.lid`):** function-word + umlaut/ß vote over the finalized transcript, locked for the session. Acoustic first-segment LID (sherpa-onnx `SpokenLanguageIdentification`) is the live-pass upgrade — it can lock before any text exists and feed a language-*requiring* backend | High for a de/en binary choice |
| Remote speaker count | community-1's speaker-count estimation on the system channel (run unconstrained, or with bounds 1–8) | Decent; explicit count still more accurate — Phase 1 (it's just "don't pass `num_speakers`") |
| Local speaker count | Same, on the mic channel | Weaker (far-field audio) — Phase 3 |
| Meeting mode (online/hybrid/in-room) | Meeting-app detection (running Zoom/Teams/browser-call process + audio activity on the tap) → remote component exists; multiple voices on mic → local component >1 | Phase 3–4; until then mode falls back to "online" if a meeting app is audible, else "in-room" |
| Participant names/glossary | Calendar-invite integration (attendees, title → re-ID hints + `initial_prompt`) | Phase 4 |

**Whisper-specific accuracy settings** (when a Whisper backend is used):
`vad_filter=True` (Silero), batch only VAD segments (never raw sliding windows),
`condition_on_previous_text=False` always (Phase 0 found the finalize-pass
`True` setting lets decoder loops snowball across windows — up to 220 repeated
words on overlap/silence; consistency isn't worth it), default temperature
fallback ladder with `compression_ratio_threshold≈2.4`, `logprob_threshold≈-1.0`,
`no_speech_threshold≈0.6`, `hallucination_silence_threshold≈2–8 s`, plus a post-filter
blacklist for phantom phrases during silence.

**In-memory guarantee:** the default mode holds audio only in bounded ring buffers +
the session PCM store in RAM; no code path writes audio to disk. (OS-level swap/crash
dumps are outside app control — worth a note in docs, not an app concern.)

**Opt-in audio recording (`--record-audio[=path]`, default off, Phase 1):** when
explicitly enabled, the Python core additionally appends the incoming PCM to a WAV
file as it arrives — mic on the left channel, system audio on the right (mono in
in-room mode), preserving channel separation in a file any player opens. Recorded
at the wire format (mono 16 kHz int16 per channel, ~230 MB/h for both channels):
sufficient for speech playback and exactly what re-transcription needs; native-rate
archival is out of scope (would require teeing in the helper before the resample).
Append-as-you-go with the WAV header patched periodically — crash-safe like the
incremental text checkpoints. Recording state must be loudly visible (CLI banner,
later UI indicator); consent stakes are higher for retained audio than transcripts
(docs note). Recorded files feed back in via a `steno transcribe <file>` batch
entry point (re-run finalize with a better model / corrected settings, and a
source of new eval data for the adjudication harness). Possible later nicety:
FLAC (~50% smaller); not MVP.

**Outputs:** speaker-labeled, timestamped Markdown + JSON (word-level timestamps
retained); SRT/VTT export. Optional post-meeting summary via local LLM (Ollama) —
separable later stage.

### Language/runtime choices & multi-platform layering

Everything is Python except one small native capture helper on macOS. The
platform boundary is drawn at two interfaces, so all platform-specific code is
swappable without touching the core:

1. **Capture provider = separate process speaking a language-neutral wire protocol**
   (Unix socket / stdio: JSON control messages + length-prefixed PCM frames carrying
   channel id, timestamp, mono 16 kHz int16 samples). Providers:
   - **macOS: Swift helper** (~300–600 lines; Core Audio process tap + AVAudioEngine
     mic + Voice Processing AEC + TCC prompts; start from AudioTee/AudioCap code).
     Required because no Python package exposes the tap API (pyobjc doesn't wrap the
     CoreAudio HAL C functions; miniaudio support is an open issue). BlackHole would
     be the pure-Python route and stays a documented fallback only.
   - **Linux: pure Python** (`sounddevice` reading PipeWire/PulseAudio monitor
     sources — system capture is easier there than on macOS).
   - **Windows: pure Python** (WASAPI loopback via `soundcard`/`pyaudiowpatch`).
   On Linux/Windows the provider may run in-process behind the same interface.
2. **Inference backends = Python ABCs.** ASR: MLX backends on Mac ↔
   CTranslate2/ONNX/CUDA backends on Linux/Windows (same models; Parakeet/Canary
   have ONNX paths). Diarization: sherpa-onnx (ONNX/CPU) everywhere today; a
   macOS-native community-1 wrapper (speakrs or FluidAudio, CoreML) can slot in
   behind the same interface later, and a pure-ONNX community-1 rebuild
   (seg-3.0 + WeSpeaker ResNet293-LM + ported VBx) is the cross-platform accuracy
   path — same interface, CPU everywhere (see §2 Diarization). VAD (Silero) is
   ONNX/CPU everywhere.

The Python core (ring buffers, session PCM store, VAD, live pass, finalize pass,
diarization orchestration, merge, transcript store, CLI/TUI) is identical on all
platforms. Swift is the only language we write besides Python; Rust appears only as
the prebuilt speakrs binary we invoke.

UI: start as a CLI/TUI (`steno start --lang de --local 3 --remote 2`); a menu-bar or
Tauri app is a later layer, not an architectural change.

### Deployment & distribution

Swift is a **build-time** dependency only — users never need a compiler. Standard
uv-based distribution works on all platforms:

- **Packaging:** normal `pyproject.toml` package; CI builds platform wheels. The
  `macosx_arm64` wheel bundles the compiled Swift capture helper + speakrs binary
  (built on a GitHub Actions macOS runner via a hatchling build hook — same pattern
  as ruff/uv shipping Rust binaries in wheels). Linux/Windows wheels are pure
  Python. The core locates bundled binaries via `importlib.resources` and spawns
  them as subprocesses.
- **Install UX:** `uv tool install stenograf` or zero-install `uvx stenograf ...`;
  plain `uv run` in the repo for development (dev-mode fallback compiles the
  helper locally when Xcode CLT is present).
- **Model weights** (several GB) are not in the wheel — pulled from HuggingFace into
  a local cache on first run.
- **macOS signing/permissions — no Apple Developer account needed.** Free ad-hoc
  signing (`codesign -s -`) is all the helper requires; permission prompts
  attribute to the responsible process (the terminal), so the user grants mic +
  system-audio to their terminal app once (how AudioTee ships via Homebrew) —
  **verified in the Phase 1 spike (July 2026, `native/spike/`)**: an ad-hoc-signed
  CLI with usage strings embedded via an `__info_plist` section captured non-silent
  system audio through a whole-system process tap (mono 48 kHz float32) and mic
  audio via AVAudioEngine on macOS 26.5. Developer ID + notarization ($99/yr) is needed *only* to distribute a
  downloadable .app bundle to other people (Gatekeeper checks the browser-set
  quarantine flag; uv/pip installs and locally built apps never have it). Decision:
  stay CLI-launched; no Developer ID.
- **UI direction: local web UI** served by the Python process on localhost (live
  captions with speaker colors, meeting archive, click-to-jump transcript) — as
  polished as desired, zero signing, same UI on Linux later. Textual TUI as the
  lighter in-terminal alternative. A native .app stays optional and personal-use
  ad-hoc-signed if ever wanted.
- **Distribution channels — PyPI + uv only** (side-project scope). Colleagues:
  install uv (one curl command), then `uv tool install stenograf`;
  pre-release channel: `uv tool install git+<repo>` or wheels on GitHub Releases.
  Rejected as effort/fit-negative: Homebrew (Python ML deps must be vendored into
  the formula — huge and brittle), deb/rpm/AUR/Flatpak/snap and winget/MSI
  (per-platform maintenance matrix, technical users don't need it), Docker (no
  host audio devices or MLX GPU access in containers on macOS).
- **Invest in first-run UX instead:** a `steno setup`/`steno doctor` command —
  checks macOS ≥ 14.4, triggers permission prompts, downloads models with
  progress, walks through the one-time HF token for gated pyannote weights
  (CC-BY-4.0 — investigate mirroring them to remove that step).

---

## 3. Locked decisions (July 2026)

- **Phase 0 data:** usable real meeting recordings exist; evaluation runs on those.
- **Crash policy:** periodic incremental finalization — every few minutes the
  finalize pipeline runs over the completed portion and checkpoints the *text* to
  disk. A crash loses at most the last few minutes of audio; audio itself is never
  persisted (this replaces Meetily-style audio checkpoints). *Revised for Phase 2
  (Option B, July 2026): once the live pass runs, its LocalAgreement-committed
  transcript is itself the checkpoint — flushed to `<meeting>.partial` as
  zero-inference file I/O, no separate periodic finalize pipeline (which would
  double GPU work to reproduce text the live pass already has). The heavy finalize
  runs only on stop. A crash recovers the on-screen (live-quality) text; only the
  finalize-grade refinement of the crashed tail is lost. `--no-live` falls back to
  a tail-only finalize checkpoint (off the consume thread — fixes the O(n²)
  whole-buffer re-finalize).*
- **Repo & license:** public + MIT from day one.
- **Distribution:** PyPI + uv only; no Developer ID; local web UI direction.
- **Name: `stenograf`** — German spelling of stenographer, the verbatim
  minute-writer. Package `stenograf` (confirmed free on PyPI), CLI `stenograf`
  with `steno` alias.

## 4. Fork vs. build decision

Source-level analysis of the closest existing tools (July 2026):

**Meetily** (MIT, 18k★, active) is the only realistic fork candidate — live capture,
Rust/Tauri, modular ASR engines (whisper.cpp + Parakeet), an existing two-pass
concept. But it structurally violates our three differentiating requirements:
- **Audio hits disk by design**: `IncrementalAudioSaver` checkpoints mixed PCM to
  disk every 30 s during the meeting, a final MP4 is written on stop, and the
  accuracy "retranscription" pass works by *re-decoding the on-disk file*
  (`retranscription.rs` → `decode_audio_file`). The in-memory requirement means
  rewriting the pipeline's spine, not flipping a flag.
- **Channels are mixed to mono** in `AudioMixerRingBuffer` before transcription
  ("we only store mixed audio") — our channel-separated diarization design is
  incompatible with the mixer at the core of their pipeline.
- **No diarization** in the OSS core (PRO roadmap item).
Its ASR engines are Rust-side whisper.cpp/ONNX servers; our MLX-based German models
(Canary/Voxtral) don't plug into that cleanly either.

**Vibe** is a file-based batch app (wrong paradigm; useful as a diarization
reference). **Hyprnote** (fastrepl) is product-shaped like our target but its audio
path is unverified — worth a one-hour source dive before Phase 1. noScribe/Scriberr
are file-based; WhisperX/FluidAudio are libraries, not tools.

**Decision: build fresh, but reuse components rather than codebases.** Everything we
would keep from a Meetily fork (capture patterns) is available as smaller, cleaner
pieces (AudioTee/AudioCap for taps, speakrs for diarization, parakeet-mlx/MLX for
ASR, Silero VAD); everything we would fight (disk-centric finalize, mono mixer,
Rust ASR servers, large Tauri/Next.js/FastAPI surface) is their architectural core.
Our new code is mostly orchestration glue. Revisit only if a polished GUI product
quickly becomes the priority over the accuracy/in-memory core.

## 5. Phased roadmap

**Phase 0 — Model validation (before building anything).**
Record 2–3 real meetings' worth of representative audio (German + English, with
consent), hand-correct reference transcripts for a few segments, and benchmark
Canary-1B-v2 vs Voxtral Small 24B vs Whisper large-v3 (and Parakeet-v3) for WER +
speed on the M4 Max. Read-speech leaderboards don't settle this; 1–2 days of eval
harness work de-risks the whole project.

**Phase 1 — Batch MVP (the accuracy core).**
Swift capture helper (tap + mic → socket) → Python core that buffers PCM in RAM and,
on stop, runs the finalize pass: VAD → best ASR backend → diarization → merged
speaker-labeled Markdown/JSON transcript. Includes the opt-in `--record-audio` WAV
tee and the `steno transcribe <file>` batch entry point (also the finalize pass's
dev/test harness). No live view yet. This alone is a usable, legally-clean meeting
transcriber.
*Status (July 2026): capture spike verified (`native/spike/`). Finalize pipeline +
`steno transcribe` shipped and validated on the Phase 0 eval audio (Silero VAD →
parakeet-mlx → sherpa-onnx diarization → merged transcript, ~8–14× realtime
end-to-end on M4 Max). Meeting orchestrator shipped (`steno start`): `SessionStore`
(per-channel in-RAM PCM, timestamp-aligned, never disk) → per-channel finalize with
each channel's known count → interleaved `Local-N`/`Remote-N` transcript, behind the
swappable `CaptureProvider` interface. Also shipped: the opt-in `--record-audio` WAV
tee (streaming, crash-safe, mic-left/system-right) and incremental text checkpointing
(`--checkpoint-interval`, writes `<meeting>.partial` every N s of capture, cleaned up
on clean stop). A `FileCaptureProvider` (`--replay mic[,system]`) also drives the whole
orchestrator over recorded files for dev/test. The production Swift capture
helper (`native/helper/`, **stenocap**) is shipped: Core Audio process tap
(system) + AVAudioEngine (mic, optional `--aec`) → AVAudioConverter to mono
16 kHz int16 → framed PCM on stdout, clean SIGINT/SIGTERM stop; consumed by
`MacOSCaptureProvider` behind the same `CaptureProvider` interface. Verified
end-to-end (July 2026): live mic capture is non-silent and real-time; German
speech played to the system output is captured through the tap and transcribed
accurately (`steno start --local 0 --remote 1`). Automatic de/en language
detection ships as a text vote over the finalized transcript (`stenograf.lid`),
auto-filling the transcript language and locking it for the session. **Phase 1
is complete** — a usable, legally-clean meeting transcriber. Deferred to later
phases: hybrid-mode cross-channel dedup (needs the helper's AEC to matter in
practice), moving checkpoint finalize off the consume thread (needs real-time
backpressure tuning), and acoustic first-segment LID for the live pass.*

**Phase 2 — Live captions.**
Streaming ASR pass with LocalAgreement commits, TUI live view; finalize pass replaces
the live transcript on stop. *Live-ASR mechanism locked by the Phase 2 spike
(§2 Live ASR): a 12–16 s re-decode window over the full `generate()` path (~7–10%
of one accelerator during speech, ~10% WER), VAD-gated, LocalAgreement-2 commit —
not parakeet-mlx's incremental streaming API, which the spike measured as garbage
at small right-context and fragile otherwise. Checkpointing revised to Option B:
the committed live transcript is itself the crash checkpoint (flushed to
`.partial` as zero-inference file I/O), superseding §3's periodic re-finalize;
the heavy finalize runs only on stop, with a single-flight inference worker so
live ASR and finalize never contend for the one accelerator.*

**Phase 3 — Speaker polish + vocabulary + auto-detection.**
Speaker re-ID with embedding profiles ("Daniel" across meetings), user glossary /
attendee-name prompting, overlap flagging, export formats, config for per-app taps,
local-speaker-count estimation and meeting-mode auto-detection (language and
remote-count auto-detection ship earlier, in Phase 1).

**Phase 4 — Product layer + Linux.**
Local web UI (live captions, meeting archive, click-to-jump transcript), optional
Ollama note-enhancement, Linux capture backend + ONNX/CTranslate2 inference
backends.

### Phase 2 build plan — live captions (start here)

Mechanism and checkpointing are locked by the Phase 2 spike (§2 Live ASR, §3 crash
policy): a **12–16 s re-decode window over the full `generate()` path**, VAD-gated,
LocalAgreement-2 commit; **Option B** checkpointing (committed live text is the
`.partial` checkpoint, heavy finalize only on stop); a **single-flight inference
worker** so live ASR and finalize never contend for the one accelerator.

**Live-pass evaluation — no hand-corrected ground truth needed.** The live pass is
provisional text that finalize replaces on stop, so its reference is the finalize
pass's own full-attention `generate()` output on the same audio, *not* a human
transcript (we only have one, `de-1`, and are not extending it). Three label-free
metrics, runnable on any raw `examples/*.mov` (hours of real de/en; use long
continuous stretches to stress length-stability, the property that broke the
incremental streaming API):
1. **Agreement with finalize** — WER of the committed-live transcript vs
   full-`generate()` on the same audio (the live-degradation number).
2. **Commit monotonicity** — a committed (black) word must never be contradicted by
   a later decode; any violation is a bug.
3. **Commit latency** — audio-arrival → commit time.
Correlated live/finalize errors are acceptable: if live matches finalize, the live
view matches the authoritative transcript, which is the whole UX contract. Absolute
accuracy is finalize's concern, characterized once (`de-1`, 10.3% WER).

**Task sequence** (independent, testable increments; interface names illustrative):
1. **`LiveDecoder`** — re-decode window + LocalAgreement-2, *composing the existing*
   `ASRBackend.transcribe`/`generate` (no new dependency, no `StreamingParakeet`).
   `feed(samples, t_offset) -> StreamingUpdate(committed, interim)`; `flush()`
   (force-commit tail at utterance end); `reset()` (drop window at long silence).
   Window = uncommitted tail + ~4 s left context, capped ~12–16 s, VAD-anchored
   (reuse `SileroVAD`); commit the LocalAgreement-2 stable prefix, keep the last
   ~2–3 s grey. **Acceptance = the three label-free metrics above on two `examples/`
   clips.** This is the starting point and de-risks the rest.
   *Status (July 2026): shipped (`stenograf.live.LiveDecoder` +
   `tests/test_live.py`, 13 tests). Re-decode window over the full `generate()`
   path, LocalAgreement-2 with a grey-zone commit horizon, Silero VAD gating
   (~0 decodes in silence) + endpoint-silence utterance finalize, append-only
   monotonic commit guard, and an ordered overflow-flush that bounds the window
   without ever dropping un-transcribed audio (the spike's window-cap "safety
   valve" is unnecessary — parakeet commits steadily). Acceptance harness
   `eval/live.py` (drives the decoder in simulated real time vs a full
   `finalize_channel` reference). Measured on de-1 + en-1 (300 s each, feed
   cadence 1 s): agreement WER 7.0% / 5.6% (better than the ~10% spike target),
   0 monotonicity violations, commit latency median ~2.5 s / p90 ~3.5 s. Params:
   `left_context=4 s`, `window_cap=15 s`, `grey_zone=2 s`, `endpoint_silence=0.6 s`.*
2. **`SessionStore` thread-safety** — add `_offsets` + a lock; new
   `view(channel, start_s, end_s)` returning O(window) float32 (the append-only
   chunk list is prefix-immortal → snapshot `len(chunks)` under the lock, concat
   outside it). Also kills the O(n²) whole-buffer re-finalize.
   *Status (July 2026): shipped (`stenograf.session.SessionStore`). `append`
   publishes each frame's chunks in one short critical section so a reader never
   sees `chunks`/`offsets`/`length` disagree; `view(channel, start_s, end_s=None)`
   bisects the per-chunk `_offsets` to the covering chunks, snapshots those
   references under the lock, and concatenates + slices outside it — O(window),
   never O(buffer). `samples`/`duration` take the same lock. `end_s` defaults to
   the current tail; out-of-range/inverted windows clamp to empty. Tests added to
   `tests/test_session.py` (view correctness incl. across a silence gap, clamping,
   and a single-writer/many-reader concurrency stress asserting every observed
   prefix matches exactly). This is the primitive the Task 3 `LiveWorker` feeds
   the `LiveDecoder` from; rewiring the checkpoint path to a tail-only finalize
   (the actual O(n²) removal) lands with the Task 3–4 orchestration refactor —
   `MeetingRecorder`'s current checkpoint path is untouched here.*
3. **Orchestration refactor** (`session.py`) — `AudioBus` (per-channel watermark +
   `Condition`, event-driven, no polling), `CaptureLoop` thread (never blocks on
   inference, never drops audio), `LiveWorker` (the *single* inference thread →
   single-flight; `inference_lock` as the belt-and-suspenders extension point).
   Reconcile-to-watermark backpressure. Stop → join worker → authoritative
   `finalize()`. Add real-time pacing to `FileCaptureProvider` so `--replay`
   exercises the live pass at meeting cadence.
   *Status (July 2026): shipped. `AudioBus`/`CaptureLoop`/`LiveWorker` in
   `session.py`; `MeetingRecorder.run(live=True, on_update=…)` runs capture on its
   own thread feeding one worker that drives a `LiveDecoder` per channel off
   `store.view` (O(window)) and streams `StreamingUpdate`s to `on_update`. The
   worker reconciles to the latest watermark each wake — a backlog collapses into
   one catch-up decode, and no audio is lost because it lives in the store; on
   close it feeds the final window and flushes. Stop joins the worker, then the
   single-flight `finalize()` (under `inference_lock`) replaces the live
   transcript. `FileCaptureProvider(paced=True)` releases frames at wall-clock
   time. The batch path (`--no-live`, periodic re-finalize checkpoint) is
   unchanged; CLI `--live` wiring is Task 7, checkpoint Option B is Task 4.
   **Bug found by end-to-end verification (not unit tests, which use a fake ASR):
   MLX GPU streams are thread-local and its weights are lazy, so parakeet's first
   decode on the worker thread died with "no Stream(gpu, 0) in current thread";
   fixed by materializing the weights on the load thread — `mx.eval(model.
   parameters())` in `ParakeetMLXBackend.load()`** (regression-guarded in
   `tests/test_asr_parakeet.py`). Tests: `tests/test_live_orchestration.py`
   (AudioBus semantics, backlog reconcile, capture+worker cover-all-audio,
   `run(live=True)` streams commits + finalizes, max-seconds) and paced/unpaced
   replay in `tests/test_capture_file.py`. Validated live on de-1 via paced replay
   through the real parakeet worker: captions stream in real time, closely match
   the finalize output, and German is auto-detected on stop.*
4. **Checkpoint Option B** — replace the periodic re-finalize with a committed-text
   `.partial` flush (pure I/O, coalesced ~10–20 s); `--no-live` falls back to a
   tail-only finalize. Keep `_cleanup_checkpoints` on clean stop.
   *Status (July 2026): shipped. Both modes checkpoint via the same
   `on_checkpoint(Transcript)` callback the CLI already writes to `.partial`,
   coalesced to `checkpoint_interval` seconds of capture, but never running any
   inference the mode does not already do. **Live:** the `LiveWorker` flushes the
   decoders' already-committed words as-is (`MeetingRecorder._live_checkpoint`,
   `pipeline.group_words` → channel-coarse `Local`/`Remote` entries) — zero
   inference, on the same thread that owns the decoders, empty flushes skipped so
   a `.partial` only appears once there is text. **Batch (`--no-live`):** a new
   `_TailCheckpointer` thread waits on the `AudioBus` and finalizes only the newest
   tail each interval (`store.view` → `finalize_channel` with `diarizer=None`,
   times shifted, coarse label), off the capture thread and each second exactly
   once — killing the old O(n²) whole-buffer re-finalize. Both are superseded on
   clean stop by the diarized full finalize, which also owns language locking.
   Tests: `group_words` (`test_pipeline.py`); `_live_checkpoint`/`_tail_entries`,
   `_TailCheckpointer` exactly-once (recording-ASR sum == total), batch checkpoints
   accumulate + coarse (`test_session.py`); `LiveWorker` interval flush + no-flush
   (`test_live_orchestration.py`). Verified end-to-end on de-1 with the **real
   parakeet backend** (the unit tests use a fake ASR): batch tail finalize runs on
   the `tail-checkpoint` thread with no MLX thread-stream error and cleans up its
   `.partial` on clean stop; paced-replay live run flushes 3 coarse-labelled
   `.partial`s whose committed text closely tracks the finalize output (the whole
   Option-B premise), then the finalize swaps in `Local-1`. See
   [[mlx-weights-thread-local-streams]]. Deferred to Task 7: `--live`/`--plain`
   CLI wiring, the `--flush-interval` alias, and tuning the default interval down
   from 180 s.*
5. **`LiveView` + `PlainLiveView`** — the event interface (`interim`, `commit`,
   `status`, `language`, `finalizing`, `finalized`, `error`) + a non-TTY/`--plain`
   impl streaming committed text via `click.echo`. **First shippable milestone:
   live captions in plain stdout, no Textual dependency.**
6. **Textual TUI** (`TextualLiveView`) — pinned header (REC/elapsed/language/
   profile), append-only `RichLog` of committed captions, dim per-channel interim
   tail (`You`/`Remote` — channel-coarse; real `Local-N`/`Remote-M` only after the
   finalize swap), footer. Minimal-redraw discipline: one 1 Hz clock is the only
   periodic repaint, animations off, `MAX_FPS≈15`; worker→UI via
   `loop.call_soon_threadsafe`. **Ctrl-C is a captured key event under Textual (not
   `KeyboardInterrupt`)** — the quit binding must cross to the worker via
   `provider.stop()`; wire it deliberately.
7. **Glue** — `steno start` gains `--live/--no-live`, `--plain`, `--flush-interval`
   (alias `--checkpoint-interval`); doctor/README; a CPU-proxy regression test (zero
   window decodes during silence; committed text never rewritten).

CPU budget target (spike-measured): **~7–10% of one accelerator during speech, ~0%
in silence**, live captions ~10% WER, ~1.5 s cadence.

---

## 6. Key sources

- Open ASR Leaderboard multilingual paper: https://arxiv.org/html/2510.06961v4
- Canary-1B-v2 / Parakeet-TDT-0.6B-v3: https://huggingface.co/nvidia/canary-1b-v2 · https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · https://arxiv.org/html/2509.14128v2
- Voxtral: https://mistral.ai/news/voxtral/ · Realtime: https://arxiv.org/html/2602.11298v2 · https://pypi.org/project/mlx-voxtral/
- Qwen3-ASR: https://arxiv.org/html/2601.21337v2 · https://github.com/moona3k/mlx-qwen3-asr
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
- speakrs: https://github.com/avencera/speakrs · senko: https://github.com/narcotic-sh/senko · FluidAudio: https://github.com/FluidInference/FluidAudio
- Core Audio taps: https://developer.apple.com/documentation/CoreAudio/capturing-system-audio-with-core-audio-taps · AudioCap: https://github.com/insidegui/AudioCap · AudioTee: https://stronglytyped.uk/articles/audiotee-capture-system-audio-output-macos
- Meetily: https://github.com/Zackriya-Solutions/meetily · Vibe: https://github.com/thewh1teagle/vibe
- WhisperX: https://github.com/m-bain/whisperX · Lightning-SimulWhisper: https://github.com/altalt-org/Lightning-SimulWhisper · SimulStreaming: https://github.com/ufal/SimulStreaming
- Whisper anti-hallucination: https://github.com/openai/whisper/discussions/679 · https://arxiv.org/html/2505.12969v1
- Apple Silicon Whisper benchmarks: https://github.com/anvanvan/mac-whisper-speedtest · https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html
