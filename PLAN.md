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

**Diarization:** two backends behind one ``Diarizer`` interface (shipped July
2026, commit a3ebff8):

- **sherpa-onnx** (pyannote segmentation-3.0 + 3D-Speaker eres2net embeddings,
  ONNX/CPU) — pip-installable on every platform, handles every run with a
  **known** speaker count. Embedding-model caveat from validation: sherpa's
  CAM++ VoxCeleb export flips cluster identity between segmentation windows
  (one speaker shredded into many); eres2net and titanet-small agree with each
  other and with the audio — eres2net is the default. Its greedy
  `FastClustering` cannot *estimate* a count: measured on the five eval
  segments it found 13/25/9/13/16 "speakers" where the true counts are 2–5 —
  no cosine threshold is robust across meetings (structural, don't re-tune).
- **stenodiar** (`native/stenodiar/`, optional) — a small Rust CLI around
  **speakrs**, which reimplements the full pyannote community-1 pipeline
  (segmentation → powerset → embedding → PLDA → **VBx clustering**) with
  native CoreML; VBx is what makes *automatic* count estimation trustworthy.
  Same segments: 3/5/2/3/3 speakers, coherent turn-taking, ~450× realtime
  warm (first run per machine downloads models from the ungated HF mirror
  `avencera/speakrs-models` and compiles CoreML — minutes; `--warmup`).
  Audio is piped as raw PCM on stdin — meeting audio never touches disk.
  `SpeakrsCliDiarizer` routes **estimated** counts to the helper and
  **explicit** counts to sherpa (speakrs exposes no way to force a count);
  re-ID voiceprints always come from sherpa's `SpeakerEmbeddingExtractor`
  regardless of backend, so enrolled profiles keep matching.
  `cli._load_diarizer` prefers the helper when built (`build.sh`, needs a
  Rust toolchain), falls back to sherpa-only otherwise; `steno doctor`
  reports it; `eval/diarize.py --sherpa-only` pins the baseline.

**Deferred task — stenodiar on Windows/Linux:** speakrs itself is
cross-platform (ONNX Runtime CPU/CUDA/MIGraphX backends; the CoreML feature is
macOS-only), so the port is "build without the `coreml` feature + package the
binary". The blocker is performance, not correctness: **speakrs' ORT CPU path
measured ~1× realtime pinned to a single core** (407 s per 300 s segment on
the M4 Max; counts matched CoreML on every file). Before shipping it anywhere
without a GPU: investigate threading (ORT intra-op/session thread settings,
speakrs `RuntimeConfig.chunk_emb_workers`, possibly an upstream issue — the
project is v0.5.0 and publishes no CPU numbers); acceptance is multi-core
scaling to well above realtime for a 1-h finalize. CUDA on Linux is already
fast (50–121× RT per speakrs' benchmarks). Fallbacks if CPU can't be fixed:
NME-SC k-estimation (`spectralcluster`, numpy/scipy) feeding sherpa's
known-count path, or pyannote community-1 direct (torch-CPU, HF-gated).
Ruled out: **DiariZen** (best DER but CC-BY-**NC** weights — not shippable,
and WavLM-Large is CPU-heavy); every *end-to-end neural* diarizer (hard-capped:
NVIDIA Sortformer at 4 speakers, LS-EEND at 10) — only clustering pipelines
scale to our 2–8-speaker case. Lever order for many speakers: **known count**
(biggest) > **VBx clustering** (shipped for estimates) > **better embedding** >
the 3-speaker-per-window segmentation cap (least important; a local per-window
limit, not a global one).

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

Speaker-bleed caveats: (a) remote audio played through speakers bleeds into the
mic — echo cancellation is mandatory whenever both channels are captured without
headphones, which is the *default* way of sitting in an online meeting, not just a
hybrid-mode concern. `stenograf.aec` feeds the system channel to WebRTC AEC3 as the
far-end reference. **Settled by PLAN-AEC.md (complete 2026-07-10):** across the full
scenario matrix (quiet/loud, batch/live, built-in/Bluetooth, double-talk) a canceller
with a live reference leaks *zero* transcript lines — 37.6 dB ERLE live, −65 dBFS
residual, AECMOS echo 4.73. No energy gate or neural residual suppressor is needed.
The one real leak mechanism is *losing* the reference (a stalled or mis-clocked tap),
so the cross-channel text dedup at merge time is now an **armed backstop**: it runs
only when `far_end_missing_ticks > 0` (or no canceller was observed), and the CLI
warns with cause and drop count when it fires. Voice Processing IO was evaluated and
rejected — it ducks the remote audio we transcribe, see native/README.md;
(b) the reverse direction is safe —
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
| Remote speaker count | **Shipped (July 2026):** speakrs' VBx estimation via the stenodiar helper on the system channel (an estimated count = "don't pass `num_speakers`") | Good — exact on the online-meeting eval segments; explicit count still authoritative when given |
| Local speaker count | Same, on the mic channel | Weaker (far-field audio) — the one eval miss was the in-room segment (2 detected vs 3 enrolled); detected count stays editable + cheap re-finalize |
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
(system) + AVAudioEngine (mic) → AVAudioConverter to mono
16 kHz int16 → framed PCM on stdout, clean SIGINT/SIGTERM stop; consumed by
`MacOSCaptureProvider` behind the same `CaptureProvider` interface. Verified
end-to-end (July 2026): live mic capture is non-silent and real-time; German
speech played to the system output is captured through the tap and transcribed
accurately (`steno start --local 0 --remote 1`). Automatic de/en language
detection ships as a text vote over the finalized transcript (`stenograf.lid`),
auto-filling the transcript language and locking it for the session. **Phase 1
is complete** — a usable, legally-clean meeting transcriber. Cross-channel text
dedup shipped later, in the PLAN-AEC pass, as an armed backstop. Still deferred:
moving checkpoint finalize off the consume thread (needs real-time
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
*Status (July 2026): Phase 2 critically reviewed (four-subagent audit); detailed
Phase 3 build plan below, leading with a foundations/hardening stage before speaker
re-ID. Glossary lands as text post-correction (Parakeet has no decode-time prompt);
overlap flagging deferred (sherpa's greedy clustering rarely emits overlapping turns).*

**Phase 4 — Product layer + distribution (macOS).**
Local web UI (live captions, meeting archive, click-to-jump transcript), a persistent
meeting archive with a defined reverse-control channel (correct count/language →
re-finalize; rename a speaker), optional local-LLM (Ollama) note-enhancement, and the
macOS wheel/distribution path (bundle + ad-hoc-sign `stenocap`, publish to PyPI).
*Re-scoped July 2026 (five-subagent design pass, decisions below): Linux moved to
Phase 5 so Phase 4 ships a tangible Mac-native product first; `steno start` writes into
a managed meeting archive by default; the in-RAM-only privacy guarantee is preserved
(archive audio playback / archived re-diarize are opt-in, gated on `--record-audio`).
Detailed build plan below.*
*Status (2026-07-10): Stage A (A1 `from_json`, A2 `title`) and Stage B (B1 archive,
B2 CLI wiring + `meetings` group, B3 `MeetingSession`, B4 `ArchivedMeeting`) are
**shipped**. **Re-prioritized 2026-07-10 (Daniel): Stage E — distribution — moves
first**, ahead of the web UI and notes. Rationale: everything built so far is
unreachable on any machine but this repo checkout (`uv tool install` → the wheel
carries no `stenocap`), so shipping is worth more than another feature. Stages C
(web UI) and D (notes) follow E, unchanged in scope.*

**Phase 5 — Linux + cross-platform ASR.**
Linux in-process capture (PipeWire/PulseAudio monitor via SoundCard/`pactl`, no helper)
+ a CPU/ONNX Parakeet-TDT-v3 ASR backend (the *same* model as the macOS MLX backend,
real word timestamps) registered through the already-shipped `stenograf.asr` factory;
diarization already runs ONNX/CPU cross-platform. *Designed in the Phase-4 pass (Track 2,
deferred by an explicit scope decision — product layer first). Finalize-first is
first-class; live captions are best-effort on CPU (Parakeet-int8 ≈ 5–36× realtime vs
MLX's ~110×, so the 12–16 s re-decode window may miss cadence on slower boxes). Sub-plan
summarized under "Deferred to Phase 5" in the Phase 4 build plan.*

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
   *Status (July 2026): shipped (`stenograf.view`, `tests/test_view.py`, 15
   tests). `LiveView` is a plain-class interface whose events all default to
   no-ops (so it doubles as a null view) and which is a context manager
   (`close()` tears down a display — the Textual view will need it). `update`
   bridges a worker `StreamingUpdate` → `commit`/`interim`, matching the `OnUpdate`
   signature so `on_update=view.update` wires straight through. `PlainLiveView`
   streams committed words onto a per-channel line, channel-coarse `You`/`Remote`
   (PLAN.md Task 6), breaking on a channel change or a >1.5 s pause so the log
   reads in utterance paragraphs; it drops the interim grey tail (a non-TTY stream
   has no cursor to erase it — committed text is the durable contract), an
   out-of-band notice always closes the open caption line first, `error` → stderr,
   and one lock keeps worker-thread commits from interleaving mid-line with
   main-thread notices. Tested through an injected echo recorder that mirrors
   `click.echo`'s message/nl/err semantics, and verified end-to-end through the
   real `click.echo`. (The `--live`/`--plain` CLI wiring and the orchestrator's
   structured `finalizing`/`language`/`finalized` events landed in Task 7.)*
6. **Textual TUI** (`TextualLiveView`) — pinned header (REC/elapsed/language/
   profile), append-only `RichLog` of committed captions, dim per-channel interim
   tail (`You`/`Remote` — channel-coarse; real `Local-N`/`Remote-M` only after the
   finalize swap), footer. Minimal-redraw discipline: one 1 Hz clock is the only
   periodic repaint, animations off, `MAX_FPS≈15`; worker→UI via
   `loop.call_soon_threadsafe`. **Ctrl-C is a captured key event under Textual (not
   `KeyboardInterrupt`)** — the quit binding must cross to the worker via
   `provider.stop()`; wire it deliberately.
   *Status (July 2026): shipped (`stenograf.tui`, `tests/test_tui.py`, 13 tests).
   `LiveApp(App)` renders the header/`RichLog`/live-line/`Footer`; `TextualLiveView`
   is the `LiveView` adapter that marshals every event onto the app loop via
   `App.call_from_thread` (Textual's `call_soon_threadsafe` wrapper), dropping
   updates that arrive before mount or after stop (`ready` gate) — the UI is
   best-effort, finalize is authoritative. Committed words stream onto a single
   interleaved "bottom line" (channel-coarse `You`/`Remote`, bright) with its grey
   provisional tail (`[dim]`); the line scrolls up into the append-only log on a
   channel change or a >1.5 s pause, and `finalized` swaps the whole log for the
   diarized `Local-N`/`Remote-M` transcript. Minimal redraw: `animation_level="none"`
   and `TEXTUAL_FPS` pinned to 15 (re-pinned defensively — `MAX_FPS`/`UPDATE_PERIOD`
   bake at import), a single 1 Hz interval as the only periodic repaint, everything
   else event-driven. `action_stop` (bound to `ctrl+c`,`q`, `priority=True`) crosses
   to `stop_callback` (→ `provider.stop`) and shows "finalizing" instead of
   aborting; a second press force-exits, and once finalized `q` just exits.
   `serve(meeting)` runs the app on the main thread while the meeting runs on a
   background thread, returning the transcript on exit. Textual is imported lazily
   here so the plain view (Task 5) stays dependency-free (regression-guarded).
   **Verified end-to-end** by driving the real `MeetingRecorder.run(live=True)`
   (fake ASR, paced provider) into the TUI under Textual's headless `run_test`: a
   channel-coarse `You` caption crossed from the real `LiveWorker` thread onto the
   UI, then the finalize pass swapped in `Local-1`, and the quit binding exited.
   (Task 7 added the `--live`/`--plain` CLI wiring that chooses this view vs the
   plain one by TTY-detect, and moved `finalizing`/`language`/`finalized` emission
   into the orchestrator; `serve`/`_finish` now only backstop the finalize swap.)*
7. **Glue** — `steno start` gains `--live/--no-live`, `--plain`, `--flush-interval`
   (alias `--checkpoint-interval`); doctor/README; a CPU-proxy regression test (zero
   window decodes during silence; committed text never rewritten).
   *Status (July 2026): shipped (`stenograf.cli`, `tests/test_cli.py`). `steno
   start` defaults to `--live`; it picks the view by terminal: the Textual TUI on a
   real TTY, else the plain stream (`--plain` forces plain, `--no-live` skips the
   live pass entirely and finalizes on stop as before). `--flush-interval` /
   `--checkpoint-interval` are one option (Option-B crash checkpoint cadence).
   `--replay` is paced to wall-clock only under `--live` so it demonstrates captions
   at meeting cadence; batch dumps it. The orchestrator now drives a single
   `LiveView` sink (`session._CallbackView` adapts the legacy `on_update`/`on_status`
   callbacks) and emits the structured `status`/`language`/`finalizing`/`finalized`/
   `error` events itself — so `finalized` is emitted by `run` before it returns
   (the TUI's `serve`/`_finish` now only backstops it), resolving the Task 5/6
   deferrals. Under the live views the `.partial` checkpoint is written silently
   (the caption stream/TUI stays clean); batch narrates it as before. CPU-proxy
   regression in `tests/test_live_orchestration.py::TestLivePassCpuProxy`: through
   the wired `run(live=True)` path, zero ASR decodes while the VAD reports silence
   (snapshotted at the finalize hand-off) and a strictly append-only committed
   stream over speech. **Verified end-to-end** driving the real `parakeet-mlx`
   backend through `steno start --replay` (live→plain): the streaming worker,
   structured finalize swap, and transcript write all ran on the background thread
   without the MLX thread-stream fault. Phase 2 complete.*

CPU budget target (spike-measured): **~7–10% of one accelerator during speech, ~0%
in silence**, live captions ~10% WER, ~1.5 s cadence.

### Phase 3 build plan — speaker polish + vocabulary + auto-detection

Phase 2's shipped code was critically reviewed before starting Phase 3 (July 2026,
four-subagent audit: live/orchestration, accuracy core, I/O edges, tests/eval).
Verdict: the live concurrency spine (`SessionStore.view`, `AudioBus` wakeups,
single-flight `LiveWorker`, LocalAgreement-2 monotonicity) is sound and preserved
as-is; the real risks are at the *edges* and in *measurability*. Three findings shape
the sequencing: (a) every test runs on fakes — the real `SherpaOnnxDiarizer`/parakeet
paths are verified only by manual runs, and re-ID stacks a second sherpa path onto that
untested surface; (b) no speaker-labeled ground truth exists anywhere (refs are
label-free plain text; no RTTM, no DER scorer), so diarization/re-ID changes are
currently unmeasurable; (c) two lifecycle bugs lose the finalized transcript on a
double quit/interrupt during finalize. Two library facts were verified against the
*installed* packages and lock two design decisions: sherpa's
`OfflineSpeakerDiarization` result carries **no embeddings** (re-ID needs a separate
`SpeakerEmbeddingExtractor`), and parakeet-mlx `generate()` has **no prompt/hotword
parameter** (a glossary lever is text post-correction, not `initial_prompt`).

So Phase 3 leads with a foundations/hardening stage that makes the headline feature
(speaker re-ID) both *safe to build on* and *measurable*, then builds re-ID, then the
largely-independent export/vocabulary and auto-detection work.

**Stage 0 — Foundations & hardening (first; small, unblocks the rest).**
- **0a — finalize crash on silent channels (HIGH).** `finalize_channel` runs
  `diarizer.diarize` unconditionally even when VAD found no words, and
  `MeetingRecorder.finalize` has no per-channel guard, so a sherpa failure on a
  fully-silent channel (silent remote, dead second mic) can lose *both* channels'
  transcripts. Compute words first, skip diarization + return `[]` when there are none;
  isolate per-channel finalize failures.
- **0b — transcript-loss on double quit/interrupt (HIGH).** A second `q`/Ctrl-C during
  the on-stop finalize makes `serve()` return `None` (the background meeting thread has
  not assigned `result["transcript"]` yet) → CLI crashes on `None.to_markdown()`,
  finalized transcript lost. Capture the authoritative transcript into `result` before
  emitting `finalized`/exiting; join the meeting thread before reading; guard
  `_write_transcript` against `None`; wrap the on-stop `finalize()` so a second
  interrupt cannot drop it. This `serve` pattern is the template the Phase 4 web UI will
  copy — lock it down now.
- **0c — first real-backend `SherpaOnnxDiarizer` test.** All diarization tests use
  `FakeDiarizer`. Add a real-library test (known-count, `num_speakers=None` estimation,
  `set_config` count-change rebuild), gated behind a model-availability marker. This is
  the surface re-ID extends; the MLX thread-stream bug is precedent for "real backend
  breaks what fakes pass."
  *Status (July 2026): shipped (`tests/test_diarization_sherpa.py`). Drives the
  real sherpa pipeline on a real eval clip — known-count (well-formed, sorted,
  in-bounds turns; FastClustering caps the speaker set), `num_speakers=None`
  estimation (`num_clusters=-1`), and the `set_config` count-change rebuild (same
  pipeline instance reused). Gated on sherpa-onnx + cached models + a real clip
  (all opt-in/gitignored) so CI/fresh checkouts skip; assertions structural.*
- **0d — speaker-labeled reference data + DER/attribution scorer (gating
  prerequisite).** Hand-label per-channel speaker turns for `de-1`/`de-2`/`en-1`
  (RTTM), add a DER + word-attribution scorer to `eval/`. Start this *first* — it is the
  long pole, and everything speaker-centric (re-ID threshold tuning, diarization
  upgrades) is unmeasurable without it.
  *Status (July 2026): scorer + tooling shipped; hand-labeling still owed (the
  long pole, Daniel's to do). `eval/rttm.py` (pure NIST RTTM I/O), `eval/der.py`
  (pure numpy+scipy frame-based DER — optimal Hungarian speaker mapping, 0.25 s
  collar, native overlap, missed/false-alarm/confusion split — plus a
  word-attribution scorer under the best label mapping), `eval/diarize.py` (drives
  the real backends to emit hyp RTTM + word JSON; `--bootstrap` seeds
  `refs/<id>.draft.rttm`), unit tests (`tests/test_eval_der.py`, hand-computed
  cases), README workflow. Verified end-to-end on de-1. **Remaining: hand-correct
  the de-1/de-2/en-1 references** (drafts bootstrapped locally; unconstrained
  estimation over-clusters — de-1 → 13 speakers — which the scorer now quantifies).*
- **0e — retain word timestamps on `TranscriptEntry`.** Merge/group already hold the
  word list before collapsing it to a string; add an optional `words` field to the
  entry and serialize it, honoring §Outputs' word-level-JSON promise and unblocking
  subtitle-grade SRT/VTT.
  *Status (July 2026): shipped. `TranscriptEntry.words` (optional `Word` tuple on
  the session clock) flows through `merge_words_turns`, `group_words`,
  `finalize_channel`'s single-speaker path, `relabel_speakers`, and the shifted
  tail-checkpoint entries; `asdict` serializes it into the JSON. Empty only when
  the ASR backend emits no word timestamps. Tests in `test_pipeline.py` /
  `test_transcript.py`.*
- **0f — load-shedding in `LiveWorker`.** The reconcile "catch-up" currently feeds the
  whole backlog into one ever-larger decode (positive feedback if inference falls below
  realtime). Add a "backlog > `window_cap` → skip the window forward" branch so live
  degrades to a caption *gap*, not a spiral — before Phase 3 puts per-frame speaker work
  on the same single worker.
  *Status (July 2026): shipped. When a channel's backlog exceeds
  `decoder.window_cap`, `LiveWorker` abandons the decoder's window (new
  `LiveDecoder.drop_window` — clears the buffer + its origin, keeps committed
  text, no silence padded across the skip) and restarts at the recent edge,
  feeding only the last `window_cap` seconds; the skipped span is a caption gap
  the finalize pass fills. `shed_seconds` tracked. Tests: worker sheds an
  over-long backlog / leaves a normal one, and the decoder stays monotonic across
  the gap (`test_live_orchestration.py`, `test_live.py`).*

**Stage 1 — Speaker re-ID (headline).** Additive interface; live/orchestration
untouched (the channel-coarse → diarized swap in `finalize_channel` is the seam).
- **1a — `DiarizationResult{turns, embeddings}` + `Diarizer.diarize_with_embeddings()`**
  (non-abstract, default `= (diarize(...), {})`). `SherpaOnnxDiarizer` holds one lazy
  `SpeakerEmbeddingExtractor` (same `models.SPEAKER_EMBEDDING` file), embeds each
  cluster's segment slices, L2-normalizes + means per cluster; duration-weight or drop
  sub-~0.5 s segments. `SpeakerTurn` unchanged (embeddings are per-cluster).
  *Status (July 2026): shipped. `DiarizationResult` + the non-abstract default in
  `diarization/base.py`; `SherpaOnnxDiarizer.diarize_with_embeddings` builds a lazy
  `SpeakerEmbeddingExtractor`, embeds each cluster's ≥0.5 s turn slices (short-turn
  fallback), duration-weighted-averages the unit vectors, re-normalizes, omits
  clusters with no embeddable audio. 192-dim eres2net. Real-backend tests assert
  per-cluster unit-norm embeddings, distinct clusters distinct. The profile
  store/relabel that consumes these is 1b (next).*
- **1b — profile store + cosine relabel.** New `profiles` module: a local store keyed by
  the embedding-model id (profiles are model-bound — record which model produced each),
  cosine-match ~0.5. Post-diarization relabel step maps clusters → named profiles or
  enrolls unmatched ones.
  *Status (July 2026): shipped (`stenograf.profiles`, `tests/test_profiles.py`).
  `SpeakerProfile` (name + embedding-model id + unit-norm mean + sample count) and
  `ProfileStore` — atomic JSON in the platform **data** dir (`STENOGRAF_DATA` /
  `~/Library/Application Support/stenograf`, deliberately not the re-downloadable model
  cache), model-scoped `match`/`for_model` (a vector only compares against same-model
  profiles), `enroll`/`rename`/`remove`/`reinforce` (sample-weighted running mean).
  `SpeakerReID.resolve(embeddings)` does the cosine relabel: greedy **one-to-one**
  cluster→profile assignment (two diarizer clusters can never collapse onto one
  profile; unmatched/embedding-less clusters are omitted so the caller keeps its
  channel-coarse label). Wired as an **opt-in, additive** seam: `finalize_channel`
  gains a `reid` resolver (uses `diarize_with_embeddings`, renames matched clusters to
  profile names), `relabel_speakers` now only renumbers raw `S<n>` labels so a matched
  "Daniel" survives instead of becoming `Local-1`, and `MeetingRecorder(reid=…)` threads
  it through the diarized finalize attempt. Default (no store) = zero behaviour change.
  Verified on **real eres2net vectors** (`test_diarization_sherpa.py`): enrol each real
  cluster, resolver re-identifies each as itself (self-match cosine 1.0), and a
  different-model query matches nothing. **Enroll-on-the-fly is intentionally NOT wired
  into the always-on finalize** (it would silently pollute the store with anonymous
  profiles): the store's enroll/rename is built + tested for the 1c CLI to drive
  explicitly ("name unmatched clusters post-meeting"); the wired default is match-only.
  See [[phase3-verified-library-constraints]].*
- **1c — enroll/name UX + CLI** (`steno profiles` list/enroll; name unmatched clusters
  post-meeting). Tune the ~0.5 threshold on the 0d data.
  *Status (July 2026): shipped (`stenograf.cli`, `tests/test_cli.py`). The re-ID
  seam built in 1b was fully wired but unreachable — nothing constructed a
  `SpeakerReID` — so 1c is two halves: the `steno profiles` management CLI and the
  wiring that finally makes enrolled voices relabel meetings. **`steno profiles`
  group:** `list` (model-scoped, flags profiles from a different embedding model as
  inactive), `enroll NAME AUDIO` (computes the voiceprint through the *same*
  `SherpaOnnxDiarizer.diarize_with_embeddings` path the finalize pass matches
  against — enrolment and match must agree — defaulting to a single-speaker clip;
  `--speakers N`/`--speaker S<n>` name one cluster from a multi-speaker recording,
  listing the clusters when the choice is ambiguous; `--reinforce` folds a sample
  into an existing profile), `rename`, `remove`. **Wiring:** `steno start` and
  `transcribe` gained `--reid/--no-reid` (default on) and `--reid-threshold`;
  `_load_reid` builds a resolver from the default store only when it holds profiles
  for the active embedding model, so the finalize pass is byte-for-byte unchanged
  with no profiles (match-only, per 1b). A shared `_load_diarizer` seam backs both
  enrolment and finalize. **Threshold stays at the 0.5 default, deliberately
  un-tuned** — empirical tuning needs the 0d hand-labelled references, which are
  not being produced (Daniel's call); `--reid-threshold` is the per-run override,
  and the `DEFAULT_THRESHOLD` docstring records why. Tests: the profiles CLI + an
  end-to-end enrol→transcribe→relabel with fakes, `--no-reid` restoring generic
  labels. **Verified with the real backends** (the audit's flagged risk: re-ID
  stacks a second sherpa path onto fakes-only surface) — enrolled cluster `S0` of a
  real de-1 slice through the real eres2net extractor, then `steno transcribe
  --speakers 2` relabelled that cluster to the profile name (self-match, real
  parakeet+sherpa, no MLX thread-stream fault), and `--no-reid` fell back to
  `Speaker 1`. See [[phase3-verified-library-constraints]]. **Stage 1 (speaker
  re-ID) complete.**

**Stage 2 — Export & vocabulary (largely independent).**
- **2a — SRT/VTT export.** `to_srt`/`to_vtt` + `--format md,json,srt,vtt`; re-flow into
  short cues using the 0e word times (entries are gap-split speaker turns, too long as
  raw cues). Time-overlapping Local/Remote cues are legal in both formats — pick the
  policy explicitly.
  *Status (July 2026): shipped (`stenograf.transcript`, `tests/test_transcript.py`).
  `to_srt`/`to_vtt` re-flow each entry's retained word timestamps (0e) into short
  cues bounded by three budgets — 84 chars of spoken text, 6 s, and a 1 s internal
  pause — falling back to one whole-turn cue when a wordless backend (Whisper/Voxtral)
  leaves nothing to re-flow. Cue text is greedily wrapped at 42 chars; each cue carries
  its speaker label (SRT `Name: ` prefix, VTT `<v Name>…</v>` voice span with `&<>`
  escaped). **Overlap policy: every speaker's cues are emitted independently and sorted
  by start time — time-overlapping Local/Remote cues are allowed to coexist (both
  formats permit it) and the per-cue label disambiguates them, rather than merging or
  splitting overlaps.** SRT uses `HH:MM:SS,mmm`, VTT `HH:MM:SS.mmm` (integer-ms rounding,
  no float drift). CLI: `steno start`/`transcribe` gained `--format` (comma list,
  default `md,json`; `_parse_formats` validates + de-dupes); `_write_transcript` writes
  each requested format and returns the paths; the `.partial` crash checkpoint stays
  md+json (subtitles of a partial are pointless). Tests cover re-flow, timestamp forms,
  voice-tag escaping, wordless fallback, cross-speaker start-ordering, and the CLI
  format-select/reject paths. **Verified end-to-end with the real parakeet backend**
  (unit tests use fakes): `steno transcribe eval/audio/de-1.wav --format md,json,srt,vtt`
  produced 102 well-formed cues from real word timestamps (each within budget, valid
  `WEBVTT`/SRT structure) at 56× realtime.*
- **2b — glossary/attendees via post-correction.** Fuzzy/phonetic match of a short
  glossary + attendee names against the finalized transcript (model-agnostic,
  deterministic, testable) — the honest lever, since Parakeet has no decode-time
  biasing. `MeetingProfile` gains `glossary`/`attendee_names`/`speaker_profile_store`
  fields + `json.dumps(default=str)` Path-safety. An optional `prompt` param on
  `ASRBackend.transcribe` (Whisper-only effect, no-op on Parakeet) is a cheap add if
  wanted, documented as such.
  *Status (July 2026): shipped (`stenograf.glossary`, `tests/test_glossary.py`).
  Deterministic post-correction over stdlib `difflib` (no ML, no new dependency):
  `apply_glossary` snaps transcript word tokens to canonical glossary/attendee
  spellings when their normalized similarity clears a threshold (default 0.82,
  tunable). Matching is casefold + NFKD accent/umlaut-folded so German ä/ö/ü/ß
  spellings match their ASCII-ish transcriptions, and operates on whole word
  windows — an n-word window snaps to the term's n canonical tokens, **preserving
  each word's timing and attached punctuation**, so the retained word timestamps
  (0e) and the SRT/VTT cues (2a) stay in sync. `words` and `text` are corrected
  independently with the same terms (never rebuilding one from the other) so a
  backend whose words don't fully cover its text can't truncate. Guardrails against
  over-correction: high threshold, a 4-char minimum term length, and greedy
  longest-window-wins non-overlapping matching. Attendee names are registered whole
  **and** per token (first/last correct individually). Known limit: term and
  transcription must share a token count (no split/merge across word boundaries).
  `MeetingProfile` gained `glossary`/`attendee_names` (tuple-coerced) +
  `speaker_profile_store` (Path); `Transcript.to_json` uses `default=str` for the
  Path. CLI: `steno start`/`transcribe` gained `--glossary` (repeatable, comma-list)
  / `--glossary-file` / `--attendee` / `--glossary-threshold` / `--profile-store`
  (a shared `_vocab_options`, gathered by `_collect_terms`); correction runs in
  `MeetingRecorder.finalize` (authoritative transcript only — checkpoints stay raw)
  and in the `transcribe` finalize; `--profile-store` threads into `_load_reid`.
  **The optional `prompt` param on `ASRBackend.transcribe` was deliberately NOT
  added** — Parakeet ignores it and Whisper is a demoted fallback, so the payoff
  didn't justify touching the ASR ABC + every backend (the plan scoped it "if
  wanted"). Tests: word/text correction, timing + punctuation preservation,
  partial-word-coverage no-truncation, multi-word window, threshold gate,
  over-correction guard, `build_terms` dedup/expand/min-length, config coercion,
  Path-safe JSON, CLI correction path. **Verified end-to-end with the real parakeet
  backend** (unit tests use fakes): `steno transcribe eval/audio/de-1.wav --glossary
  "Bierkliniken, Schmieder"` snapped the real ASR tokens "Bierklinik"→"Bierkliniken"
  and both "Schmiederer"→"Schmieder" with word timings preserved into the JSON,
  while correctly leaving the compound "Argus-Bierklinik" untouched (below
  threshold), at 65× realtime.*

**Stage 3 — Auto-detection polish.**
- **3a — local-speaker-count estimation.** Mechanism is one line (`plan_channels` passes
  `None` on the mic channel; remote-count estimation already ships); the real work is
  far-field estimation *quality*, surfacing "Detected: N" as editable, and the cheap
  re-run (already supported over the retained store).
  *Status (July 2026): shipped. `plan_channels` passes the mic count straight through
  (unknown `--local` → estimate, symmetric with `--remote`), dropping the Phase-1 `→1`
  placeholder — so the common `steno start` with no `--local` now diarizes the mic and
  estimates the local count (and `--remote 0` alone becomes a fully-auto in-room run).
  `MeetingRecorder.finalize` records a per-channel `SpeakerCount(channel, requested,
  detected)` list on `recorder.speaker_counts` (`requested` = the plan's count,
  `None`=estimated; `detected` = distinct speakers found) and emits a
  `<channel>: detected N speaker(s)` status for estimated channels. The CLI surfaces it
  as editable: `start` prints `speakers: N local (detected), M remote (given)` and, for
  any estimated channel, `estimated — re-run with --local N [--remote M] to lock or
  correct`; `transcribe` gained the matching `speakers: N detected` + `--speakers N`
  hint. The cheap re-run is the existing `transcribe`/`--record-audio` path over the
  retained/recorded audio (a wrong estimate is never fatal — §2). **Far-field estimation
  quality is deliberately unchanged** (the documented weakness): sherpa's unconstrained
  `FastClustering` over-clusters, so the surfaced count is the honest lever — tuning it
  needs the 0d hand-labelled references (not being produced) or the community-1/VBx
  clustering upgrade (later work). Verified end-to-end on `de-inroom.wav` via
  `steno start --remote 0 --replay … --no-live` through the **real parakeet+sherpa**
  backends (unit tests fake the diarizer): the mic estimated 8 local speakers
  (over-split, as expected) and printed the detected count + correction hint. Tests:
  `plan_channels` estimates unknown/all-unknown counts (`test_session.py`), `finalize`
  populates requested/detected `speaker_counts`, and the CLI detected/given/hint output
  (`test_cli.py`). **3b (parameter provenance written back to the transcript) is the
  remaining Stage 3 task.**
- **3b — parameter provenance** (`explicit | detected | default`) written back to the
  transcript/profile (today only `None`=auto, which collapses once filled, and detected
  values are not recorded back). Meeting-mode (online/hybrid/in-room) detection needs
  capture-side signals (meeting-app process + tap activity) → late Phase 3 / Phase 4.
  *Status (July 2026): shipped. `stenograf.config` gained a `Provenance` enum
  (`explicit`/`detected`/`default`), a `ResolvedValue{value, provenance}`, a pure
  `resolve_value(explicit, detected)` (explicit wins → detected → default; `None`,
  not falsiness, marks "absent" so an explicit `0` listen-only channel is kept), and a
  `ResolvedParameters{language, speakers: dict[channel→ResolvedValue]}`. `Transcript`
  gained an optional `parameters` field serialized under a `"parameters"` JSON key
  (`null` on crash checkpoints, which predate the resolving finalize). The **meeting
  path** (`MeetingRecorder.finalize`) builds it via `session.resolve_parameters(profile,
  language=…, speaker_counts=…)` — both `mic`/`system` channels always recorded so an
  explicit `0` ("channel off") is captured. The **file transcribe path** records language
  + a single `"audio"` channel count (no local/remote model for one un-split stream), and
  its profile now keeps the *user's* language (`None`=auto) rather than back-writing the
  detected value — so `transcript.language`=resolved while `profile.language`=input,
  matching `start` and giving `parameters.language` the sole provenance record. **Mode
  provenance is intentionally out** (the plan's `explicit|detected|default` triad; mode
  auto-detection is the deferred capture-side-signals work). Tests: `resolve_value`
  triad + zero-is-a-value (`test_config.py`), `resolve_parameters` explicit/detected/
  default + finalize-attaches-parameters (`test_session.py`), JSON `parameters` shape +
  `null`-when-absent (`test_transcript.py`), CLI transcribe auto→detected /
  explicit→explicit JSON (`test_cli.py`). **Verified end-to-end with the real
  parakeet+sherpa backends** (unit tests use fakes): `transcribe` auto → language/audio
  both `detected`, `--lang de --speakers 2` → both `explicit`; `start --remote 0` in-room
  replay → language `detected`, mic `detected 1`, system `explicit 0`, with top-level
  `language: de` but `profile.language: null`. **Stage 3 (auto-detection polish)
  complete** — remaining meeting-mode detection is deferred to late Phase 3 / Phase 4.

**Deferred (noted, not built in Phase 3):** overlap flagging is structurally
near-silent with sherpa's greedy clustering (rarely emits overlapping turns) — real
overlap needs the community-1/VBx upgrade, so keep the merge code but do not
over-invest; the wheel build hook that bundles/signs `stenocap` (no non-repo install
works without it) is a Phase 4 distribution blocker; smaller hardening (atomic model
extraction, per-channel `WavTee` drain so a laggard channel cannot stall the tee,
piping helper stderr so it does not splatter the TUI) folds into Stage 0
opportunistically.

### Phase 3 → Phase 4 readiness audit (July 2026)

Before starting Phase 4 (local web UI, Ollama note-enhancement, Linux capture +
ONNX/CTranslate2 backends), Phase 3's shipped code was critically reviewed by a
four-subagent audit (correctness of the new modules; lifecycle/concurrency/I-O
edges; Phase-4 architectural readiness; tests/eval/docs/packaging).

**Verdict: architecturally ready to *start* Phase 4 — no hard blocker to
development.** The platform seams the plan promised are real: the
``CaptureProvider`` ABC and the ``LiveView`` event interface are clean and
terminal-free (a Linux provider and a websocket web-UI view each drop in as new
implementations with zero core changes), diarization is already ONNX/CPU
cross-platform, and MLX is lazy-imported so the package imports on Linux. The two
Stage-0 lifecycle hardenings (0a silent-channel, 0b double-quit) are genuinely
fixed and tested, the live concurrency spine is clean, and the Phase-3 shipping
path (Parakeet, no re-ID) is correct. Full suite green (263 passed on macOS with
models cached). But Phase 4 copies the two most fragile parts of the current code
(the ``serve()`` teardown template, the single hardcoded ASR backend), and the
green suite hides that the real diarizer has *zero* CI coverage and the suite does
not even collect on Linux — so a focused pre-Phase-4 hardening pass is warranted.

**Tier 1 — fix before Phase 4 (small, high-leverage; Phase 4 builds on these).**
*Implemented in the pre-Phase-4 hardening pass (July 2026) — see per-item status.*
1. **Lock down the capture-teardown / ``serve()`` template.** The plan calls
   ``serve()`` "the template the Phase 4 web UI will copy", but ``provider.stop()``
   blocks (up to 5 s ``proc.wait``) *on the Textual event loop* (freezing the UI and
   deadening the second-Ctrl-C escape), a capture-thread error re-raised *past*
   finalize discarded a fully-finalizable buffer (contradicting "finalize is
   authoritative"), and ``MacOSCaptureProvider.stop()`` was called from 2–3 threads
   with no lock.
   *Status (July 2026): shipped.* ``MacOSCaptureProvider.stop()`` is now
   idempotent + thread-safe (captures and nulls ``_proc`` under a lock, so
   concurrent/repeat calls are no-ops); the TUI's ``action_stop`` runs the blocking
   teardown on a background thread so the UI stays responsive and a second Ctrl-C
   still force-exits; and capture-thread errors in both ``_run_live`` and
   ``_run_batch`` are surfaced via ``view.error`` but no longer abort the finalize —
   a desync/late error still yields a transcript of the captured audio.
2. **ASR backend-selection factory.** ``_load_backends`` hardcoded
   ``ParakeetMLXBackend()`` and ``doctor`` hardcoded the same; there is no second
   backend yet (no Whisper, no ONNX/CTranslate2). Add the factory/registry *before*
   writing the Linux backend so it is a drop-in, not a ``_load_backends`` rewrite.
   *Status (July 2026): shipped.* ``stenograf.asr`` gains a lazy registry
   (``create_backend`` / ``default_backend_name`` / ``get_spec`` /
   ``available_backends``); ``cli._load_backends`` and ``doctor._asr_check`` route
   through it. Registration is the single seam a Linux ONNX/CTranslate2 backend
   plugs into; imports stay lazy so choosing a backend never imports another's deps.
3. **Unbreak Linux CI + give the real diarizer a regression net.** ``scipy`` (used
   by ``eval/der.py`` + ``test_eval_der.py``, both in the default suite) was declared
   in no dependency group — it only resolved transitively via the macOS-only
   ``parakeet-mlx``, so ``pytest`` failed to *collect* on Linux. And
   ``diarization/sherpa.py`` executed zero test lines on any fresh checkout (its
   embedding aggregation is reachable only through the model-gated real-backend test).
   *Status (July 2026): shipped.* ``scipy`` declared in ``dev`` + ``eval``;
   ``tests/test_diarization_sherpa_unit.py`` drives ``diarize_with_embeddings`` +
   ``_l2_normalize`` through a fake ``SpeakerEmbeddingExtractor`` (unit-norm output,
   duration weighting, empty-cluster omission, short-turn fallback, zero-vector
   guard) — no models, runs everywhere.
4. **Atomic writes for the crash-recovery artifacts.** ``_write_transcript`` used
   ``write_text`` (truncate-in-place), so a crash mid-checkpoint corrupts *and*
   destroys the previous good ``.partial`` — the artifact does not survive the crash
   it exists for.
   *Status (July 2026): shipped.* ``_write_transcript`` writes via a temp file +
   ``os.replace`` (the same atomic pattern ``ProfileStore.save`` already uses),
   covering both the final transcript and every ``.partial`` checkpoint.
5. **Fix small correctness landmines a form-driven web UI will trip.**
   ``SpeakerProfile`` (frozen dataclass with an ndarray field) had a
   ``__hash__``/``__eq__`` that *raise*; ``--local 0 --remote 0`` raised an uncaught
   ``ValueError`` (traceback, not a clean error); and the detected-count correction
   hint was unclamped (a silent channel → nonsensical "re-run with ``--local 0``";
   an over-cluster estimate → an uncorrectable out-of-range hint).
   *Status (July 2026): shipped.* ``SpeakerProfile`` is ``eq=False`` (identity
   equality, hashable by id); ``start`` maps the profile ``ValueError`` to a
   ``ClickException``; the lock-count hint is suppressed when nothing was found and
   clamped to the settable range (with a note) when the estimate exceeds it.

**Tier 2 — design up front as Phase 4 opens (its own scope, but decide early).**
- ``Transcript.from_json`` loader + a meeting archive/index with stable IDs (the
  "meeting archive" view needs to reload persisted transcripts; today ``Transcript``
  serializes four formats but cannot read one back).
- A structured reverse-control channel (correct the count/language and re-run
  finalize; rename a speaker). ``MeetingRecorder.finalize`` is already re-runnable
  over the retained store, so the seam exists — it needs a defined interface, not a
  web-UI afterthought. The informal ``stop_callback`` is the only reverse channel today.
- Resolve the in-RAM-audio ↔ click-to-jump tension: text-jump works (word timestamps
  are in the JSON), but archive audio playback contradicts the in-memory-only
  guarantee unless ``--record-audio`` was on. Decide the UX before building it.

**Known deferrals (acknowledged, not surprises).**
- **Wheel build hook + CI to bundle/sign ``stenocap``** — the one true *distribution*
  blocker (today only ``uv run`` in-repo captures audio; ``uv tool install`` / ``uvx``
  → ``HelperNotFoundError``). **Now the current priority: Phase 4 Stage E**, promoted
  ahead of the web UI and notes on 2026-07-10.
- **0d hand-labelled RTTM references** — the DER/word-attribution scorer is built and
  tested, but no references exist, so diarization/re-ID quality and any Phase-4
  backend swap stay unmeasurable (Daniel's call not to hand-label). Consequences: the
  re-ID threshold stays at an untuned 0.5, and far-field local-speaker-count estimation
  is known to over-split (a small group measured as 8).
- **Capture-tap fragility (two open defects, see PLAN-AEC.md §5).** (1) Any Python-side
  stall over ~1 s permanently kills the Core Audio tap with no recovery — a drain thread
  in ``MacOSCaptureProvider`` would decouple it; two separate bugs have already been
  traced to this. (2) A tap that keeps delivering **all-zero** PCM is undetected:
  ``far_end_missing_ticks`` counts only *absent* far-end frames, so the armed text
  backstop never arms and no warning fires while the canceller runs blind.
- **Lower-priority, independent:** greedy re-ID → optimal (Hungarian) assignment;
  SRT/VTT dropping text not covered by ``words`` (latent — Parakeet emits full-or-none);
  helper-stderr piping; atomic model extraction (tar path — folded into Stage E2);
  meeting-mode auto-detect; hybrid cross-channel dedup.
  *(The README gap — missing ``--format``/SRT-VTT and the glossary family — was closed
  on 2026-07-10 along with the stale "capture isn't wired up" status note.)*

---

### Phase 4 build plan — product layer + macOS distribution (Linux → Phase 5)

Planned July 2026 by a **five-subagent design pass** (web UI · persistence/archive ·
Linux backends · Ollama notes · distribution), each grounded in the shipped code seams
and returning a staged sub-plan; synthesized here into one sequence. Full per-workstream
sub-plans were captured in the design session.

**Locked scope decisions (Daniel, July 2026):**
- **Product layer first; Linux deferred to Phase 5.** Phase 4 = a tangible Mac-native
  product (web UI + archive + notes) + the macOS shipping path. The whole Linux track
  (ONNX ASR backend, `LinuxCaptureProvider`, Linux dep markers) is designed but moves to
  Phase 5 — summarized under "Deferred to Phase 5" below so the work isn't lost.
- **`steno start` writes into a managed archive dir by default** —
  `data_dir()/meetings/<id>/transcript.*`; `--out PATH` overrides and still registers.
  Makes the archive an authoritative library, not an index over scattered files.
- **In-RAM-only privacy guarantee preserved.** Audio never touches disk unless
  `--record-audio`. Text click-to-jump is *always* available (word timestamps live in the
  JSON); archive audio **playback** and archived **re-diarize** are opt-in, gated on one
  `record.has_audio()` predicate.

**Adopted recommendations (defaults; overridable):** web server = **Starlette + uvicorn**
(one localhost port for HTTP + WebSocket + static assets + reverse-control POSTs;
in-process `TestClient` = the headless-test analogue of the TUI's `app.run_test()`); web
front-end = **vanilla JS + server-rendered shell, no build step**, assets packaged via
`importlib.resources`; web auth = **per-process bearer token + Host/Origin guard by
default** (reverse-control can re-finalize / rename; DNS-rebinding defense); notes stored
as **sibling `<stem>.notes.md`/`.notes.json`** (keep `Transcript` pure); notes LLM =
**pluggable backend** (revised 2026-07-10, see Stage D preamble): an `ollama` backend
(fully local; default model **`qwen3:8b`**, ~5 GB, won't swap the 48 GB Mac; no `ollama`
pip dep — **stdlib `urllib` HTTP** to `localhost:11434`) and a `command` backend (any
CLI that takes a prompt on stdin and emits schema JSON — e.g. `claude -p`);
refinalize **keeps the locked language** unless
overridden; transcript JSON gets an **index-only `version` stamp** (metadata lives in the
archive index); macOS signing stays **ad-hoc only** (no Developer ID — verified in the
Phase-1 spike); platform deps via **markers, not extras**; Windows **left installable**
with an honest `doctor`.

**The keystone.** `Transcript.from_json` (a loader `Transcript` lacks today) is a
prerequisite for the archive, the web reader, and `steno notes` — three workstreams
independently flagged it. Build it first; it unblocks the most.

**Evaluation stays label-free** (Daniel's standing no-hand-labels call): round-trip /
property tests, fakes + headless `TestClient`, real-backend end-to-end via `--replay`,
and real-backend-gated e2e (Ollama and/or a real `claude` CLI, skipped when absent)
mirroring the model-gated ASR tests. No accuracy scoring.

Task sequence (independent, testable increments; interface names illustrative; ``[dep:…]``
marks a hard prerequisite):

**Stage A — Shared foundations (unblock everything).**
- **A1 — `Transcript.from_json` + `SCHEMA_VERSION`.** A classmethod that faithfully
  round-trips every field `to_json` writes (entries, `Word` timestamps, `MeetingProfile`,
  `language`, `ResolvedParameters`), plus an additive `"version"` stamp. The only typing
  snag is `ResolvedValue.value: object` — a single `_value_from_json` coerces `language`→
  `Language`, speakers→`int`. Compat: missing `version` ⇒ legacy v1; a future major raises
  `UnsupportedTranscriptVersion`; unknown keys ignored. Acceptance: `from_json(t.to_json())
  == t` over a matrix (words present/absent, params None/populated, provisional, Path-valued
  store, hour-scale timestamps); legacy + version-999 + extra-key cases; real e2e via
  `steno transcribe` → reload. `[dep: none]`
  *Status (July 2026): shipped (`stenograf.transcript`, `tests/test_transcript.py`).
  `to_json` now stamps a top-level `"version": SCHEMA_VERSION` (=1); `Transcript.from_json`
  is a classmethod that faithfully reconstructs every field — entries, `Word` timestamps
  (incl. `confidence`), the `MeetingProfile` (glossary/attendee tuples + Path-valued store,
  reproduced by leaning on `MeetingProfile.__post_init__`'s coercion), `language`, and the
  `ResolvedParameters` provenance. The one type-erased snag (`ResolvedValue.value: object`)
  is handled by a single parametrized `_value_from_json(obj, coerce)` — `coerce=Language`
  for the language value, `coerce=int` for speaker counts, `None` (the DEFAULT provenance)
  kept as-is (so an explicit `0` listen-only channel round-trips). Compat: a missing
  `version` is read as legacy v1, unknown keys are `.get`-ignored (additive-field tolerant),
  and a `version` newer than `SCHEMA_VERSION` raises `UnsupportedTranscriptVersion`.
  Tests: a parametrized round-trip matrix (`from_json(t.to_json()) == t` across
  words-present/absent, populated/default/absent params, provisional, Path store, hour-scale
  times, zero-count channel), the version stamp, legacy-missing-version, version-999 reject,
  and extra-key ignore. **Verified end-to-end with the real parakeet+sherpa backends**
  (unit tests use synthetic transcripts): `steno transcribe eval/audio/de-1.wav --lang de
  --speakers 2` → reload the written JSON through `from_json` was **byte-identical on
  reserialize** (`t.to_json() == raw`) and dataclass-equal, faithfully carrying 12 entries /
  230 word timestamps / resolved parameters. This is the keystone A1 the archive (B1), the
  web reader (C6), and `steno notes` (D3) build on.
- **A2 — add `MeetingProfile.title`.** Small field used by the archive record and the notes
  prompt (both siblings want it); `__post_init__` already normalizes the profile. `[dep: none]`
  *Status (July 2026): shipped (`stenograf.config`, `tests/test_config.py`). Optional
  `title: str | None = None` on `MeetingProfile`; `__post_init__` strips it and collapses a
  blank/whitespace-only title to `None` (so `""` and `None` aren't two states), alongside
  the existing glossary/store normalization. It serializes through `asdict` into the
  transcript JSON's `profile` and reloads via the A1 `Transcript.from_json` (added to
  `_profile_from_json`; a legacy transcript with no `title` key reads back as `None`, no
  version bump — additive field). Tests: default-None + strip/blank-collapse
  normalization, and a `titled_profile` case in the `from_json` round-trip matrix. The
  CLI `--title` entry point and the archive record's `title` land with the archive wiring
  (B1/B2) and the notes prompt (D2), which read `profile.title`.

**Stage B — Persistence: meeting archive + reverse-control channel.**
- **B1 — `MeetingArchive` + `MeetingRecord` + index** (`stenograf/archive.py`, mirroring
  `ProfileStore`). Atomic-JSON `data_dir()/meetings/index.json` + a managed
  `meetings/<id>/` layout where the dir name *is* the stable id
  (`meeting-YYYYMMDD-HHMMSS` + collision suffix). `MeetingRecord{id, title, created_at,
  duration_s, language, speakers, formats, dir, audio_path}`. Maintained index (not scan)
  + a `reconcile()` self-heal. `load_transcript(id)` reads through A1. Acceptance:
  add/list/get/remove round-trip, atomic save, reconcile drops vanished + adopts orphan
  dirs, id collision-suffixing. `[dep: A1]`
  *Status (July 2026): shipped (`stenograf.archive`, `tests/test_archive.py`). Mirrors
  `ProfileStore`: lives at `data_dir()/meetings/` (new `meetings_dir()`; honors
  `$STENOGRAF_DATA`, distinct from the model cache), atomic temp+replace `index.json`
  writes. `MeetingRecord` is a plain (unfrozen, value-eq, deliberately unhashable — the
  `speakers` dict) dataclass with the full field set + `has_audio()` (the one predicate
  gating B4 archived audio playback/re-diarize — true only when the referenced WAV
  actually exists). `MeetingArchive`: id-keyed dict for O(1) `get`/`remove`; `add`/`remove`
  persist immediately; `allocate_id(created_at)` mints `meeting-YYYYMMDD-HHMMSS` and
  suffixes `-2/-3/…` past any collision in *both* the in-memory index and on-disk dirs;
  `meeting_dir(id)` is the managed default output location (B2 wires `--out` default to
  it); `load_transcript(id)` reads `<dir>/transcript.json` back through
  `Transcript.from_json` (A1). `reconcile()` drops records whose `dir` vanished and adopts
  orphan managed dirs — `_record_from_dir` rebuilds metadata from the transcript
  (title/language/speakers from its resolved parameters, `duration_s` from the last entry
  end, `formats` from the present `transcript.*` files, `created_at` recovered from the id,
  audio from a present `audio.wav`), skipping dirs with no readable transcript. External
  `--out` dirs are never scanned for adoption (only the managed root). Tests
  (label-free, no backends): add/list/get/remove round-trip, atomic reload-equal (no `.part`
  turds), missing-index-is-empty, collision-suffixing (in-index + on-disk), `load_transcript`
  faithful A1 round-trip, reconcile drop-vanished / adopt-orphan-with-correct-metadata /
  skip-transcript-less-dir, `has_audio` file-existence gating, `$STENOGRAF_DATA` default.
  Unblocks B2 (CLI archive wiring), B3/B4 (reverse control), and the C5/C6 web archive/reader.
- **B2 — wire CLI writes into the archive + a `meetings` group.** After `_write_transcript`
  in `start`/`transcribe`, build a `MeetingRecord` and `archive.add()`; **default `out_dir`
  → `meetings_dir()/<id>`** (managed-dir decision), `--out` an explicit registering
  override; record the `--record-audio` WAV as `audio_path`. New `meetings list/show/rm`
  group mirroring `profiles`. `--no-archive` escape hatch. Acceptance in `test_cli` via
  `--replay`/fake-stenocap. `[dep: B1]`
  *Status (July 2026): shipped (`cli.py`, `tests/test_cli.py`). A new shared
  `_prepare_output(no_archive, out, created_at, legacy_dir, legacy_stem)` resolves, for
  both `start` and `transcribe`, `(archive, meeting_id, out_dir, basename, audio_default)`.
  Archive-on (the default): a managed per-meeting dir (`archive.meeting_dir(id)`, or `--out`
  used as that meeting's own dir) holding **plainly named** `transcript.{fmt}` + `audio.wav`
  — the exact layout B1's `load_transcript`/`reconcile` read back — and a `MeetingArchive`
  to register into after the write. To make that naming possible, `_write_transcript`'s 3rd
  arg became a full `basename` (`transcript` managed, `<stem>.transcript` legacy) instead of
  a stem it appended `.transcript.` to; `_checkpoint_writer`/`_cleanup_checkpoints`/`_make_tee`
  thread the same basename/`audio_default`. `--no-archive` restores the pre-Phase-4 flat,
  timestamp-named output (`<stem>.transcript.{fmt}` into `--out`/cwd/input-parent) with no
  registration. `--title` (A2) flows into the `MeetingProfile` → transcript JSON → record;
  `transcribe` references the **source file** as the meeting's audio (already on disk, so
  archived playback/re-diarize work at no cost to the live-capture in-RAM guarantee).
  `_meeting_record` denormalizes the same fields `_record_from_dir` recovers (created_at
  stamped `isoformat(timespec="seconds")` to match reconcile's format). `meetings list`
  (reconciles first when the root exists, then lists newest-first), `show <id>`, and
  `rm <id>` (`--yes`, `--keep-files`; only ever `rmtree`s a dir that is the archive root's
  own child — an external `--out` dir is just unregistered) mirror the `profiles` group.
  Tests: an autouse `$STENOGRAF_DATA` fixture isolates the archive for every CLI test; new
  cases cover default-managed-dir + index + A1 round-trip, `--out` registering at the
  override, `transcribe` archiving + source-audio reference, `--no-archive` flat+unregistered,
  `--record-audio` → managed `audio.wav` gated by `has_audio()`, and the full
  `list`/`show`/`rm`(+`--keep-files`) lifecycle. Unblocks B3/B4 (reverse control over these
  records) and C5/C6 (web archive list + reader).*
- **B3 — reverse-control interface: `MeetingSession` + `FinalizeRequest`** (`stenograf/
  control.py`), replacing the informal `stop_callback` as the one defined reverse path.
  `FinalizeRequest{local_speakers, remote_speakers, language, reid}` (all optional, None =
  keep). `MeetingSession` holds the recorder + retained store + current transcript:
  `refinalize(request)` overrides the profile via `dataclasses.replace`, re-runs
  `recorder.finalize` (backends reused, not reloaded); `rename_speaker(old,new)` is a pure
  entry relabel (`rename_entry_speaker`, timestamps untouched). Acceptance: fakes +
  synthetic store; refinalize changes the plan/provenance; rename touches only that label;
  ASR loaded once across finalize+refinalize. `[dep: A1; reuses session.finalize]`
  *Status (July 2026): shipped (`stenograf.control`, `tests/test_control.py`). `FinalizeRequest`
  is a frozen dataclass of four optionals (`local_speakers`, `remote_speakers`, `language`,
  `reid`), all defaulting `None` = "keep whatever the session already resolved" — so an empty
  request re-finalizes unchanged and a one-field request corrects just that parameter.
  `MeetingSession(recorder, store, *, transcript=None, stop=None)` holds the warm recorder, the
  in-RAM store, and the current transcript. `refinalize(request)` applies the set fields **in
  place and sticky** — speaker counts / language via `dataclasses.replace(recorder.profile, …)`,
  and an explicit language *also* sets `recorder.language` directly (the profile alone can't
  override a locked auto-detection), then re-runs `recorder.finalize(store)` on the **same**
  backend objects (never reloads a model). So a changed count flips the plan's requested count
  and the transcript's provenance DETECTED→EXPLICIT; a language override beats the German lock
  (reaches the ASR) while "keep" preserves a DETECTED lock without re-detecting. `reid` is a
  sticky toggle over a resolver reference captured at construction (`True` re-enables, a no-op
  with no resolver; `False` disables; `None` keeps). `rename_speaker(old,new)` is a pure
  `rename_entry_speaker` relabel (new `pipeline.rename_entry_speaker`: `replace(e, speaker=new)`
  only where `e.speaker == old`, text/timestamps/words/order untouched; absent label = no-op),
  requiring a finalized transcript. `stop()` is the formalized capture-stop hook that replaces
  the informal `stop_callback` (a no-op once finalized). Both mutators update **and** return
  `session.transcript`. 12 label-free tests (fakes + synthetic store): empty-request reuse (same
  `asr` object, 0 loads), remote-count plan+provenance change, language override vs kept lock,
  sticky re-ID toggle + resolver-less no-op, invalid-override raise, rename-only-target /
  absent-noop / before-finalize raise, stop delegate. Unblocks B4 (archived twin) and C7 (web
  reverse-control POSTs). TUI/CLI still use `stop_callback`; folding them onto `MeetingSession`
  is a later integration, not B3.*
- **B4 — archived reverse control + audio policy** (`ArchivedMeeting`, store-is-gone case).
  `rename_speaker` **always** works (relabel loaded transcript, rewrite managed formats,
  re-add record). `refinalize` works **only when `record.has_audio()`** — rehydrate a store
  from the WAV (the `transcribe`-over-WAV path), rewrite under the same id; else raise
  `AudioUnavailable`. Same `has_audio()` predicate gates playback and re-diarize; audio-
  synced seek is `word.start * SAMPLE_RATE` (the `WavTee` shares the t=0 clock). Recording
  stays off by default. Acceptance: fakes + synthetic WAV; rename persists; refinalize on a
  no-audio record raises; e2e via `--replay --record-audio` → reopen → refinalize. `[dep:
  A1, B1–B3]`
  *Status (July 2026): shipped (`stenograf.control.ArchivedMeeting` + `AudioUnavailable`,
  `recording.read_channels`, `MeetingArchive.rewrite`; `tests/test_control.py`,
  `test_recording.py`, `test_archive.py`). `ArchivedMeeting(archive, record, *, transcript=None)`
  loads the meeting's transcript through A1 and applies the same two corrections as
  `MeetingSession`, persisting each under the stable id. `rename_speaker(old,new)` **always**
  works — pure `rename_entry_speaker` relabel, then `archive.rewrite`. `refinalize(request, *,
  recorder)` is gated on `record.has_audio()` (else `AudioUnavailable`): it rehydrates a
  per-channel `SessionStore` from the recorded WAV (`read_channels` — the exact inverse of
  `WavTee`'s mic-left/system-right layout, an int16 16 kHz PCM reader that takes the meeting's
  captured channel list to disambiguate a mono file; anchored at the shared t=0 clock), anchors
  the freshly-loaded `recorder` to the archived profile+language, then **delegates to a
  `MeetingSession`** so the B3 override/provenance rules apply verbatim, and writes back under
  the same id. New `MeetingArchive.rewrite(record, transcript)` is the shared persistence half:
  re-render each of the record's formats into `<dir>/transcript.{fmt}` (atomic temp+replace) and
  refresh the index metadata (title/language/speakers/duration; id/created_at/dir/formats/audio
  kept) — factored `_speakers_from_transcript` + `_atomic_write_text`, reused by `_record_from_dir`.
  The `has_audio()` predicate is the single gate for archived re-finalize (and, later, C6
  playback / re-diarize); recording stays off by default, so the in-RAM-only guarantee holds.
  9 label-free tests (fakes + real synthetic WAV): rename persists across a reload without audio;
  record→reopen→refinalize rewrites under the same id with the rehydrated per-channel count and
  DETECTED→EXPLICIT provenance, recording untouched; no-audio refinalize raises; `read_channels`
  stereo/mono round-trip + channel-count mismatch; `rewrite` re-render + no-temp-turds. **Scope:**
  targets the live-captured `--record-audio` WAV (mic/system); re-finalizing an imported non-
  recording source (a file-`transcribe` record's external source) is not wired through
  `ArchivedMeeting` — that record still supports rename + playback. Unblocks C7 (web reverse-
  control POSTs consume `MeetingSession`/`ArchivedMeeting`).*

**Stage E — macOS distribution: COMPLETE (E1–E5 shipped 2026-07-10). stenograf 0.1.0 is
on PyPI; the install path is `uv tool install stenograf`, verified from a clean
environment. Remaining Phase 4 work: C (web UI) ∥ D (notes).**
Ships the current Mac tool to colleagues via PyPI; the `stenocap` bundling is the one true
shipping blocker. Promoted ahead of C and D on 2026-07-10 — Stages A+B built a real product
that no one but this checkout can run.

*Verified repo state at the start of this stage (2026-07-10):* there is no `hatch_build.py`,
no `src/stenograf/bin/`, and no `.github/`; `native/helper/stenocap` is a **gitignored build
artifact**, built by `sh native/helper/build.sh`. A wheel built today is therefore a pure
`py3-none-any` wheel carrying no helper, and `uv tool install stenograf` yields a package
whose `capture.macos.find_helper` raises `HelperNotFoundError` the moment `steno start` runs.
Only `uv run` in-repo captures audio. E1 closes exactly that gap; the rest make it safe,
verifiable, and repeatable.

- **E1 — `hatch_build.py` build hook + `find_helper` hardening.** A hatchling custom build
  hook that, **only on macOS-arm64**, shells `native/helper/build.sh` (reuses the one
  `swiftc` + `codesign -s -` line), force-includes the binary at `stenograf/bin/stenocap`
  (mode `0o755`), and stamps `build_data["pure_python"]=False` + `tag=
  "py3-none-macosx_14_0_arm64"`; no-op elsewhere → pure `py3-none-any` wheel. Register it via
  `[tool.hatch.build.targets.wheel.hooks.custom]` in `pyproject.toml`. `find_helper` gains an
  `os.access(X_OK)` guard that `chmod +x`es its own binary. Note the build hook must fail
  loudly if `swiftc` is absent on an arm64 Mac (a silently pure wheel is the bug this stage
  exists to kill). Acceptance: `uv build` emits the arm64 wheel carrying
  `stenograf/bin/stenocap`; clean-venv install → `find_helper` returns an executable
  site-packages path; the `any` wheel has no `bin/`. `[dep: none]`
  *Status (2026-07-10): shipped (`hatch_build.py`, registered via
  `[tool.hatch.build.targets.wheel.hooks.custom]`; `tests/test_build_hook.py`). Verified:
  `uv build` emits the tagged arm64 wheel from the sdist (proving the Swift sources ship in
  it) with the signed helper at 0755; a clean `uv tool install git+file://<repo>` → `steno
  doctor` fully green, helper resolves to site-packages, and `steno setup` captures real
  audio on both channels. Two resolver traps fixed along the way: `requires-python` capped
  `<3.14` (the parakeet-mlx → librosa → numba chain has no 3.14 support, and a 3.14-default
  machine otherwise backtracks into unbuildable sdists) and a marker-gated `numba>=0.60`
  floor (else a fresh resolve keeps the newest numpy and backtracks numba into pre-metadata
  sdists).*
- **E2 — dep markers/matrix + atomic tar extract.** Confirm the wheel matrix
  (arm64-with-helper + pure `any`); keep `parakeet-mlx` marker-gated; reserve a `[ollama]`
  extra name only (no dep). Fold in the deferred **atomic model extraction (tar path)** fix
  (`models._extract_member` → temp+`os.replace`). Acceptance: `uv sync` resolves on both OSes;
  interrupted extraction leaves no truncated model. `[dep: none]`
  *Status (2026-07-10): shipped (empty `[ollama]` extra reserved; atomic extract in
  `models._extract_member` + truncated-archive test). Cross-OS resolution rides on the
  existing lock; the Linux leg is re-checked by E4's CI matrix.*
- **E3 — signing verified + `doctor`/`steno setup` permission UX.** Verify the ad-hoc
  signature survives the zip round-trip (`codesign -v`, no `com.apple.quarantine`) and the
  binary is launchable; extend `_capture_helper_check` (present + executable + signed) and
  add a `steno setup` that deliberately triggers the one-time TCC mic+system-audio prompt.
  Acceptance: `steno doctor` green on a clean install; honest limits documented (per-terminal
  grant, no headless system-audio). `[dep: E1]`
  *Status (2026-07-10): shipped. Signature verified through the wheel zip round-trip
  (`codesign -v` passes, no quarantine attr); `_capture_helper_check` now also requires
  executable + valid signature; `steno setup` runs the helper until the first mic frame —
  the helper requests mic permission, then creates the system tap, then starts the mic
  engine, so one mic frame proves both TCC grants (the system channel is silent without
  playback and can't be the signal). Per-terminal-app grant documented in README + the
  command's output.*
- **E4 — CI + release pipeline.** `.github/workflows/ci.yml` (matrix macos-14 + ubuntu-latest:
  `ruff` + `pytest`, model-gated + real-audio tests self-skip — the Linux job keeps the suite
  collecting, per the Tier-1 `scipy` fix) and `release.yml` (build the arm64 + `any` wheels +
  sdist, clean-env `uv tool install ./dist/…` smoke → `steno doctor` green + `steno start
  --replay` pipeline smoke on a synthetic WAV, publish to PyPI via Trusted Publishing/OIDC).
  Acceptance: green both OSes; on a tag, a *different* clean Mac's `uv tool install stenograf`
  captures. `[dep: E1, E2]`
  *Status (2026-07-10): shipped and green on GitHub — ci.yml (macos-15 + ubuntu; macos-14
  runners are deprecated) and a full release.yml workflow_dispatch dry run (both builds +
  both clean-install smokes; publish correctly skipped without a tag). The macOS smoke
  installs the wheel, runs the new `steno setup --models-only` (headless model prefetch —
  TCC prompts can't be answered on a runner), requires `steno doctor` green, and drives the
  pipeline via `--replay` on a synthetic WAV; the build job asserts the wheel carries a
  signed `stenocap` with `minos 14.4`. Shaken out along the way: `build.sh` now pins
  `-target arm64-apple-macos14.4` (swiftc otherwise stamps the build host's OS as the
  minimum — a helper built on macOS 26 refused to launch on 14/15 despite the wheel tag),
  and `tests/test_asr_parakeet.py` gates MLX via `importorskip` (a plain import aborted
  Linux collection). Linux jobs were validated in a local container before pushing.
  **Publishing still needs one manual step by Daniel:** register the Trusted Publisher on
  PyPI (project `stenograf`, owner `daniel-om-weber`, repo `stenograf`, workflow
  `release.yml`, environment `pypi`); then tagging `v0.1.0` publishes.*
- **E5 — README install path.** Once E1–E4 land, the README's "Install from source" section
  reverts to `uv tool install stenograf`, and the pre-alpha status note drops the shipping
  caveat. (Written down because the README currently documents the source install *as* the
  install path, and that must not silently outlive the blocker.) `[dep: E4]`
  *Status (2026-07-10): shipped. **stenograf 0.1.0 is on PyPI** — Daniel registered the
  Trusted Publisher, tag `v0.1.0` ran release.yml green end to end (publish included,
  attestations attached to the GitHub release). Verified from PyPI itself: a clean
  `uv tool install stenograf` → doctor fully green, bundled helper reports `minos 14.4`
  and delivers live mic frames. README now leads with the PyPI path; `git+<repo>` is the
  documented pre-release channel.*

**Stage C — Web UI (`stenograf.web`).** *Follows Stage E.* The web view is "a new `LiveView`
+ a `serve()` twin, zero core changes" — confirmed against `view.py`/`tui.py`/`session.py`.
- **C1 — wire protocol + `WebLiveView` (start here; no server).** Pure event→JSON encoders
  in `web/protocol.py` (`encode_commit`/…/`encode_finalized` reusing the `Transcript.to_json`
  shape); `web/live.py::WebLiveView(LiveView)` overriding each event, marshaling onto the
  server loop via `loop.call_soon_threadsafe` (the `TextualLiveView._marshal` analogue) into
  a `CaptionHub` (per-connection `asyncio.Queue`s + retained backlog for late joiners).
  Acceptance: unit-test encoders; drive `WebLiveView` from a fake worker thread → subscriber
  gets ordered frames; before-ready/after-close drops (marshal-gate parity). `[dep: none new]`
- **C2 — server + `serve()` + `steno start --web`.** `starlette`+`uvicorn` deps;
  `web/app.py::create_app(hub, controls, security)` (`GET /` live shell, `Mount(StaticFiles)`,
  `WebSocketRoute("/ws")` — subscribe + inbound `{type:"stop"}` → `controls.stop()` on a bg
  thread, the `tui._invoke_stop` discipline); `web/server.py::serve(...)` the `tui.serve`
  twin (uvicorn on main thread, meeting on a bg thread, join before return); `web/static/`
  `live.js`+`app.css`. CLI: `--web` as a 4th branch in `_run_meeting`, precedence
  `--web > --plain > TTY→TUI > non-TTY→plain`. **Post-finalize: server stays up** and hands
  off to the reader (adopted rec). Acceptance: headless `TestClient` WS end-to-end + real
  `steno start --replay --web`. `[dep: C1]`
- **C3 — security (token + Origin/Host guard).** `web/security.py::mint_token()`; ASGI
  middleware / WS-accept hook enforcing the per-process token (header/query) + a
  `Host`/`Origin` ∈ {127.0.0.1, localhost}:port check (DNS-rebinding defense; token-in-header
  means classic CSRF doesn't apply). Bind `127.0.0.1` only; print the tokenized URL. Lands
  **before** any reverse-control POST. Acceptance: `TestClient` rejects missing-token /
  foreign-Origin, accepts token+loopback. `[dep: C2]`
- **C4 — live-view resilience + polish.** Late-join backlog replay, reconnect-on-drop,
  speaker colors, REC/elapsed header, animation-free finalize swap; **a browser disconnect
  must NOT stop the meeting** (a tab is detachable, unlike the TUI). Acceptance: `TestClient`
  drops+reopens the WS and converges; disconnect doesn't call `controls.stop`. `[dep: C2]`
- **C5 — archive list view.** `GET /meetings` + `GET /api/meetings` over the B1 index;
  `archive.js`. Acceptance: `TestClient` lists a seeded index. `[dep: C2, B1]`
- **C6 — transcript reader + click-to-jump (text).** `GET /meetings/{id}` +
  `GET /api/meetings/{id}` via A1; `reader.js` renders `<span data-start>` per word (click
  highlights/scrolls); `GET …/audio` streams the WAV **only** when `has_audio()`. Text-jump
  ships regardless of the audio decision. Acceptance: word spans carry timestamps; audio
  endpoint 404s cleanly with no recording. `[dep: C5, A1]`
- **C7 — reverse-control POSTs.** `POST …/refinalize` and `POST …/speakers/{label}/rename`
  consuming the B3/B4 interface; Task-C3 token+Origin applied; the "Detected: German, 2
  remote — [edit]" affordance. Acceptance: `TestClient` POST-with-token refinalizes a fake
  session; without token → 403. `[dep: C3, B3, B4]`
- **C8 — `steno serve` (archive-only) + asset packaging + docs.** A standalone server for
  Views 2/3 without starting a meeting (the everyday "browse my meetings"); package
  `web/static`+`web/templates` into the wheel; `steno doctor` web check. Acceptance: boots
  headless, `TestClient` lists+reads a seeded archive from the *installed* package. `[dep:
  C5–C7]`

**Stage D — meeting notes: pluggable LLM backend + note export (`stenograf.notes`).**
*Follows Stage E; parallel to C.* Opt-in, stdlib-only; fully local **when the Ollama
backend is chosen**. Nearly independent — only `steno notes` needs A1.

*Status (2026-07-10): **Stage D COMPLETE** — D1–D8 all shipped in one pass
(`stenograf.notes` package + `stenograf.settings`, `steno notes`, `--notes`,
doctor check; `tests/test_notes_*`, `test_settings`, `test_cli_notes`,
doctor additions). Verified against the real `claude` CLI end-to-end
(`STENOGRAF_NOTES_E2E=1` gates that test — it bills a real model call). Real
`ollama` e2e still pending a machine with Ollama installed. Remaining Phase 4
work: Stage C (web UI) only.*

*Redesigned 2026-07-10 (was "Ollama note-enhancement").* Driver: Daniel's production
meeting workflow (reference implementation: `~/.config/typewhisper/meeting-summary.sh`)
pipes the transcript through the **`claude` CLI** (Opus) because local models aren't
accurate enough for him yet, then writes a titled note (YAML frontmatter, summary,
collapsible transcript) into an Obsidian vault folder. Three lessons folded in:
1. **The LLM is a backend, not a dependency.** Mirror the `stenograf.asr` registry seam
   (`register_backend`/`create_backend`/`available_backends`): an `ollama` backend for
   fully-local users and a `command` backend that runs any configured CLI (prompt on
   stdin → schema JSON on stdout). Prompt building, chunking, schema, and parsing stay
   in stenograf, shared by all backends — providers are one line of config.
2. **Title is a schema field, not a `TITLE::` sniffing hack.** The LLM derives a title;
   it flows into the note filename and back into an untitled archive record.
3. **"Obsidian export" is just a markdown exporter.** A vault is a folder; the exporter
   writes one combined `{date} – {title}.md` note to any configured directory. No
   Obsidian-specific code; the callout syntax degrades gracefully elsewhere. TypeWhisper's
   detached-worker/nohup dance and `MIN_CHARS` gate are obsolete here — notes run in our
   own process, only when asked.

- **D1 — notes model + backend seam.** `notes/model.py::MeetingNotes` (**`title`**,
  summary, decisions, `ActionItem{task,owner,due,timestamp}`, `SpeakerHighlight`,
  open_questions + provenance **backend**/model/strategy/language) with
  `to_markdown`/`to_json`/`from_json`; `notes/backend.py::NotesBackend` protocol
  (`name`, `is_available()`, `complete(messages, schema) -> str`) + a registry mirroring
  `stenograf.asr.registry`; typed `NotesBackendUnavailableError`/`NotesGenerationError`.
  Acceptance: round-trip + markdown golden; registry register/create/unknown-name;
  **zero non-stdlib imports**. `[dep: none]`
- **D2 — Ollama backend.** `notes/ollama.py::OllamaBackend` over `urllib`
  (`is_available` via `/api/version`, `installed_models` via `/api/tags`,
  `chat(..., format=schema, stream=False)`), `ModelNotFoundError`; `OLLAMA_HOST`/
  `--ollama-url`; default model `qwen3:8b`. Acceptance: monkeypatched `urlopen` fakes
  the 3 endpoints. `[dep: D1]`
- **D3 — command backend.** `notes/command.py::CommandBackend`: run a user-configured
  argv (e.g. `["claude", "-p", …, "--output-format", "text"]`), rendered prompt +
  schema instruction on **stdin**, expect a JSON object matching `NOTES_SCHEMA` on
  **stdout** (tolerate surrounding prose/fences: extract the first top-level JSON
  object); configurable timeout (default 600 s); `is_available` = argv[0] resolvable
  on PATH; non-zero exit / no JSON / timeout → typed error, **never a partial**.
  Acceptance: fake-script fixtures (canned JSON, fenced JSON, garbage, exit 1, sleep >
  timeout); a real-`claude`-gated e2e (skipped when absent). `[dep: D1]`
- **D4 — prompt + chunking + generate (backend-agnostic core).**
  `notes/prompt.py::build_messages` (system role, respond-in-`transcript.language`,
  inject title/attendees/glossary, anti-hallucination: cite speaker+timestamp, never
  invent; optional user **instructions file appended** to — never replacing — the
  built-in system prompt), `chunk_entries` (whole-turn map-reduce for long meetings, no
  entry dropped), `NOTES_SCHEMA`; `notes/generate.py::generate_notes(transcript,
  backend, model=…)` (single-shot vs map-reduce, parse+validate schema JSON, stamp
  provenance, derive `title` when `profile.title` is None; never write a partial).
  Acceptance: `FakeBackend` canned JSON → populated notes; over-budget forces >1
  completion; unavailable backend → typed error, nothing written. `[dep: D1, A2]`
- **D5 — settings file.** `stenograf/settings.py`: `data_dir()/settings.toml` read via
  stdlib `tomllib`, first consumer is a `[notes]` table — `backend = "ollama"|"command"`
  (default `ollama`), `model`, `command = [argv…]`, `timeout_s`, `instructions =
  "<path>"`, `[notes.export] dir = "<path>"`. Precedence: CLI flag > env
  (`STENOGRAF_NOTES_BACKEND`, …) > settings.toml > built-in default. Missing file ⇒ all
  defaults; malformed TOML / unknown backend ⇒ one clear error naming the file.
  Machine-specific config lives **here, not in `MeetingProfile`** (profiles serialize
  into transcripts; a local argv must not). Acceptance: tmp-file precedence matrix;
  malformed-TOML message. `[dep: none]`
- **D6 — CLI: `steno notes <id|transcript.json>` + `--notes` flag.** New command
  accepting an **archive id or a path** (id → `MeetingArchive.load_transcript`, path →
  A1 loader); generate → `_write_notes` sibling `<stem>.notes.md`/`.notes.json`
  (via the existing `_atomic_write_text`); when the meeting is archived and its record
  is untitled, **write the derived title back to the index** (same path as
  reverse-control rename). An opt-in `--notes` flag on `transcribe`/`start` runs after
  `_write_transcript`, **non-fatal on failure** (warn, transcript stands, exit 0; rerun
  later via `steno notes <id>`). Never contacts any backend unless asked. Acceptance:
  `CliRunner` + fake backend; id and path forms; backend-down → clean message, no notes
  file, exit 0 with `--notes`, exit ≠0 for the dedicated command; title back-fill
  visible in `meetings list`. `[dep: D4, D5, A1, B]`
- **D7 — combined-note export (the Obsidian consumer).** `notes/export.py::
  export_note(transcript, notes, dir) -> Path`: one self-contained markdown note —
  YAML frontmatter (`title`, `date`, `created`, `source: stenograf`, `type: meeting`,
  `language`, `tags: [meeting]`), summary, decisions, **action items grouped by
  owner**, open questions, then the full speaker-labeled transcript in a collapsible
  `> [!quote]- Transcript` callout. Filename `YYYY-MM-DD – <title>.md`: strip
  `[]#^|`, replace `/:*?"<>\`, collapse whitespace, cap 80 chars, ` (n)` suffix on
  collision; atomic write; parent dir created. Wired as `--export-dir` on `steno
  notes`/`--notes` runs, defaulting from `[notes.export] dir` (unset ⇒ no export);
  `--no-export` opt-out. Export failure is non-fatal to the notes files. Acceptance:
  golden note; slug edge cases (umlauts kept, emoji kept, path chars gone); collision
  suffix; dir-with-spaces (iCloud vault paths). `[dep: D6]`
- **D8 — `doctor` notes check.** `_notes_check` resolves the configured backend and
  reports it: Ollama → reachable + model pulled; command → argv[0] on PATH. Uses a
  `Check.optional` field so an absent backend doesn't fail the overall `doctor` exit
  gate. Acceptance: monkeypatched backends; not-configured/not-running → not-ok but
  exit 0. `[dep: D2, D3, D5]`

**Ordering (revised 2026-07-10).** ~~A → (B ∥ C-live ∥ D) → C-consumers → E.~~
A and B are shipped. The remaining order is **E → (C ∥ D)**:

1. **E1 → E2 → E3 → E4** — the shipping path, in that order (E3 gates on E1; E4 on
   E1+E2). This is the current priority: it touches no runtime feature, and until it
   lands the tool cannot leave this checkout.
2. Then **C (web UI)** and **D (notes)** in parallel — both independent of each other.
   Within C: C1 → C2 → C3 → C4, with C5–C7 gating on the shipped Stage B and C8
   packaging the assets. Within D: D1 → (D2 ∥ D3 ∥ D4, with D5 anytime) → D6 → D7,
   D8 alongside. Daniel's own path needs only D1 → D3 → D4 → D5 → D6 → D7 (the
   Ollama backend D2 is for local-LLM users and can trail).
3. C8 re-touches packaging (the `web/static` assets), so re-run E4's clean-install
   smoke test after it; E1's build hook needs no change (assets are package data).

**Deferred to Phase 5 (Linux Track 2 — designed, not built).** A CPU/ONNX ASR backend
`stenograf/asr/sherpa.py::SherpaOnnxASRBackend` (`name="parakeet-onnx"`) wrapping the *same*
Parakeet-TDT-v3 int8 model with real per-token timestamps, registered through the existing
`stenograf.asr` factory (`create_backend` already the seam — zero CLI change; only
`default_backend_name()` goes platform-aware and two `doctor` strings change). **Open
Decision A:** whether the pinned `sherpa-onnx<1.13` (pin exists because 1.13.x macOS wheels
are broken) already yields Parakeet-v3 timestamps — if yes, **zero new dependency**; if it
needs 1.13.x, use `onnx-asr` (small MIT dep, isolated runtime, leaves the diarization pin
untouched) — probe first. A `LinuxCaptureProvider` (`stenograf/capture/linux.py`, in-process,
no helper): monitor discovery via `pactl`, capture via **SoundCard** (`include_loopback`) or
`parec`/`pw-record` subprocess (**Decision B** — prototype both; macOS is already
subprocess-based), 16 kHz mono direct (PipeWire resamples → no resampler dep), idempotent
thread-safe `stop()` like `MacOSCaptureProvider`. Known-count diarization already runs
ONNX/CPU via sherpa (Task = verification); *estimated* counts need the **stenodiar
port** — build speakrs without the `coreml` feature (ORT CPU/CUDA) and fix its
single-core ~1×-realtime CPU throughput first (details in §2 "Deferred task — stenodiar
on Windows/Linux"). **Decision C** (settled): finalize-first is first-class, live captions
best-effort with a CPU-RTF probe. Verification is label-free throughout (parakeet-onnx↔MLX
parity + timestamp sanity, reusing the Phase-2 agreement harness). Distribution then gains
the Linux pure-`any` wheel's dep markers and a Linux functional-transcription CI step.

*Settings portability (audited 2026-07-10):* `settings.toml` load/validate/show/edit is
already fully cross-platform — pure stdlib `tomllib`, `click.edit`, `os.replace` (atomic on
Windows too) — so Phase 5 inherits it as-is. Two small follow-ups when Windows becomes real:
`data_dir()` has no `win32` branch (data currently lands in `~/.local/share/stenograf`
instead of `%APPDATA%`; adding the branch implies a migration for early Windows users), and
backend-name validation is deliberately registry-level, not platform-aware (`backend = "mlx"`
validates anywhere; runnability is the backend's own check at use).

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
