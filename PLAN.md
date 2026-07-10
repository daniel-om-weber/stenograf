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
│         optional: meeting notes via a pluggable local-LLM backend (stenograf.notes)│
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
retained), plain-text prose, and SRT/VTT export. Post-meeting notes shipped as
`stenograf.notes` (Phase 4 Stage D): pluggable LLM backends — in-process MLX
(the Apple-Silicon default), Ollama, or any configured CLI.

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
  ad-hoc-signed if ever wanted. *(Superseded 2026-07-10: the Textual TUI is the
  UI; the web UI was dropped with the pipeline de-scope — see §3 product
  philosophy and §5 Stage C.)*
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
- **Distribution:** PyPI + uv only; no Developer ID. *(The original "local web
  UI direction" was dropped 2026-07-10 — see the product philosophy below.)*
- **Name: `stenograf`** — German spelling of stenographer, the verbatim
  minute-writer. Package `stenograf` (confirmed free on PyPI), CLI `stenograf`
  with `steno` alias.
- **Product philosophy (added 2026-07-10): a pipeline, not a manager.**
  stenograf's responsibility ends at producing text — the full transcript and
  the notes/summary, written into a user-visible folder. Managing, re-reading,
  and listening to past meetings belongs to other tools (Obsidian via the note
  export, Finder, any audio player). Machine state (voiceprints, settings,
  model cache) stays in the app data dir; user documents do not. Feature
  requests that add management, browsing, or playback should be declined or
  pointed at the exporters.

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

**Status (2026-07-10): Phases 0–4 are complete.** stenograf 0.1.0 is on PyPI
(`uv tool install stenograf`); Stage C — the final Phase 4 stage, re-scoped
twice on 2026-07-10 (web UI → reverse-control CLI → the final **de-scope to a
pipeline**) — shipped the same day: outputs moved to a visible folder and the
meeting-management layer (index, `meetings` group, archived reverse control)
was retired; the web UI is dropped outright. Phase 5 (Linux) is designed but
deferred. The per-task build logs of the completed phases were removed from
this file on 2026-07-10; they live in its git history (and in PLAN-AEC.md for
echo cancellation).

### Completed phases

**Phase 0 — Model validation. COMPLETE.** Blind adjudication of model
disagreements on real meeting audio (de+en) confirmed **Parakeet-TDT-0.6B-v3 as
the default ASR**; Whisper large-v3 is the fallback, Voxtral and Canary demoted.
Results and methodology in §2 "Finalize ASR"; harness at `eval/adjudicate.py`.

**Phase 1 — Batch MVP. COMPLETE.** The signed Swift capture helper
(`native/helper/`, **stenocap**: system tap + mic → framed PCM on stdout),
per-channel in-RAM `SessionStore`, and the finalize pipeline (Silero VAD →
parakeet-mlx → sherpa diarization → merged `Local-N`/`Remote-N` transcript)
behind the swappable `CaptureProvider` interface. `steno start` and
`steno transcribe <file>`, the opt-in `--record-audio` WAV tee, incremental text
checkpoints, the `--replay` file provider, and text-vote de/en language
detection (`stenograf.lid`).

**Phase 2 — Live captions. COMPLETE.** `LiveDecoder` (12–16 s re-decode window
over the full `generate()` path, LocalAgreement-2 commits, VAD-gated), O(window)
`SessionStore.view`, the `AudioBus`/`CaptureLoop`/`LiveWorker` orchestration
(single-flight inference; load-shedding degrades an over-realtime backlog to a
caption gap the finalize fills), Option-B checkpointing (the committed live text
is the `.partial` crash checkpoint; batch mode uses a tail-only finalize
checkpointer), `PlainLiveView` + the Textual TUI, and the CLI glue
(`--live/--no-live`, `--plain`, `--flush-interval`). Measured on de-1/en-1:
agreement-with-finalize WER 7.0%/5.6%, zero monotonicity violations, ~2.5 s
median commit latency, ~7–10% of one accelerator during speech and ~0% in
silence. Label-free live-pass eval harness at `eval/live.py`. Hard-won lesson:
MLX weights must be materialized on the load thread, or background-thread
inference dies ([[mlx-weights-thread-local-streams]]).

**Phase 3 — Speaker polish + vocabulary + auto-detection. COMPLETE.**
- *Stage 0 hardening:* silent-channel finalize guard, double-quit
  transcript-loss fix, first real-backend sherpa test, DER/word-attribution
  scorer (`eval/rttm.py`, `eval/der.py`, `eval/diarize.py` — references never
  hand-labeled, see open items), word timestamps retained on
  `TranscriptEntry.words`, `LiveWorker` load-shedding.
- *Stage 1 speaker re-ID:* `diarize_with_embeddings` (eres2net, per-cluster
  duration-weighted mean), `ProfileStore` (model-bound voiceprints, atomic JSON
  in the platform data dir), match-only greedy one-to-one relabel in the
  finalize pass, the `steno profiles` group (enroll/list/rename/remove,
  `--reinforce`, multi-speaker `--speakers`/`--speaker` enrolment), and the
  `--reid/--no-reid`/`--reid-threshold` wiring.
- *Stage 2 export & vocabulary:* SRT/VTT cue re-flow from word timestamps
  (time-overlapping per-speaker cues allowed, each labeled), and
  glossary/attendee post-correction (`stenograf.glossary`: difflib +
  umlaut/accent folding, word timing and punctuation preserved) behind the
  `--glossary`/`--glossary-file`/`--attendee`/`--glossary-threshold` family.
  Parakeet has no decode-time prompt, so post-correction is the honest lever
  ([[phase3-verified-library-constraints]]).
- *Stage 3 auto-detection:* local-speaker-count estimation (the mic channel
  passes `None` through like the system channel), detected counts surfaced as
  editable with a lock/correct re-run hint, and parameter provenance
  (`explicit|detected|default`) serialized into the transcript JSON.

**Phase 3→4 readiness audit + Tier-1 hardening. COMPLETE.** Four-subagent audit
verdict: architecturally ready for Phase 4 (the `CaptureProvider` and `LiveView`
seams are real and terminal-free). Tier-1 fixes shipped: thread-safe idempotent
capture teardown with the `serve()` template locked down (capture errors no
longer abort a finalizable buffer), the `stenograf.asr` lazy backend registry,
Linux test collection (`scipy` declared) + a model-free sherpa unit test, atomic
writes for every transcript/checkpoint artifact, and the small web-UI landmines
(`SpeakerProfile` hashability, `--local 0 --remote 0`, clamped count hints).

**Phase 4 Stages A, B, D, E — SHIPPED** (summaries under the build plan below).

**Phase 5 — Linux + cross-platform ASR. DEFERRED (designed, not built).** Linux
in-process capture (PipeWire/PulseAudio monitor) + a CPU/ONNX Parakeet backend
through the already-shipped `stenograf.asr` factory; full sub-plan under
"Deferred to Phase 5" at the end of the Phase 4 build plan.

### Open items & known deferrals (standing)

- **0d hand-labelled RTTM references** — the DER/word-attribution scorer is
  built and tested, but no references exist (Daniel's call not to hand-label),
  so diarization/re-ID quality stays unmeasured: the re-ID threshold stays at an
  untuned 0.5 and far-field local-speaker-count estimation is known to
  over-split (a small group measured as 8).
- **Capture-tap fragility (two open defects, see PLAN-AEC.md §5).** (1) Any
  Python-side stall over ~1 s permanently kills the Core Audio tap with no
  recovery — a drain thread in `MacOSCaptureProvider` would decouple it; two
  separate bugs have already been traced to this. (2) A tap that keeps
  delivering **all-zero** PCM is undetected: `far_end_missing_ticks` counts only
  *absent* far-end frames, so the armed text backstop never arms and no warning
  fires while the canceller runs blind.
- **stenodiar on Windows/Linux** — blocked on speakrs' single-core ~1×-realtime
  ORT CPU throughput; details in §2 "Deferred task — stenodiar on
  Windows/Linux".
- **Lower-priority, independent:** greedy re-ID → optimal (Hungarian)
  assignment; SRT/VTT dropping text not covered by `words` (latent — Parakeet
  emits full-or-none); helper-stderr piping; meeting-mode auto-detect; hybrid
  cross-channel dedup; acoustic first-segment LID for the live pass; a real
  Ollama notes e2e (needs a machine with Ollama installed).

---

### Phase 4 build plan — product layer + macOS distribution (Linux → Phase 5)

Planned July 2026 by a five-subagent design pass (web UI · persistence/archive ·
Linux backends · notes · distribution); all five stages are shipped and
summarized below. Stage C was re-scoped twice on 2026-07-10 — web UI →
reverse-control CLI → the final **de-scope to a pipeline** (below); the web UI
is dropped (its full W1–W8 design lives in this file's git history).

**Locked scope decisions (Daniel, July 2026):**
- **Product layer first; Linux deferred to Phase 5.** Phase 4 = a tangible
  Mac-native product (web UI + archive + notes) + the macOS shipping path.
  *(Amended 2026-07-10, twice: the web-UI leg was first re-scoped to a
  reverse-control CLI, then the whole management layer was de-scoped — see
  Stage C. The browser view is dropped; the TUI and the Obsidian note export
  cover its use cases.)*
- **`steno start` writes into a managed archive dir by default** —
  `data_dir()/meetings/<id>/transcript.*`; `--out PATH` overrides and still
  registers. Makes the archive an authoritative library, not an index over
  scattered files. *(Superseded 2026-07-10 by the Stage C de-scope: outputs
  move to a visible folder and the index is retired — the filesystem is the
  library.)*
- **In-RAM-only privacy guarantee preserved.** Audio never touches disk unless
  `--record-audio`. Text click-to-jump is *always* available (word timestamps
  live in the JSON); archive audio **playback** and archived **re-diarize** are
  opt-in, gated on one `record.has_audio()` predicate.

**Adopted recommendations still standing:** macOS signing stays **ad-hoc
only**; platform deps via **markers, not extras**; Windows **left installable**
with an honest `doctor`. (The notes-side recommendations shipped with Stage D,
below; the web-server recommendations — Starlette + uvicorn, vanilla-JS
no-build front-end, per-process token + Host/Origin guard — went to git history
with the dropped web UI.)

**Evaluation stays label-free** (Daniel's standing no-hand-labels call):
round-trip / property tests, fakes + headless `TestClient`, real-backend
end-to-end via `--replay`, and real-backend-gated e2e (Ollama and/or a real
`claude` CLI, skipped when absent) mirroring the model-gated ASR tests. No
accuracy scoring.

**Stage A — shared foundations: SHIPPED.** `Transcript.from_json` +
`SCHEMA_VERSION` (a faithful round-trip loader — the keystone the archive, the
web reader, and notes all build on; legacy/newer-version/unknown-key compat
rules included) and `MeetingProfile.title`.

**Stage B — persistence (archive + reverse control): SHIPPED.**
`stenograf.archive`: `MeetingArchive`/`MeetingRecord` with an atomic
`index.json` at `data_dir()/meetings/`, stable `meeting-YYYYMMDD-HHMMSS` ids
(collision-suffixed), a `reconcile()` self-heal (drops vanished dirs, adopts
orphans), `load_transcript(id)`, and the `has_audio()` predicate that gates
everything audio-dependent. CLI writes into the managed archive by default
(`--out` registering override, `--no-archive` flat-file escape hatch,
`--title`), with the `meetings list/show/rm` group. Reverse control:
`stenograf.control.MeetingSession` + `FinalizeRequest` (re-finalize with
sticky per-field overrides on the warm backends — counts/language/reid — plus
pure `rename_speaker`) and `ArchivedMeeting` (rename always works; refinalize
gated on `has_audio()`, rehydrating a per-channel store from the recorded WAV
and delegating to `MeetingSession`; persists via `MeetingArchive.rewrite`).
*(2026-07-10: the Stage C de-scope retires most of this layer — the index, the
`meetings` group, and the archived reverse control; what Stage B built and its
full build log remain in git history.)*

**Stage D — meeting notes: SHIPPED** (`stenograf.notes` + `stenograf.settings`;
verified against the real `claude` CLI via the `STENOGRAF_NOTES_E2E=1`-gated
test; a real-Ollama e2e is still pending). As built — **three** pluggable
backends behind an asr-style registry, one more than the original two-backend
design:
- **`mlx`** — in-process `mlx-lm`, **the zero-setup default on Apple Silicon**
  (chosen by `default_backend_name()` whenever `mlx_lm` imports); Qwen3 with
  thinking mode on by default; generation bound to the mlx_lm-import thread.
- **`ollama`** — stdlib-`urllib` HTTP to `localhost:11434`, default model
  `qwen3:8b`; no pip dependency.
- **`command`** — any configured CLI taking the prompt on stdin and emitting
  schema JSON on stdout (e.g. `claude -p`; Daniel's production path), typed
  errors, never a partial.

Shared core: the `MeetingNotes` model (title/summary/decisions/action-items/
highlights/open-questions + backend/model provenance), prompt builder +
whole-turn map-reduce chunking (the single-shot budget is a backend property),
schema-validated parsing, `steno notes <id|path>` + a non-fatal `--notes` flag
on `start`/`transcribe`, sibling `.notes.md`/`.notes.json` files, LLM-derived
titles back-filled into untitled archive records, combined-note export
(`[notes.export] dir` — Obsidian-friendly frontmatter + collapsible transcript),
and a `doctor` notes check. `settings.toml` outgrew the planned notes-only
scope into **six tables** (`[transcript]`, `[vocab]`, `[archive]`, `[speakers]`,
`[asr]`, `[notes]` + `[notes.export]`) with a `steno settings show/edit`
command; the library never reads settings — the CLI resolves
flag > env > file > default ([[settings-toml-architecture]]).

**Stage E — macOS distribution: SHIPPED. stenograf 0.1.0 is on PyPI;
`uv tool install stenograf` verified from a clean environment.** The
`hatch_build.py` wheel hook compiles + ad-hoc-signs stenocap into the tagged
arm64 wheel (pure `any` wheel elsewhere; fails loudly if `swiftc` is missing on
arm64), `requires-python <3.14` + a numba floor fix the resolver traps, atomic
model extraction, the signature verified through the wheel zip round-trip,
`steno setup` (one-time TCC prompts + model prefetch; `--models-only` for
headless CI), a CI matrix (macos-15 + ubuntu) and `release.yml` with
clean-install smoke tests and Trusted-Publishing. Release procedure: bump the
version, tag `vX.Y.Z`, push — release.yml does the rest.

**Stage C — de-scope to a pipeline: SHIPPED (2026-07-10; final re-scope, after
web UI → reverse-control CLI).** Decision (Daniel): stenograf's responsibility
ends at producing text — the full transcript and the notes/summary. Managing,
re-reading, and listening to recorded meetings is other tools' job (Obsidian
via the Stage D export, Finder/`ls`, any audio player), and **there is no
index at all** — the filesystem is the index. As built:

- **Visible output home** (`stenograf.output`): every run writes its own
  `meeting-YYYYMMDD-HHMMSS/` folder (on-disk collision suffixing; nothing is
  created until the first write) under `~/Documents/Meetings`, configurable
  via the `[output] dir` setting (the `[archive]` table is gone — a stale one
  gets a targeted rename error); `--out DIR` uses DIR itself as the meeting's
  folder. Plainly named `transcript.*`, notes siblings, optional `audio.wav`,
  and the `.partial` checkpoint all land together. Machine state (voiceprints,
  settings.toml, model cache) stays in the data dir — user documents do not.
- **Deleted:** `archive.py` (`MeetingArchive`/`MeetingRecord`/`reconcile`/
  `index.json`), the `meetings list/show/rm` group, `--archive/--no-archive` +
  the legacy flat layout, the index-side title back-fill (the exported note's
  filename carries the LLM title), and the orphaned reverse-control layer —
  `control.py` (`MeetingSession`, `ArchivedMeeting`, `AudioUnavailable`) and
  `pipeline.rename_entry_speaker`. `recording.read_channels` stays (the AEC
  eval rig consumes it). Re-processing a recorded meeting is what it always
  was: `steno transcribe <folder>/audio.wav --speakers N`.
- **`steno notes`** takes a meeting folder or transcript path, or `--last`
  (newest finished `meeting-*` folder in the output home, by name scan —
  crashed runs without a `transcript.json` are skipped).

Acceptance held: label-free CLI tests (outputs land in the visible home,
`--last` picks the newest, no index file is ever written) plus a real-backend
`--replay` e2e into a configured home. Everything deleted remains in git
history.

**Web UI: dropped (2026-07-10).** With no archive to browse there is nothing
left for a browser to show that the files themselves don't, and the TUI covers
live captions. The deferred design (W1–W8: Starlette server + token/Origin
security + archive/reader views + `steno serve`) lives in this file's git
history should it ever be wanted.

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
