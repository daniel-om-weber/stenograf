# Echo cancellation: evaluation & improvement plan

> **Status: complete (2026-07-10). All four tasks closed.** Tasks 1–2 shipped
> (measurement rig; dedup data-loss fix); Tasks 3–4 (energy gate, neural residual
> suppressor) were **closed as unnecessary** — a canceller with a live reference
> leaks nothing the ASR decodes. What remains open is capture-tap robustness, in
> §5 below. Read §1 for the measurements, §4 for how each task resolved.

Speakers + built-in mic is the default way to sit in a meeting, so remote voices
re-enter the mic and, untreated, get transcribed as `Local-N`. This document
planned the next iteration of the echo path: replace the text-dedup backstop with
an audio-domain gate, and build the measurement rig that every future change is
judged against. It extends PLAN.md §2 ("Hybrid-mode caveats"); the shipped state
it revised is `stenograf/aec.py` + `session.drop_echo_duplicates`.

## 1. Where we stand, and what the numbers mean

Two layers shipped 2026-07-09:

1. **WebRTC AEC3** (`stenograf.aec`, via livekit's `AudioProcessingModule`):
   near end = mic, far end = the process tap. Measured 36 dB ERLE synthetic,
   **30.5 dB live** on real acoustics.
2. **Character-coverage text dedup** (`session.drop_echo_duplicates`): drops a
   mic line whose text is ≥0.8-covered by an overlapping remote line.

Findings that reshape the plan:

- **~30 dB is the physics ceiling of the linear stage, not a tuning failure.**
  The process tap captures the digital mix *upstream* of the smart-amp
  speaker-protection DSP (nonlinear, time-varying excursion/thermal limiting on
  Apple Silicon), so the reference AEC3 adapts against is not what the speaker
  physically emitted. A linear filter cannot cancel distortion absent from its
  input; the ~−70 dBFS residual is that nonlinear remainder. Don't chase linear
  ERLE past 30 dB.
- **"Conference software cancels completely" is a misconception.** Zoom/Meet
  ship the same ~20–40 dB linear canceller (this architecture *is* Chrome's:
  AEC3 + system-loopback reference) followed by an aggressive residual echo
  suppressor that ducks the mic into comfort noise. That suffices for a human
  ear; it fails our requirements twice over: the ASR happily decodes −70 dBFS
  speech-shaped residue a human never hears, and the suppressor damages
  near-end speech during double-talk — exactly the overlapping speech a
  transcriber must keep. Their quoted 55–65 dB ERLE is far-end-single-talk only.
- **The text-dedup layer is a measured data-loss bug, not a safe backstop.**
  `_covered_by` normalizes by the mic line's length only, so a short local line
  that is a chance subsequence of a long remote monologue scores ~1.0: against
  a 56-word remote line, "no I don't think so" → 1.00, "yeah I think so" → 0.93;
  6 of 10 generic local utterances were destroyed, unrecoverably (the `.partial`
  checkpoint is deleted on clean finalize). It also false-positives on
  headphones (no acoustic path exists, dedup runs anyway; `--no-aec` disables
  only the canceller, not dedup), and it never protected the live view — echo
  lines display live and vanish only at finalize.
- **An audio-domain gate is viable, contra the aec.py docstring.** Genuine
  near-end speech is an independent source and *raises* post-AEC output energy.
  Measured through the real AEC3: echo-only output −58 dBFS vs double-talk
  −15.6 dBFS — a **42 dB gap**. (Coherence on AEC3's *output* is dead — its
  suppressor already stripped the coherent part — so gate on post-AEC energy,
  or coherence on the raw mic.)
- **The current stack has no knobs and no eyes.** livekit's
  `AudioProcessingModule` is four booleans; no `EchoCanceller3Config`, no ERLE
  stats. No PyPI package ships tunable AEC3 for macOS arm64, and tuning the
  suppressor trades double-talk transparency for suppression — the wrong trade
  here. So: keep AEC3 as-is, add stages after it, and build our own metering.

## 2. Target architecture

```
mic ──► AEC3 (linear, unchanged) ──► residual gate (energy) ──► [neural RES]* ──► ASR
tap ──► reference ────────────────────┘         │                     │
                                        (*only if the gate measurably isn't enough)
```

- **Residual energy gate** (Task 3): during far-end activity, post-AEC mic
  audio at the residual floor is replaced by silence before it reaches the ASR.
  Sits in the audio path, so it protects the live pass and the finalize pass
  identically. Headphone-safe by construction: with no echo, low-energy mic
  audio during playback is room noise that gates to silence harmlessly.
- **Neural residual suppressor** (Task 4, conditional): the principled tool for
  nonlinear residue. Candidate: **LocalVQE** (Apache-2.0 code, CC-BY-4.0
  weights, streaming DeepVQE derivative; 16 kHz mono, mic+reference input,
  16 ms latency, 49K–203K params, GGML CPU backend). Fallback: **DTLN-aec**
  (MIT, TF-Lite). Adopt only if it beats the gate on the rig *and* its
  double-talk near-end degradation is nil — an aggressive RES eating local
  speech is the failure mode that disqualifies.
- **Text dedup**: demoted to a diagnostic once the gate ships; removed when the
  rig shows zero leaked lines without it. Until then its data-loss bug is fixed
  (Task 2) because it destroys real transcript lines today.

## 3. Measurement (before any behavior change)

Three layers, all automated — no hand-labeling.

**Layer 0 — signal.** `--aec-dump DIR` writes three clock-aligned mono 16 kHz
WAVs per session: `mic.wav` (raw near end), `lpb.wav` (loopback/tap reference),
`enh.wav` (post-AEC, post-gate mic — what the ASR hears). Opt-in like
`--record-audio`, since it writes audio to disk. On these triples,
`eval/aec_score.py` computes:

- **ERLE** over far-active/near-silent spans (energy-based; waveform
  correlation is useless on AEC3 output — it is fractionally delayed).
- **AECMOS** via the `speechmos` package: the AEC-Challenge metric, scoring
  *echo annoyance* and *near-end degradation* separately, with a double-talk
  mode. The degradation score is the guard on every suppression stage.

**Layer 1 — does residue become text (the metric that matters).**
`eval/aec_rig.py` orchestrates repeatable runs on real hardware: play a fixed
far-end WAV out the speakers while the real pipeline captures with
`--aec-dump`, then score. Scenarios:

| scenario | far end | near end | pass criteria |
|---|---|---|---|
| far-only | fixed WAV | silence | 0 `Local-N` lines ≥3 words |
| near-only | silence | scripted speech | local WER ≈ speakers-muted baseline |
| double-talk | fixed WAV | scripted speech | local lines survive; AECMOS-DT degradation ≈ nil |

Scripted near end is played from a second device at fixed position, so runs are
reproducible without a human performing each one. Report far-only leakage both
pre- and post-backstop so the canceller and the gate are measured separately.

**Layer 2 — backstop false positives.** Adversarial fixture: the local speaker
repeating what the remote just said within the dedup window ("so you're saying
we should ship Friday…") — any surviving whole-line text matcher must not
delete it.

**Scenario matrix** (the echo path is not one thing): speaker volume 50/75/100 %
(smart-amp nonlinearity grows with level), Bluetooth speaker (large variable
delay), headphones (zero drops, zero gating), German + English, music as far
end, device switch mid-session. For regression breadth beyond this one MacBook:
the Microsoft AEC-Challenge dataset (real recordings, 10k+ devices, genuine
nonlinear echo, double-talk) replayed through `--replay`.

Also surfaced by the dump: `far_end_missing_ticks` (exists, currently
unobservable) and the known long-session tap failure where PCM goes all-zeros —
which would silently blind the canceller.

## 4. Tasks, in order

1. **Measurement rig** — `--aec-dump`, `eval/aec_score.py` (+ `speechmos` in
   the eval group), `eval/aec_rig.py`. Cheapest item; de-risks everything else.
   Acceptance: one command produces scored far-only / near-only / double-talk
   results on this machine. **DONE 2026-07-10.** Its first far-only run caught
   a bug this plan didn't predict: the batch tail-checkpointer busy-spun on
   `AudioBus.wait`, starved the capture thread, and Core Audio killed the
   system tap ~3 s into every real-hardware `--no-live` meeting — the
   canceller lost its reference and leaked everything (fixed in `ebf660a`;
   regression-tested). After the fix, far-only measures **37.6 dB ERLE live,
   −65 dBFS residual, AECMOS echo 4.73/deg 5.00, and 0 leaked lines before any
   text backstop** (vs −27 dBFS raw / echo 1.49 uncancelled). The historical
   "AEC leaks lines" evidence predates this fix and needs re-measuring.
2. **Fix the dedup data loss** — normalize coverage against the aligned remote
   span, not the whole remote line; `--no-aec` disables dedup; dedup skipped
   when no echo path exists. Acceptance: the measured false-positive utterances
   survive; the original leaked-echo fixtures still drop. **DONE 2026-07-10**
   (`c2795f4`): span-density normalization drops the chance-subsequence scores
   from 0.80–0.95 to 0.08–0.16 while real echoes stay at 0.89–1.00, and
   `--no-aec` now disables dedup entirely.
3. **Post-AEC energy gate** — in `stenograf.aec`, behind the same `--aec` flag.
   Threshold placed with rig data (the 42 dB gap), not hand-tuned feel.
   Acceptance: far-only leakage 0 lines pre-dedup; near-only WER unchanged;
   double-talk AECMOS degradation unchanged. **CLOSED as unnecessary
   2026-07-10.** The full scenario matrix ran clean with no gate: far-only at
   volume 63 and 100 % (37.6 / 33.0 dB ERLE, 0 leaks), far-only under live
   inference load (0 leaks ≥3 words), double-talk (0 leaks, 0 false drops,
   local speech transcribed throughout), and Bluetooth after the aggregate-rate
   fix (`7dd1510`; 28.1 dB ERLE, 0 leaks). A healthy canceller's residual
   simply does not decode. What *does* leak is a canceller that lost its
   reference — two capture bugs proved it — so the shipped mitigation is the
   **armed backstop** (`3d079cb`): `drop_echo_duplicates` runs only when
   `far_end_missing_ticks > 0` (or the canceller was unobserved), and the CLI
   warns with cause and drop count when it acts. Healthy meetings never run
   it, so a verbatim local repeat can never be deleted.
4. **Neural RES spike (conditional)** — only if (3) leaves leakage. **CLOSED
   with (3)**: no decodeable residual to suppress. LocalVQE/DTLN-aec remain in
   §5 as the escalation path if a future device class measures differently.

## 5. Open items (the echo path is settled; the tap that feeds it is not)

Both tasks 3 and 4 closed on the same finding: **a canceller with a live reference
does not leak.** Every measured leak came from *losing* the reference. That makes
tap robustness — not suppression — the remaining work. Neither item below is
scheduled; both are cheap, and either would silently reintroduce echo lines.

1. **The tap dies on any Python-side stall >~1 s, permanently, with no recovery.**
   `stenocap`'s 64 KB stdout pipe fills, Core Audio kills the tap, and nothing
   restarts it. Two separate bugs have already reached production through this
   path (the tail-checkpointer busy-spin, `ebf660a`; the aggregate-rate mismatch,
   `7dd1510`). A **drain thread in `MacOSCaptureProvider`** — reading the pipe
   into a queue independently of the consumer — decouples capture from every
   downstream stall. Highest-value hardening in the capture layer.

2. **An all-zero tap is undetected.** `far_end_missing_ticks` (`aec.py:235`)
   increments only when a far-end frame is **absent**. The known long-session
   failure where the tap keeps delivering frames of silent PCM therefore leaves
   the counter at 0: the armed text backstop never arms, the CLI never warns, and
   AEC3 adapts against silence while echo passes straight through to the ASR —
   the exact failure the backstop exists to catch, in its quietest form. Fix: an
   energy check on the far-end tick (a reference that is bit-exact zero for many
   consecutive seconds *while the near end is not* is a dead tap, not a quiet
   meeting), feeding the same `reference_gap_s` signal.

## 6. Sources

- livekit APM surface: `livekit-rtc` `apm.py` (four booleans; no config/stats).
- AEC3 suppressor internals & config: `api/audio/echo_canceller3_config.h`;
  switchboard.audio "How WebRTC AEC3 works".
- Smart-amp DSP downstream of the tap: Apple loudspeaker-protection patents
  (US10015593, US9525945, US10219074); tap-based EQ tools documenting
  post-tap limiting.
- LocalVQE: github.com/localai-org/LocalVQE (weights: HF `LocalAI-io/LocalVQE`).
- DTLN-aec: github.com/breizhn/DTLN-aec (ICASSP 2021 AEC Challenge, 3rd).
- AECMOS / dataset: github.com/microsoft/AEC-Challenge; `speechmos` on PyPI.
- Double-talk detection basis: Benesty–Morgan–Cho (2000); Gänsler (1996).
