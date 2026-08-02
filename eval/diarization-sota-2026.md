# Speaker diarization & voice-profile re-ID — state of the art, August 2026

Research record for `PLAN-DIARIZATION.md`. Compiled 2026-08-02 from three
parallel web-verified research passes (diarization systems; speaker-embedding
models, scoring and enrollment; integrated pipelines and evaluation). Every
load-bearing number carries its source. Numbers quoted from primary sources
(papers, model cards, repos) are stated plainly; vendor claims and
extrapolations are marked as such.

What stenograf ships today, for comparison: per-channel diarization (mic and
loopback separately, never mixed, often with an exact per-channel speaker
count), via sherpa-onnx on CPU — pyannote **segmentation-3.0** + **ERes2Net**
VoxCeleb embeddings + agglomerative clustering at threshold 0.5
(`diarization/sherpa.py`) — and the pyannote **community-1** pipeline via
CoreML (vendored speakrs) on macOS. Cross-meeting re-ID: one duration-weighted
mean embedding per cluster, cosine-matched against a local named-profile store
at a flat 0.5 threshold (`profiles.py`).

---

## 0. How to read DER numbers

Three incompatible scoring conventions circulate; mixing them produces 5–10
point phantom deltas:

| Family | Definition | Used by |
|---|---|---|
| **A — strict** | 0 s collar, overlap scored, fully automatic | pyannote's tables, DiariZen's tables, Sortformer's DIHARD/AMI rows |
| **B — collar** | 0.25 s collar, overlap scored | ETH benchmark (arXiv:2509.26177), Sortformer CALLHOME rows |
| **C — lenient** | 0.25 s collar, overlap excluded | FluidAudio, SpeechBrain recipes |

Family A is the only one where cross-system comparison is defensible, because
pyannote, DiariZen and NVIDIA all publish it. Our own `eval/der.py` defaults
to a 0.25 s collar with overlap scored (family B); compare accordingly.

## 1. Diarization ("who spoke when")

### 1.1 The comparable results (family A: 0 s collar, overlap scored)

| System | AMI IHM | AMI SDM | DIHARD3 | VoxConverse | AliMeeting | Weights license | Gated |
|---|---|---|---|---|---|---|---|
| pyannote 3.1 (≈ our baseline family) | 18.8 | 22.7 | 21.4 | 11.2 | 24.5 | MIT | yes |
| pyannote community-1 | 17.0 | 19.9 | 20.2 | 11.2 | 20.3 | CC-BY-4.0 | yes (auto) |
| Sortformer streaming v2.1 (30 s latency) | 15.90 | 17.80 | 19.49 | — | 13.55 far | NVIDIA OML | no |
| **DiariZen meeting-base** | — | **15.6** | — | — | 17.7 | **MIT** | **no** |
| DiariZen wavlm-large-s80-v2 | — | 13.9 | 14.5 | 9.1 | 10.8 | CC-BY-NC ✗ | no |
| EEND-TA (Interspeech 2025) | 11.04¹ | 15.16 | 14.49 | 14.29 | — | never released ✗ | — |
| pyannote precision-2 (paid API) | 12.9 | 15.6 | 14.7 | 8.5 | 15.2 | proprietary ✗ | — |
| 3D-Speaker pipeline | — | 21.76 | — | 11.75 | 19.73 | Apache-2.0 | no |

¹ author-reported "AMI-Mix", no code or weights — a research ceiling, not a
candidate.

Sources: <https://huggingface.co/pyannote/speaker-diarization-community-1> ·
<https://github.com/BUTSpeechFIT/DiariZen> ·
<https://huggingface.co/BUT-FIT/diarizen-meeting-base> ·
<https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1> ·
<https://arxiv.org/abs/2509.14737> (EEND-TA)

**Headroom conclusion:** the field is not plateaued — open meeting-audio DER
roughly halved 2023→2026, driven by WavLM-based EEND+clustering hybrids. The
permissive ladder on AMI SDM: pyannote 3.1 = 22.7 → community-1 = 19.9 →
DiariZen meeting-base = **15.6** (matching the paid precision-2 tier). Against
DiariZen's own pyannote-3 run (21.1 AMI SDM) that is a **5.5-point gain**.
Three caveats shrink it for us: our exact stack (seg-3.0 + ERes2Net + AHC, no
VBx) is likely somewhat *worse* than pyannote 3.1 proper, so real headroom is
larger; our near-field per-channel known-count audio is easier than every
benchmark, so absolute deltas compress; and diarizen-meeting-base is trained
only on far-field SDM meeting corpora — near-field generalization unmeasured.

### 1.2 The one independent evaluation (ETH Zurich, arXiv:2509.26177, 2025-09)

Out-of-the-box, no tuning, 196.6 h, 0.25 s collar, overlap scored. DER by
speaker count (aggregated over CALLHOME/VoxConverse/AMI/AliMeeting):

| System | 1 spk | 2 spk | 3 spk | 4 spk | 5+ spk |
|---|---|---|---|---|---|
| pyannoteAI (paid) | 2.7 | 9.9 | 9.1 | 10.1 | 6.6 |
| DiariZen (large-s80) | 2.3 | 11.4 | 10.3 | 12.7 | 7.1 |
| Sortformer v2-stream | 4.7 | 10.4 | 14.1 | 13.2 | 22.7 |
| pyannote 3.1 | 3.2 | 19.9 | 19.8 | 17.1 | 10.6 |

Two findings we build on:

- **Speaker count dominates DER.** 1-speaker audio still costs 1.5–4.7 % DER
  (essentially all missed speech); 2-speaker ~10 %; 4+ speaker 13–24 %.
  EEND-EDA measured the known-vs-unknown-count penalty directly: CALLHOME
  2-speaker 8.07 % vs unknown-count 15.29 % — ~2× (arXiv:2005.09921). Our
  per-channel split with known counts is a structural advantage worth more
  than any model swap.
- **Missed speech dominates every system's DER, and it is onset/offset
  boundary imprecision (~350 ms average), not wholly-missed segments.** This
  is why collar choice swings numbers so violently, and it means boundary
  handling — not a new model — is where the next points are. Corroborated
  by pyannote 3.1's own AMI decomposition (DER 18.8 = FA 3.6 + **Miss 9.5** +
  Conf 5.7) and TST-Bench (miss = 72 % of DER, §5.3).

Sortformer is also notably weak on English (14.1 vs DiariZen 7.0) despite
winning Mandarin/Japanese.

### 1.3 Overlap

AMI is ~16 % overlapped speech, AliMeeting ~19 %, CALLHOME ~12.6 %
(arXiv:2509.26177 §2.3). pyannote segmentation-3.0 attributes overlap via its
powerset head (max 3 speakers per 10 s window, max 2 simultaneous —
arXiv:2310.13025); Sortformer natively (4 sigmoids); plain VAD→cluster
pipelines (3D-Speaker default, FunASR, senko) not at all — their Miss floor
equals the overlap fraction. pyannote 4.0 deleted its resegmentation and
overlapped-speech-detection pipelines; there is no cheap bolt-on overlap
resegmenter in 2026 — overlap must come from the segmentation model.

**Verified in sherpa-onnx source**: the pyannote pipeline builds the full
powerset mapping (overlap decoded into turns) but applies `ExcludeOverlap`
before embedding extraction — overlap frames inform segmentation output but
are excluded from clustering embeddings. Our own `cluster_embeddings()`
re-slices audio by turn times and does NOT exclude overlap; see §4.4 for why
it should.

### 1.4 Does a known speaker count help?

- Clustering pipelines: yes, mechanically — sherpa's
  `FastClusteringConfig(num_clusters=k)` switches threshold-cut to k-cut. The
  lever we already hold.
- pyannote `num_speakers`: benefit unproven and possibly negative —
  arXiv:2308.02160 measured forcing the true count improving speaker-change
  detection +17.2 % relative while **worsening DER 9.3 % relative** (forcing k
  cannot recover missed speech and can produce a worse partition). See also
  pyannote discussion #1726.
- Sortformer cannot consume a count at all (hard 4-speaker architectural cap;
  DER roughly triples at ≥5 speakers: 13.24 → 42.56 on DIHARD III).

Consequence: known-count conditions structurally favor clustering pipelines
and handicap end-to-end models. Combined with §5.1, this closes the
architecture question.

### 1.5 Compute (60-minute meeting, laptop CPU)

| Stack | Size | CPU RTF | 60 min → | Source |
|---|---|---|---|---|
| stenograf today (seg-3.0 + ERes2Net + AHC, 8 ORT threads) | 32 MB | ~0.13 | ~8 min | measured in-repo (`sherpa.py:26-31`) |
| same models, stride 3 + per-chunk embedding | 32 MB | ~0.22 (M4, 1 thread) | ~13 min → ~1–2 min at 8 threads | arXiv:2606.08505 Table III |
| DiariZen meeting-base (WavLM-Base+ + Conformer) | ~400 MB | ~0.2–0.4 (extrapolated, unverified) | ~12–24 min | DiariZen-large 0.3 CPU RTF |
| Sortformer v2/v2.1 | 471 MB | unpublished — "not currently optimizing on any CPU models" | unknown | HF discussion, nvidia/diar_sortformer_4spk-v1 #10 |
| speakrs/FluidAudio CoreML (our macOS path) | 19 MB | ~0.002 | ~7 s | M4 Pro |

**The two measured engineering speedups** (two independent sources —
SDBench/Argmax, Interspeech 2025; arXiv:2606.08505):

- Segmentation sliding-window stride 1 s → 3–4 s: 8.5–38× speedup; DER cost
  ≈0.02 for ≤5 speakers. Failure mode: coarser stride under-counts speakers
  in-the-wild because fewer embeddings hit pyannote's `min(12, round(0.1·n))`
  minimum-cluster-size floor; setting `mcs = round(0.01·n)` uncapped recovers
  89 % of that loss.
- Per-chunk instead of per-speaker-per-window embedding: pyannote embeds ~21×
  redundant audio; embedding each chunk once and applying speaker masks at
  aggregation is mathematically equivalent (DER 0.255 → 0.257).

### 1.6 Licensing (diarization)

| Component | Weights | Shippable |
|---|---|---|
| pyannote segmentation-3.0 | MIT (HF-gated; ungated ONNX mirrors exist) | ✅ |
| pyannote community-1 | CC-BY-4.0 — may be bundled/mirrored with attribution | ✅ |
| **DiariZen meeting-base** | **MIT, ungated** | ✅ |
| DiariZen wavlm-* (all -s80 variants) | CC-BY-NC-4.0 | ❌ |
| Sortformer v1 | CC-BY-NC-4.0 | ❌ |
| Sortformer streaming v2 | CC-BY-4.0 | ✅ (but see §1.7) |
| Sortformer streaming v2.1 | NVIDIA Open Model License (indemnification clause) | ⚠️ |
| Rev reverb-diarization v1/v2 (sherpa ships them) | bespoke non-commercial | ❌ |
| EEND-TA | never released | ❌ |

Why meeting-base escapes DiariZen's NC restriction: it is trained only on
AMI + AISHELL-4 + AliMeeting; the NC tag on the other checkpoints protects
non-commercial corpora in their larger training mixture (DiariZen's own
compliance note). CLAUDE.md's blanket "DiariZen is CC-BY-NC" predates this
checkpoint.

### 1.7 Disqualified candidates, with reasons

- **Sortformer (all variants)**: ONNX export broken for v2/v2.1
  (NeMo #15077 closed without fix), no published CPU RTF and no CPU
  optimization planned, hard 4-speaker cap, cannot consume a known speaker
  count, weak on English. Only interesting as CoreML on macOS, where speakrs
  already serves.
- **Joint ASR+diarization / LLM-based SA-ASR** (G-STAR, JEDIS-LLM, MOSS,
  SpeakerLM, DiCoW): all GPU-only, unreleased, API-only, or Whisper-large
  class. See §5.1.
- **VAD→cluster fast pipelines** (senko, FunASR default, 3D-Speaker default):
  no overlap attribution — Miss floor ≈ the 16–20 % meeting overlap rate.
- **pyannote.audio 4.x as a dependency**: ships opt-out OTLP telemetry
  (`telemetry/config.yaml`, `metrics_enabled: true`, endpoint
  `otel.pyannote.ai`) sending per-file duration and speaker-count parameters;
  kill switch `PYANNOTE_METRICS_ENABLED=0` before import. Also dropped
  onnxruntime entirely and added the paid-API SDK as a dependency. The
  *models* remain worth extracting; the *package* is a poor host for an
  offline privacy-positioned tool.

## 2. Speaker embeddings (the re-ID voiceprint)

### 2.1 Model table (EER % on VoxCeleb1-O unless noted; all ONNX-exportable unless noted)

| Model | Params | Vox1-O EER | 2 s EER | License | Note |
|---|---|---|---|---|---|
| ERes2Net-base (**shipped today**) | 6.6 M | 0.84 | **3.28** | Apache-2.0 | collapses on short turns — worse at 2 s than ECAPA (1.95) and ResNet34 (2.12); arXiv:2406.02167 |
| **ERes2NetV2** | 17.8 M | 0.61 | **1.48** | Apache-2.0 | successor, fixes the short-turn cliff; 12.6 GMACs |
| CAM++ | 7.2 M | 0.65 | — | Apache-2.0 | cheapest (RTF 0.013 1-thread) — but **rejected empirically in-repo 2026-07**: the VoxCeleb export flips cluster identity between segmentation windows (`models.py`) |
| WeSpeaker ResNet34-LM | 6.6 M | 0.66–0.87 | 2.12 | CC-BY-4.0 | the embedding pyannote 3.1/community-1 actually use — known-good pairing with segmentation-3.0 |
| WeSpeaker ResNet293-LM | 28.6 M | 0.425 | — | CC-BY-4.0 | accuracy/size ceiling of the shippable set |
| NeMo TitaNet-Large | ~23 M | 0.66 | — | CC-BY-4.0 | |
| ReDimNet-B6 / ReDimNet2 | 12–13 M | 0.29–0.40 | — | split | accuracy leader; MIT code, but best checkpoints VoxBlink2-trained = CC-BY-NC-SA; official ONNX export broken since 2024 (IDRnD/ReDimNet #13, #20); vox2-only b0–b6 are CC-BY-4.0 — watch, don't adopt |
| WavLM-large + TDNN | 316 M | 0.38–0.52 | — | varies | SSL front-ends are the most fragile under noise/RIR (SVeritas: 23 % EER clean far-field) |

Cross-toolkit control (Kiwano, arXiv:2606.22369, 2026-06: same training data,
cosine-only): all mainstream toolkits land within 15 % of each other —
**architecture choice is not where the remaining error lives**.

int8 quantization of speaker embedders: no official project ships one;
community results split (static int8 TitaNet on x86: 1.9× faster, EER
unchanged; dynamic int8 CAM++ on ARM: slower than fp32 and −9 % margin).
Not a lever.

### 2.2 The benchmark-to-reality gap (why 0.6 % EER means 12–16 % naming error)

The models we ship degrade 11–15× from VoxCeleb to realistic conditions
(3D-Speaker's own numbers, arXiv:2403.19971):

| Model | Vox1-O | Multi-Device | Multi-Distance | Multi-Dialect |
|---|---|---|---|---|
| ERes2Net-base | 0.84 | 7.12 | 9.82 | 12.10 |
| ERes2NetV2 | 0.61 | 6.52 | 8.88 | 11.30 |

Independent stacking factors, each measured:

- **Duration**: full → 3 s → 2 s = 0.61 → 0.98 → 1.48 % (ERes2NetV2,
  arXiv:2406.02167); at 1 s, 4–13 % for every model; far-field @1 s = 16.96 %
  vs close-talk full 1.45 % — 12× (arXiv:2002.06033).
- **Cross-device**: swapping mic type alone costs 2.24× at the same position
  (Hi-MIA, arXiv:2109.12056). Mic-to-mic spread within one room (3.2×) is as
  large as the near→far gap (Mošner, Interspeech 2018).
- **Session boundary alone**: same-session 1.25 % vs different-session 1.60 %
  (+28 %) at zero age gap (arXiv:2306.07501). Aging: ~+1 point EER over
  10 years, ~5× faster for female speakers (VoxAging, arXiv:2505.21445).
- **Speaking-style mismatch beats aging**: conv→conv 0.57 vs conv→read 3.03
  (5×; Afshan, Interspeech 2020) — enroll from natural conversational speech,
  i.e. from meetings themselves, not a read passage.
- **Overlap contamination**: including overlapped intervals in the embedding
  makes it *worse* (12.84 → 14.11 % EER, Guided Speaker Embedding,
  arXiv:2410.12182); frame-level identity is 97.7 % in single-speaker regions
  vs 35.2 % in overlap (arXiv:2306.00625).
- **Codec compression costs ~nothing**; RIR+noise costs 3–5× (SVeritas,
  arXiv:2509.17091).

**Realistic end-to-end expectation** (Microsoft, real meetings, d-vectors,
cosine-to-nearest-profile, arXiv:2102.03634): **16.2 % segment error with
~5 s profiles, 11.6 % with ~25 s profiles**. Never promise near-perfect naming
from meeting audio.

### 2.3 Mitigations with measured effect (and their traps)

| Mitigation | Recovery | Source |
|---|---|---|
| Enroll on the device you verify on | 18 % rel, free | Hi-MIA, arXiv:1912.01231 |
| Enrollment augmented with the test clip's own noise | closes 53 % of channel gap | ibid. |
| WPE dereverb — good mic | 48 % rel | Mošner IS2018 |
| WPE dereverb — bad mic | **worse** (16.88 → 19.71) | ibid. |
| In-domain back-end on OOD front-end | 12 % rel | arXiv:1911.01799 |

Nothing recovers more than ~half the far-field gap.

## 3. Scoring and thresholds

### 3.1 The state of the art in scoring

- **Cosine has won for matched-domain scoring** on large-margin embeddings
  (cosine *is* PLDA with identity covariances). PLDA survives only under
  severe domain mismatch with labelled in-domain data — and in one case that
  matters to us: **varying enrollment amount**. With mixed enrollment sizes,
  cosine + embedding averaging EER 2.85 % vs spherical PLDA 1.99 % pooled —
  purely because the cosine score scale drifts with how much audio went into
  the mean (arXiv:2302.09523). Directly indicts our duration-weighted mean.
- **AS-Norm** buys 6–14 % relative EER in verification, and its real value is
  score-distribution *stability* (C_act 0.577 → 0.257 out-of-domain,
  arXiv:2203.15106). **But in open-set identification it degraded the two
  strongest models tested** (VoxWatch, arXiv:2307.00169) while QMF calibration
  helped every configuration (−21 % rel FAR). For open-set:
  calibration > fusion > score normalization. Measure before adopting.
- **QMF (quality-measure fusion)**: logistic regression over score + quality
  features (duration, speech duration, embedding magnitude, cohort mean
  score). Cut minDCF 35 % relative with no model change (ID R&D VoxSRC-23,
  arXiv:2308.08294). The cohort-mean feature comes free with an AS-Norm
  cohort.

### 3.2 Open-set identification (the "unknown speaker" problem)

- Decision rule everywhere: max cosine over gallery vs threshold. A
  margin-to-second-best statistic is standard in face recognition but
  **absent from the speaker literature** — unvalidated heuristic.
- **Gallery-size scaling**: FPIR ≈ N · FMR (NIST identity). Measured: FAR at
  FRR=5 % goes 2.4 % → 16.1 % as the gallery grows 50 → 500 (VoxWatch) with
  no acoustic change. **N ≈ 50–60 is the last comfortable size for a flat
  threshold** — our 5–50-profile regime sits just inside it.
- Given correct detection, closed-set identification is >99 % precise
  (VoxWatch) — **the reject decision is the entire problem**.

### 3.3 Toolkit thresholds — published defaults, and three traps

| Toolkit / model | Documented threshold | Raw-cosine equivalent |
|---|---|---|
| SpeechBrain ECAPA | 0.25 | 0.25 |
| pyannote 3.1 clustering | 0.7046 (distance) | 0.295 |
| 3D-Speaker ERes2Net en | 0.356 | 0.356 |
| 3D-Speaker ERes2NetV2 zh | 0.360 | ~0.36 |
| NeMo TitaNet | 0.7 (rescaled (cos+1)/2) | 0.40 |
| sherpa-onnx identification | 0.6 | 0.6 |
| **stenograf `profiles.py`** | **0.5** | **0.5** |

Traps: (1) NeMo and the WeSpeaker CLI operate on `(cos+1)/2`; (2) pyannote's
number is a distance; (3) **no toolkit applies AS-Norm in its inference API**
— published EERs are AS-Norm'd, shipped defaults threshold raw cosine; the
two are not on the same score distribution. Verification-tuned defaults
cluster at raw cosine 0.25–0.40; sherpa's 0.6 is an identification default
biased against strangers. Our flat 0.5 matches no published operating point —
and **no toolkit publishes a threshold together with its FAR/FRR**, so the
operating point must be measured locally (which the harness makes possible).

Thresholds are model × training-data × domain specific (ERes2NetV2 zh vs en
defaults differ by 0.004; CAM++ zh vs en by 0.19). A threshold is only
meaningful relative to the embedding model — which `profiles.py` already
encodes by binding profiles to the model id.

## 4. Enrollment

- **~30 s net speech is the knee**; below 10 s is the steep part; beyond 60 s
  near-zero returns (VoxWatch; NIST SRE24 found three 10-s segments comparable
  to three 60-s; Amazon Connect Voice ID enrolled at 30 s).
- **Multiple utterances beat one long one**: 1 → 3 enrollment utterances is
  worth 2.7–3.3× EER (arXiv:2203.05642) or +18 DIR points (VoxBlink2);
  3 → 5 much less.
- **Score averaging beats embedding averaging** for multi-enrollment: 2.05 %
  vs 2.85 % EER on identical data (arXiv:2302.09523). The "average the
  embeddings" rule is an i-vector-era result the paper explicitly retracts
  for modern embeddings. Store per-session embeddings; average scores.
- **Channel match beats sample quality**: enrolling on the matched (worse)
  device beats a clean mismatched sample by 18 % relative — enrollment from
  the meeting's own audio is the right protocol, and it also matches
  speaking style (§2.2).
- **Online enrollment beats up-front enrollment** (arXiv:2509.18377, AMI
  headset, ECAPA): uncorrected DER 36.32 / speaker error 22.05; one up-front
  enrollment per speaker → 27.23 / 12.96; **online enrollment from
  user-corrected segments → 26.82 / 12.55**; full workflow 24.70 / 10.43
  (−52.7 % speaker error). One rename per speaker, applied as enrollment, is
  the single biggest re-ID win available.
- **Never update a profile ungated** — memory poisoning is measured, not
  hypothetical: with the reliability gate disabled, iterative enrollment
  update falls *below* the no-update baseline (EvoTSE, arXiv:2604.06810;
  corroborated arXiv:2601.12769 — anchor the original enrollment, gate on
  confidence, prefer user-confirmed segments).
- Not found in the literature: multi-centroid per-person profiles evaluated
  against a single centroid; duration-weighted vs unweighted aggregation; a
  controlled multi-session-vs-single-session gain at matched duration.
  Per-session storage + score averaging subsumes the first two anyway.

## 5. Integrated pipelines

### 5.1 Modular vs joint, 2026 verdict

No joint speaker-attributed ASR model is simultaneously (a) better than
modular on meeting audio, (b) CPU-runnable, (c) permissively licensed. The
2026 frontier (G-STAR: Qwen2-7B; JEDIS-LLM: Phi-4-multimodal) is GPU-only
with unreleased weights. On wide-band meeting data the classic cascade beats
end-to-end by 2–3× DER (DiaPer vs VBx: AISHELL-4 41.43 vs 14.46, VoxConverse
21.11 vs 6.69 — arXiv:2312.04324/2407.08752). Even diarization-conditioned
Whisper (DiCoW, Apache-2.0) degrades 16.5 → 23.6 tcpWER on AMI-SDM when
oracle diarization is swapped for real — **~30 % of end-to-end error is the
diarizer, not the recognizer**, in every architecture.

The advertised advantage of joint models is overlap handling and speaker
counting on a single mixed stream — the two problems our per-channel capture
already removes. E2E models also degrade over long spans (G-STAR global vs
local: AMI DER 19.00 → 32.23). Speechmatics productizes per-channel
diarization as its own mode; we get it structurally.

**TS-VAD** (using enrolled profiles as *input* to diarization): mature in
research (CHiME winners), but strictly additive to a clustering front-end
(profiles must first be estimated by one), and **no permissively-licensed,
ONNX-exported, CPU-practical implementation exists** (NeMo/WeSpeaker/
3D-Speaker/pyannote/diart/sherpa all ship enrollment-free only, verified).
Personal-VAD-style models are 130 K–1 M params but have no open weights.

### 5.2 The task benchmark: Target Speaker Tagging (TST-Bench, arXiv:2606.14091, 2026-06)

Defines exactly our task: long recordings, pre-enrolled speakers, label known
speakers, reject unknowns. Its baseline is our architecture (ECAPA + cosine +
AS-Norm, spectral clustering, ~20 s enrollments). Results: **DIR 88.79 % @
FAR 0.5 %** on TST-Bench; **94.51 % on real ICSI meetings**.

Its ablations are the only published ones on a diarize→match pipeline:

| Change | DIR@FAR=0.5 % |
|---|---|
| balanced clustering (baseline) | 88.79 |
| under-clustering | 86.75 |
| **over-clustering** | **89.46** |
| + 0.1 s segment margin | 89.05 |
| + top-3 merging of similar short segments | 89.03 (94.15 @ FAR 1 %) |

Over-clustering raised raw speaker confusion *more* than under-clustering yet
won: "split clusters can be re-merged at the identification stage, whereas
under-clustering irreversibly contaminates segment embeddings." One decision
per cluster (our design) measured 81.82 % vs 89.03 % for per-segment pooling
in the strict regime — a 7-point gap.

### 5.3 Where the errors are (three independent 2025–26 analyses converge)

1. **Missed speech / boundary error** — >50 % of the modular gap-to-oracle
   (arXiv:2509.10143: missing segments ≈1.0 % WER of a 1.5 % gap), 72 % of
   TST-Bench's DER, dominant in the ETH benchmark, attributed to onset/offset
   timing.
2. **Cluster granularity** — second; fix is the over-clustering bias.
3. **Word↔turn intersection** — real but small (≈0.2 % WER). WhisperX's
   sum-intersection-argmax (what we do) is the reference implementation.

### 5.4 Commercial landscape (documented facts only)

pyannoteAI is the only vendor with documented voiceprint enrollment
(≤30 s clips, 1–50 voiceprints/request, threshold 50–70 recommended,
exclusive matching default true, explicit false-match warning). AssemblyAI:
per-file name substitution only, ≥30 s/speaker guidance. Deepgram/Gladia:
no enrollment. Speechmatics: ships per-channel + within-channel diarization
as a product mode. No vendor publishes DER/cpWER for meeting naming.

## 6. Evaluating without hand labels

- **No published proxy metric correlates with DER without labels** (systematic
  sweep found essentially nothing). Cross-model disagreement can rank *where*
  errors are (review queue), not *how many*. Simulated/TTS meetings are
  validated for training only — the one synthetic corpus with baselines
  (LibriConvo) shows a 2.2× Sortformer-vs-pyannote gap far wider than on any
  real corpus, i.e. synthetic data distorts rankings.
- **The unlock: AMI and ICSI are CC-BY-4.0** with per-speaker headset channels
  and full human annotations (explicit statement on
  <https://groups.inf.ed.ac.uk/ami/corpus/>). Mixing N−1 headset channels
  into a synthetic "loopback" and keeping one as "mic" reproduces our exact
  two-channel topology with real human labels — a genuine reference set, not
  a proxy. AMI scenario meetings repeat the same 4 participants across
  sessions (ES2002a–d …), which additionally gives enroll-on-session-A /
  test-on-session-B re-ID ground truth with named identities.
- Metric of record for the full attributed pipeline: **tcpWER with a 5 s
  collar** (CHiME-8 convention, MeetEval implementation); for the naming
  stage: **DIR @ FAR** (TST-Bench / VoxBlink2 convention). Our `eval/der.py`
  already computes DER + word attribution; it needs only the open-set naming
  metric added.

## 7. Summary: findings → implications for stenograf

| Finding | Evidence | Implication |
|---|---|---|
| Modular cascade + per-channel + known count is the right 2026 architecture | §1.2, §1.4, §5.1 | keep; do not pursue joint models, TS-VAD, Sortformer |
| Missed speech at boundaries dominates | §1.2, §5.3 | boundary margin, overlap-aware turns — before any model swap |
| Over-clustering beats balanced; merging happens at ID stage | §5.2 | bias cluster count up where estimated; merge by profile match |
| Mean-embedding-per-cluster + flat 0.5 is triply indicted | §3.1, §3.3, §4 | per-session embeddings, score averaging, measured threshold |
| Overlap frames poison embeddings; short clusters unreliable | §2.2 | exclude overlap from embedding slices; don't name <3 s clusters |
| Enrollment from meeting audio + rename-once online enrollment | §4 | biggest single re-ID win (−52.7 % speaker error); gate updates |
| ERes2Net-base collapses on short turns; V2 fixes it | §2.1 | embedding upgrade candidate (CAM++ stays rejected) |
| DiariZen meeting-base (MIT) is the only shippable model clearly above baseline | §1.1 | candidate, gated on measurement + ONNX export work |
| Stride 3 + per-chunk embedding ≈ 8.5× speedup at ≈0 cost | §1.5 | pays for any heavier model; requires owning the pipeline loop |
| AMI/ICSI CC-BY-4.0 with headset channels | §6 | labelled eval in our exact topology without hand-labelling |

The implementation sequence lives in `PLAN-DIARIZATION.md`.
