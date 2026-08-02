# PLAN-DIARIZATION — the diarization + speaker re-ID accuracy program

Opened 2026-08-02. Evidence: `eval/diarization-sota-2026.md` (the research
record — read it before questioning any decision here; every claim below has
a source there). Goal: maximize the accuracy of (1) separating speakers
within a channel and (2) naming them from the voice-profile library, on the
CPU-first, offline, permissively-licensed constraints that are locked in
CLAUDE.md.

**Two principles govern every step.**

- **Measure first.** Nothing lands without a before/after on the step-0
  harness. The research record is full of "obvious" improvements that
  measured negative (forcing the true speaker count worsened DER 9.3 %
  relative; AS-Norm degraded strong models in open-set; WPE dereverb helps a
  good mic and hurts a bad one; ungated profile updates fall below the
  no-update baseline). The declined hand-labelling item blocked exactly this
  kind of measurement for a year; step 0 removes the blocker.
- **One path.** Each change replaces the old behavior outright — no
  primary+fallback pairs, no config switches to keep both. Where a candidate
  loses on the harness, it is declined with its numbers, not kept as an
  option.

The architecture itself is settled and not part of this plan: modular
cascade, per-channel, word-timestamp intersection. The 2026 research verdict
is that this is the *right* architecture (no joint model is better + CPU +
permissive; per-channel known-count is a measured ~2× DER advantage), so the
program is: keep the skeleton, upgrade the organs.

---

## Step 0 — the labelled harness (the long pole; everything gates on it)

The scorer exists (`eval/der.py`: frame DER + word attribution, pure
functions, tested). What is missing is references — and AMI + ICSI are
CC-BY-4.0 **with per-speaker headset channels and full human annotations**,
which means references in our exact topology can be *built*, not hand-made:

- **`eval/ami.py`** — fetch a fixed subset of AMI scenario meetings (the
  ES/IS series: same 4 participants across 4 sessions each) plus 2–3 ICSI
  meetings for a larger-group condition. For each meeting, synthesize the
  stenograf topology: one headset channel is the "mic" channel; the other
  N−1 mixed are the "loopback" channel. Emit per-channel RTTM references
  from the corpus word/segment annotations (global speaker names preserved),
  into `eval/refs/ami/`.
- **Extend `eval/diarize.py`** to run the real per-channel pipeline over
  those channels (it already runs the real diarizer + finalize).
- **`eval/reid_score.py`** — the naming-stage metric `der.py` doesn't have:
  enroll each named participant from session A of their group, run
  cluster→profile matching on sessions B–D, score **DIR @ FAR** (detection &
  identification rate at fixed false-accept, the TST-Bench/VoxBlink2
  convention) plus the FAR/FRR curve as the threshold sweeps. Unknown-speaker
  trials come from enrolling only a subset of participants.
- Score DER with `eval/der.py` as-is (0.25 s collar, overlap scored) and keep
  the word-attribution number — it is what the user reads.

Gates: the harness reproduces sane ballparks (our stack should land in the
teens of DER on the mixed channel, low single digits on the solo channel —
far off means a harness bug, not a model discovery); one command re-runs the
whole matrix; total runtime ≤ ~1 h on this machine so it gets run.

Calibration note: published AMI numbers in the research doc are family-A
(0 s collar); our der.py is family-B (0.25 s). Only compare our numbers to
our numbers — the harness exists for deltas, not leaderboard placement.

## Step 1 — policy fixes in the current pipeline (no new models)

Each lands separately, gated on its own harness delta. Ordered by expected
size (the error analyses: boundary miss > cluster granularity > everything;
word intersection explicitly last and not touched).

1. **Solo channel bypass.** When the resolved per-channel speaker count is 1,
   do not run the diarizer on that channel: VAD spans + the single fixed
   label. Even 1-speaker audio costs 1.5–4.7 % DER through a diarizer;
   VAD-only can only be better, and it removes an entire failure mode from
   the common headset case. (One path: this replaces diarization for count=1,
   not a fallback around it.)
2. **Boundary margin.** Pad diarization turn onsets/offsets by ~0.1 s
   (clamped at neighbors) before word intersection. Missed speech from
   boundary imprecision (~350 ms average across systems) is the dominant
   error everywhere; TST-Bench measured +0.26 DIR for exactly this margin,
   and 0.25/0.5 s progressively worse — sweep 0.05–0.25 s on the harness.
3. **Over-clustering bias + merge-at-naming.** Where the count is estimated
   (auto-count channels), bias the estimate up rather than down; where known,
   test k and k+1. After matching, clusters that hit the *same* profile above
   threshold merge back to one speaker. Over-clustering measured +0.67 DIR
   over balanced and under-clustering −2.04; splits are recoverable at the
   naming stage, merges are not. Also relevant to the known far-field
   over-splitting complaint in PLAN.md: the fix for that is this merge step,
   not a lower cluster count.
4. **Overlap-clean embeddings.** `cluster_embeddings()` currently slices by
   turn times, overlap included. Exclude spans where ≥2 turns are active
   (computable from the turn list alone) before embedding — including
   overlap measurably *worsens* the embedding (12.84 → 14.11 % EER), and
   frame-level identity in overlap is near-chance (35 %).
5. **Minimum-audio gate for naming.** A cluster with < ~3 s of clean
   (non-overlap) speech is not matched against profiles — EER at 2 s is
   2.4× the full-duration figure, and ERes2Net-base specifically collapses
   on short turns (3.28 % @ 2 s). Short clusters keep their local S-label
   (and are candidates for the step-1.3 merge instead).
6. **Docstring fix** (immediate, no gate): `diarization/sherpa.py` says
   "CAM++ embeddings"; `assets.py` ships ERes2Net and records why CAM++ was
   rejected. Say what ships.

## Step 2 — profile store v2 + the enrollment loop

The research indicts all three quiet assumptions in today's re-ID design:
the single mean embedding, the flat 0.5 threshold, and enrollment as a
one-shot act.

1. **Per-meeting embeddings, score averaging.** `_STORE_VERSION` 2: a profile
   holds up to N (start: 8, LRU by meeting date) per-meeting embeddings
   instead of one running mean. Match score = mean cosine against the stored
   set (score averaging measured 2.05 % vs 2.85 % EER for embedding
   averaging on identical data; the i-vector-era "average the embeddings"
   rule is explicitly retracted for modern embeddings). Migration: a v1
   profile's mean becomes its first stored embedding. Threshold semantics
   unchanged (max-over-gallery vs threshold — margin-to-second-best is
   unvalidated in the speaker literature; not adopting it).
2. **Rename-once online enrollment.** When the user corrects/assigns a
   speaker name for a meeting (already possible via profile naming), the
   corrected cluster's embedding from *that meeting's audio* is added to the
   profile — measured as the single biggest available win (−52.7 % speaker
   error on AMI from at most one correction per speaker), and enrollment
   from the meeting's own channel beats any clean sample (channel match
   +18 % rel; conversational style vs read speech 5×).
3. **Gated automatic updates, anchored.** An auto-matched cluster may add its
   embedding to the profile only above a high-confidence bar (score margin
   above threshold + step-1.5 duration gate); the original user-confirmed
   enrollment is never evicted. Ungated iterative updates measured *below*
   the no-update baseline (memory poisoning) — silent drift into a
   colleague's voice is the one unrecoverable failure here.
4. **Threshold, measured at last.** Sweep the FAR/FRR curve on the step-0
   re-ID harness and pick the operating point (bias: strangers must not be
   named — low FAR, accept more "unknown"). Today's flat 0.5 matches no
   published operating point; verification defaults cluster at raw cosine
   0.25–0.40 and sherpa's identification default is 0.6, and no toolkit
   publishes threshold and error rate together — ours will be the measured
   exception. `DEFAULT_THRESHOLD`'s docstring gets the curve's numbers.
   AS-Norm/QMF are *not* adopted in this step: AS-Norm measurably hurt
   strong models in open-set; a duration-aware QMF is a step-5 candidate
   only if the measured curve is unsatisfying.

## Step 3 — embedding model upgrade

Candidates, in order: **ERes2NetV2** (Apache-2.0, 17.8 M, ONNX on
ModelScope; fixes exactly the short-turn cliff our shipped ERes2Net-base has:
1.48 % vs 3.28 % @ 2 s) and **WeSpeaker ResNet34-LM** (CC-BY-4.0, 6.6 M,
ONNX shipped by sherpa; the known-good pairing with segmentation-3.0 —
literally pyannote's own embedding). CAM++ stays rejected (measured in-repo
2026-07: cluster identity flips between segmentation windows — `assets.py`).
ReDimNet/ReDimNet2 is watch-only (best checkpoints CC-BY-NC-SA, official
ONNX export broken; trigger: a CC-BY vox2 checkpoint with working export
beating ERes2NetV2 on the harness).

Gate: harness DER (clustering quality) *and* DIR@FAR (naming) both
non-regressing, short-turn naming improved. Profiles are model-bound by
design (`voiceprints.py`), so the swap starts a fresh profile set — ship
together with step 2's rename-once loop so re-enrollment is one correction
per colleague, not a manual chore. Loser is declined with numbers; one model
ships (one path).

## Step 4 — own the diarization loop (VBx + stride + per-chunk embedding)

sherpa's `OfflineSpeakerDiarization.process()` is a monolith: to change
clustering or stride we take over the loop, keeping the *models* and the
`SpeakerEmbeddingExtractor` — run segmentation-3.0 via onnxruntime directly
(powerset decode is ~30 lines; sherpa's own implementation is the reference),
then our clustering. This buys, in one step:

- **VBx clustering** (Apache-2.0, ~200 lines of numpy: AHC init + one
  eigen-decomposition + ~10 VB iterations) — the main source of pyannote
  community-1's −2.8 AMI-SDM gain over 3.1, at zero new-model cost.
- **Stride 3 s + per-chunk embedding** — measured 8.5× CPU speedup at ≈0 DER
  cost for ≤5 speakers (with the min-cluster-size floor set uncapped-1 %,
  which recovers the stride's under-counting failure mode). Finalize
  diarization drops from ~8 min to ~1–2 min per meeting hour — the budget
  that makes step 5 affordable.
- The step-1 policies (overlap exclusion, margins) become first-class in our
  own loop instead of post-processing sherpa's output.

Gate: DER within noise of the sherpa path on the harness *before* the VBx
switch (loop parity), then VBx must beat AHC by a visible margin. The sherpa
diarization pipeline dependency is then dropped on non-mac platforms (the
extractor stays); the old path is deleted, not kept as a fallback. macOS
(stenodiar/speakrs, community-1 CoreML) is untouched by this step.

## Step 5 — the DiariZen gate (the only model that beats everything above)

`BUT-FIT/diarizen-meeting-base` (MIT, ungated) is the one shippable model
clearly above the pyannote family: AMI-SDM 15.6 vs 21.1 for the same paper's
pyannote-3 run, domain-matched to meetings. Costs: PyTorch-only today (an
ONNX export of WavLM-Base+ + Conformer must be made and verified — plausible,
unproven), ~400 MB on disk, an estimated 2–3× the *current* finalize
diarization time (≈ parity with today after step 4's speedup).

Decide only after steps 1–4 are measured: **trigger = the harness still shows
≥2 points of DER (or a visible word-attribution gap) between our stack and
what the research record says meeting-base delivers.** If triggered, the
export is the first task, and near-field behavior on the harness's mic
channel is the go/no-go (its training data is exclusively far-field). If
adopted it *replaces* the segmentation+clustering half everywhere off-mac —
one path; if it loses or the export stalls, decline here with numbers.

---

## Deferred and declined (with triggers, so they stay closed)

- **Sortformer, all variants — declined.** Broken/absent ONNX+CPU path
  (closed-without-fix upstream), hard 4-speaker cap, cannot consume a known
  speaker count (our best signal), weak on English. No trigger; the CoreML
  builds are irrelevant while speakrs serves macOS.
- **Joint / LLM speaker-attributed ASR — declined.** Nothing is
  better+CPU+permissive in 2026, and per-channel capture removes the mixed-
  stream problems joint models exist to solve. Trigger: a permissive,
  CPU-practical joint model that beats the modular cascade *on meeting
  benchmarks with real (non-oracle) diarization*.
- **TS-VAD with enrolled profiles — declined.** Strictly additive to a
  clustering front-end, and no permissive CPU implementation exists.
  Trigger: an Apache/MIT ONNX implementation appears (a Personal-VAD-class
  small model with open weights would also reopen this).
- **AS-Norm — not adopted, measure-first.** Degraded the two strongest
  models in the open-set benchmark. Trigger for a harness experiment: the
  step-2.4 threshold curve is unsatisfying, in which case QMF-style
  duration-aware calibration is tried *first* (helped every configuration,
  −21 % rel FAR).
- **EEND-TA — watch.** The research ceiling (AMI 11.04, 13.3 M params,
  ~460× realtime CPU claimed) with no code or weights. Trigger: a release.
- **ReDimNet2 — watch.** Trigger in step 3.
- **pyannote.audio as a dependency — declined.** Opt-out telemetry phoning
  home audio durations + speaker counts, onnxruntime dropped, paid-SDK
  dependency. We keep extracting weights; the package never imports here.
- **Hand-labelling our private eval audio — stays declined** (PLAN.md). The
  step-0 harness sidesteps it with public labelled corpora in our topology;
  the private segments remain for ASR adjudication and smoke checks only.
