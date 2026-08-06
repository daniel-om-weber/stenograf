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

## Step 0 — the labelled harness — BUILT 2026-08-06, gates green

`eval/ami.py` (corpus fetch, topology synthesis, references, re-ID trials,
one-command matrix), `eval/diarize.py --ami` (the real per-channel pipeline,
known counts, cluster embeddings), `eval/reid_score.py` (DIR @ FAR + the
FAR/FRR curve, pure and unit-tested), and `eval/der.py` now scoring
`refs/ami/`. One command re-runs everything:
`uv run --group eval eval/ami.py run` — **22.2 min** full matrix (gate ≤1 h).
Subset: AMI ES2003 + IS1009, sessions a–d (the plan's ES2002/IS1000 examples
both carry documented audio faults — dataproblems table), ICSI Bmr021 +
Bmr025 (8 close-talk speakers). Full tables land in `eval/out/diar-report.md`
and `eval/out/reid-report.md`; the 2026-08-06 baseline:

- **Loop (mixed) channels**: ES2003 4.8–19.0 % DER (b/c at 4.8/5.8 %), ICSI
  14.0/18.4 % — the "teens" ballpark holds. IS1009 22.5–47.3 % with
  13–23 pts *confusion* at false alarm ≤4 % — a genuinely hard
  (quiet-speaker, overlap-heavy) group, i.e. the program's target, not a
  harness artifact.
- **Word attribution** (what the user reads): 77–99 % on loops, 100 % on
  every mic channel.
- **Naming**: DIR **88.2 % @ FAR 0 %** (threshold 0.609) and **94.1 % @ FAR
  3.4 %** (threshold 0.447), from 17 known / 59 unknown trials. FAR
  granularity is 1.7 % at this trial count — grow trials (more groups/ICSI
  recurrence) before fine threshold work. First measured curve *brackets*
  the shipped 0.5 default rather than indicting it (step 2.4 refines).
- **Solo (mic) channels**: 17–29 % AMI, 38–45 % ICSI, miss-dominated, false
  alarm low. The original "low single digits" gate here was mis-calibrated
  and is RETIRED on measurement: the floor decomposes into Silero-vs-verbatim
  missed speech (12.8 % on the ungated channel — backchannels, soft tails,
  disfluencies the pipeline intentionally does not transcribe), the crosstalk
  gate's uniform ~10-pt cost, and ICSI's utterance-granular references.
  Constant costs cancel in deltas; false alarm — the axis that would corrupt
  a measurement — is the one that was driven to ~0.

Building it surfaced three measured facts, each recorded where it bites:

- **Headset channels are open mics.** Naive synthesis measured 63–76 % DER
  of pure bleed; the label-free crosstalk gate in `ami.py` (raw cross-channel
  dominance + own-speech-level hysteresis) is what makes the topology honest.
  Three simpler gate designs each failed a *measured* way — the docstrings
  carry the numbers; don't re-simplify without re-running them.
- **sherpa's `process()` needed ~31 GB** on one dense 38-min channel,
  thread-count-independent — a leak, not an algorithm. `SpeakerEmbeddingExtractor`
  retained ~21 MB per call at fixed input shape, unbounded and below the
  extractor object (a fresh extractor per call reclaimed nothing, and the same
  ONNX graph under plain onnxruntime held flat), so no in-process workaround
  existed. Fixed upstream: the 1.13.4 floor holds that channel at 5.4 GB with
  byte-identical turns. The matrix still isolates each loop channel in its own
  process — it costs nothing and bounds the next one.
- **The `num_clusters=1` length limit is gone.** A crash at ≈15 min on
  sherpa-onnx 1.12.40 looked like a segfault; on 1.13.4 both full-length mic
  channels (ES2003c 37.6 min, Bmr021 36.9 min) diarize at count 1 cleanly
  (4.5 / 2.8 GB, 2026-08-06), and 1.12.40 itself survives 20 min — so it was
  most likely the leak failing an allocation, not a distinct bug. Step 1.1's
  diarizer-on-solo comparison arm runs on whole channels.

Calibration note: published AMI numbers in the research doc are family-A
(0 s collar); our der.py is family-B (0.25 s). Only compare our numbers to
our numbers — the harness exists for deltas, not leaderboard placement.

## Step 1 — policy fixes in the current pipeline (no new models)

Each lands separately, gated on its own harness delta. Ordered by expected
size (the error analyses: boundary miss > cluster granularity > everything;
word intersection explicitly last and not touched).

1. **Solo channel bypass — ALREADY SHIPPED, and now measured 2026-08-06.**
   `finalize_channel` has never run a diarizer at `num_speakers=1`; the
   entries are the ASR/VAD spans under one label. `eval/solo_arms.py` scores
   it against the arms it replaces on all ten single-speaker channels
   (`eval/README.md`). The premise written here was wrong in both directions
   and the conclusion survives anyway: VAD-only is **not** better on DER
   (24.8 % vs 11.4 % for the diarizer at k=1) — but word attribution, the
   thing the user reads, is 100 % in both arms on every channel, so the DER
   gap is speech the transcript never contains. Diarizing a solo channel is
   declined on those numbers. The measurement's real payload is the third
   arm: **estimating the count on a solo channel splits it 2–3 ways on 10/10
   channels**, costing 22 pts of word attribution (worst 54). That is a live
   user-facing failure whenever nobody states a count — see 3.
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
   Step 1's own measurement raises the stakes and exposes a hole: an estimated
   solo channel splits 2–3 ways on 10/10 corpus channels (−22 pts word
   attribution), and merge-at-naming recovers that **only for a speaker who
   has a voice profile**. What an unprofiled solo speaker gets back is
   unanswered — decide it here, with the estimator's own confidence or a
   single-cluster-dominance test as the candidates, and measure it on
   `solo_arms.py`'s est arm.
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
