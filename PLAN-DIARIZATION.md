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
`uv run --group eval eval/ami.py run` — 22.2 min at the original 20 channels;
**41.8 min** since the 2026-08-07 growth (gate ≤1 h).
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

**Grown 2026-08-07 (the "grow trials" prerequisite for 2.4):** five groups,
40 channels, 41.8-min matrix (gate ≤ 1 h holds). ES2007 + TS3010 joined after
a vetting pass (all three AMI sites now covered; the pass also *measured*
clipping in table-clean IS1004/IS1006 — declined; `AMI_GROUPS` docstring),
and the ICSI Bmr series became a real re-ID group — Bmr021 enrolls,
Bmr024/025/030 carry recurrence plus *natural* strangers (attendance churn),
mapped onto session letters so every session-a convention holds
(`ICSI_SESSIONS` docstring). Growing the pool surfaced a scoring flaw the
2026-08-06 baseline hid: pooling cross-group trials dilutes FAR with easy
negatives that scale with group count, so the headline is now **same-group
trials only** (70: 53 known, 17 hard strangers — honest stranger count 3×
the old 6, granularity 5.9 %) with cross-group kept as a separate big-store
diagnostic. New baseline: DIR **88.7 % @ FAR 0 %** (threshold 0.605), known
ceiling 92.5 % (4/53 wrong-profile confusions no threshold reaches), and at
the shipped 0.5 default hard-stranger FAR is **11.8 %** — the default is
measured too permissive; 2.4 fixes it. Pre-2026-08-07 naming numbers pooled
both populations and are not comparable to successors (`eval/README.md`).

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
2. **Boundary margin — DECLINED 2026-08-06, measured and explained.** The
   TST-Bench +0.26 DIR presumes an intersection that loses words falling
   outside every turn; ours snaps them to the nearest turn, which already
   splits each gap at its center — a point no symmetric clamped pad can move.
   The sweep (`eval/margin_sweep.py`, 0.05–0.25 s, ten loop channels) measured
   the residue: mean word attribution 87.77 % unpadded vs 87.74–87.77 % at
   every margin, ≤0.5 % of words moving at all, and padded *turns* worsen DER
   on 8/10 channels (`eval/README.md`). Re-open trigger: an intersection rule
   that no longer assigns every word.
3. **Over-clustering bias + merge-at-naming — SHIPPED 2026-08-06, all three
   halves measured** (`eval/split_recovery.py`, `eval/kplus1.py`,
   `eval/collapse_probe.py`; numbers in `eval/README.md`):
   - **Merge-at-naming**: `SpeakerReID.resolve` is many-to-one — the
     exclusivity constraint measured strictly worse everywhere (blocked every
     profiled split recovery, and *forced* an over-split cluster onto a wrong
     profile when the right one was claimed). Recovers estimate-split profiled
     speakers to 100 % word attribution on 6/6 channels; the threshold, not
     exclusivity, is the false-accept control (one stranger named in twelve
     known-count namings at 0.5 — step 2.4's territory).
   - **Unprofiled recovery** = `collapse_single_voice`: an estimated channel
     whose clusters are ALL mutually ≥ 0.6 similar is one voice. The
     discriminator gap is empty in [0.45, 0.74]: split solo channels 0.74–0.98
     (10/10 recovered to 100 %), synthesized true-2-speaker channels 0.00–0.45
     (0/24 falsely collapsed), larger groups ≤ 0.18. The plan's other
     candidates are dead: estimator confidence does not exist (stenodiar emits
     turns only), dominance does not separate (46 % occurs on both sides), and
     pairwise self-merge without the all-pairs gate is catastrophic
     (cross-speaker cluster means reach 0.95; one loop chain-merged 7→1,
     93 → 44 %).
   - **Known counts** run at k+1 and fold the most-similar cluster pair back
     (`fold_excess_clusters`): +2.3 pts mean word attribution, one channel
     +20.8 (exact-k had fused two voices), worst channel −0.3, and the user
     always sees exactly the count they stated. Raw k+1 (phantom speaker) and
     naming-only recovery (wrong merges, −1.3 on IS1009d) are declined as
     shipped forms. "Bias the estimate up" needed no knob: the estimator
     already over-splits (est k ≥ true k on 30/30 measured channels); recovery
     is the mechanism.
4. **Overlap-clean embeddings — SHIPPED 2026-08-06**, and the gate that
   failed on the way is the important part: cleaning broke the step-1.3 fold,
   whose most-similar-pair criterion turned out to have been *luck* riding on
   overlap-inflated similarity around the tiny spare cluster (first matrix
   gate: 90.1 → 83.1 % attribution, DIR 88.2 → 66.7 %). The fold now picks
   the spare by duration and only its partner by similarity — bounded damage,
   embedding-insensitive, 90.1 % under both embedding types — and cleaning
   stays, with the collapse band measured wider (2-speaker ≤ 0.39 vs solo
   ≥ 0.73). Full story and the four-arm table: `eval/README.md`.
5. **Minimum-audio gate for naming — DECLINED 2026-08-06, measured.** The
   short-turn cliff is real on our stack (`eval/naming_gate.py`: known
   speakers' top-correct score 0.90 full → 0.40 at 1 s of clean speech) but
   it lands entirely on the miss axis: strangers' scores stay ≤ 0.37 at every
   truncated duration, so the 0.5 threshold already rejects what the gate
   would gate (FAR 0.0 % in every short arm). The shipped k+1-fold leaves no
   sub-3.2 s clusters to gate, and in estimate mode the only sub-5 s cluster
   that reaches naming is named correctly while both wrong namings are long
   (9.5 s / 75.1 s — clustering confusion, steps 4–5's territory). A gate
   would have prevented nothing and cost one correct naming plus its
   merge-at-naming recovery (`eval/README.md`). Re-open trigger: step 2.4's
   sweep picks an operating threshold below ~0.4, where short-stranger
   maxima live.
6. **Docstring fix — DONE** (312c0a2): `diarization/sherpa.py` says ERes2Net
   and points at `assets.SPEAKER_EMBEDDING` for the CAM++ rejection.

## Step 2 — profile store v2 + the enrollment loop

The research indicts all three quiet assumptions in today's re-ID design:
the single mean embedding, the flat 0.5 threshold, and enrollment as a
one-shot act.

1. **Per-meeting embeddings, score averaging — SHIPPED 2026-08-06, gated on
   `eval/store_v2.py`.** `_STORE_VERSION` 2: a profile holds up to 8
   per-meeting embeddings (oldest meeting date evicted beyond the cap,
   undated v1 migrations first) instead of one running mean; match score =
   mean cosine against the stored set (score averaging measured 2.05 % vs
   2.85 % EER for embedding averaging on identical data; the i-vector-era
   "average the embeddings" rule is explicitly retracted for modern
   embeddings). The harness gate (a+b enrollment, c–d+ICSI trials): all arms
   tie at full duration (11 known trials — single-trial granularity), but at
   2 s of clean trial audio a single enrollment collapses 90.9 → 81.8 %
   DIR@FAR0 while both combined-mean forms hold 90.9 %, and max-cosine
   collapses with the single — multi-meeting mean is measured insurance;
   score-vs-embedding averaging is within noise locally and ships on the
   literature (`eval/README.md`). Migration: a v1 profile's mean becomes its
   first stored embedding. Threshold semantics unchanged (max-over-gallery vs
   threshold — margin-to-second-best is unvalidated in the speaker
   literature; not adopting it).
2. **Rename-once online enrollment — SHIPPED 2026-08-06, gated on
   `eval/rename_once.py`.** Every diarized meeting now writes a
   `voiceprints.json` sidecar (each speaker's cluster embedding under the
   label the transcript shows), and `steno profiles assign LABEL NAME
   (MEETING|--last)` turns one correction into everything at once: the
   embedding joins NAME's profile (enroll-or-reinforce, dated by the
   meeting), the transcript files are rewritten with the name, and the
   sidecar follows. The gate measured the research claims in sign on our
   stack: at equal enrollment coverage, cluster enrollment matches the
   clean-headset arm at every practical operating point and *beats* it on
   2 s trials (DIR@FAR≤5 85.7 % vs 78.6 %); reinforcement on top of a clean
   enrollment is never worse and better at 2 s FAR-0 (82.4 % vs 76.5 %). The
   two measured costs are both the *enrollment meeting's diarization*, not
   the flow's: an impure cluster enrolls an impure profile (IS1009a fused
   FIO084 into FIO087's cluster → 0.789 leak, later false match at 0.860 —
   no threshold fixes it; steps 4–5 own the confusion, `eval/README.md` has
   the probe), and a fused-away speaker has no cluster to assign.
   **Solo channels included 2026-08-07, gated by the meeting's diarization
   switch** (Daniel's call: the switch decides). The counts alone cannot
   express "1:1 with the switch on" — diarization-off also collapses to
   count 1 — so `MeetingProfile.diarization` now carries the switch and the
   speaker machinery loads from it. A solo channel still never diarizes
   (nothing to separate): it gets *one* embedding over its transcribed
   speech (`Diarizer.channel_embedding`), which re-ID matches like any
   cluster — a 1:1 counterpart is named automatically after one assign, and
   both sides land in the sidecar. Evidence is the matrix's own mic trials
   (solo channels by construction): 6/6 known named at 0.935–0.970, all 10
   strangers ≤ 0.258 — a far wider margin than cluster naming; span choice
   (segmentation vs ASR spans) moved a solo embedding by ≤ 0.001 cosine in
   the rename-once run. Switch off keeps the zero-cost path: no models, no
   embeddings, no sidecar.
3. **Gated automatic updates — DECLINED 2026-08-07, measured**
   (`eval/auto_update.py`; numbers in `eval/README.md`). The research's
   poisoning warning reproduces on our stack as *ungateability*: replaying
   sessions b+c through the deployed loop absorbs three wrong updates — all
   IS1009's fused-away FIO084 pulled into colleagues' profiles — at scores
   0.609–0.860, top-minus-second gaps up to 0.855, and 40–721 s of clean
   speech, above every implementable bar (score margin, second-best margin,
   duration; the wrong clusters are among the *longest*). Meanwhile the
   oracle upper bound (append only correct namings) ties the no-update
   baseline at every held-out full-duration and 3 s operating point: nothing
   measurable to gain, unfilterable poison to absorb. Profile growth stays
   user-confirmed (step 2.2's assign); the never-evict anchor is moot with no
   automatic path, since every stored embedding is user-confirmed. Re-open
   triggers: steps 4–5 fix the clustering confusion behind all three wrong
   updates (re-run the harness then), or a trial set that can resolve
   sub-20-pt (single-trial) benefits.
4. **Threshold — SHIPPED 2026-08-07, 0.5 → 0.62, measured on both enrollment
   arms** (`eval/threshold_pick.py`; curves in `eval/README.md`). The old 0.5
   accepted 11.8 % of hard strangers (headset arm) and 25 % (cluster arm);
   0.62 is the smallest threshold rejecting every *non-pathological* stranger
   in both arms at every measured duration (clean strangers top out at 0.597
   — 0.62 keeps a 0.023 margin where 0.60 kept 0.003; cluster-arm clean
   strangers at 0.505), at zero full-duration DIR cost (88.7 % headset /
   90.0 % cluster, identical to 0.5) and solo/1:1 margins untouched (known
   solos 0.870–0.970). What remains above 0.62 is only the IS1009
   impure-enrollment leak (0.860/0.681/0.622) — diarization confusion that
   no threshold fixes (FAR0 on that arm would cost 26 pts DIR), owned by
   steps 4–5; re-run `threshold_pick.py` after them. The measured cost lives
   in truncated regimes (cluster arm at 2 s: DIR 42 → 30 % vs 0.5) — mostly
   artificial for the shipped k+1-fold path (no sub-3.2 s clusters), real
   only for estimate-mode short clusters. Step 1.5's re-open trigger (<0.4)
   did not fire. AS-Norm/QMF stay un-adopted: the curve is satisfying —
   the residual FAR is not a calibration problem, it is a clustering purity
   problem.

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
