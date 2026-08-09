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
   **Trigger fired and re-run 2026-08-07 post-ward: DECLINE REAFFIRMED.**
   The IS1009 poison dropped below every gate (0.655/0.688), but ward-era
   clusters supply new ungateable wrong updates — Bmr024's 13 s cluster into
   the wrong profile at 0.838 with a 0.603 top-minus-second gap, past the
   margin-.20 gate — and the oracle still ties no-update at the shipped
   threshold at every duration (`eval/README.md`). The next diarization
   change re-runs this harness as part of its gate.
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
   **Re-run 2026-08-07 after step 4, as owed: 0.62 → 0.56.** Ward removed
   the strangers 0.62's margin was bought against (max stranger now 0.531
   anywhere, the IS1009 leak gone); 0.56 is the cluster-arm FAR0-strict
   point, holds FAR 0 / FRR 0 at full duration in both arms with a 0.029
   margin, and recovers +13.7 pts DIR at 3 s / +5.9 at 2 s over 0.62
   (`eval/README.md`). Any future stride or model change re-runs this
   calibration as part of its gate.

## Step 3 — embedding model upgrade — DECLINED 2026-08-07, both candidates measured

The premise half-died in research and the rest died on the harness
(`eval/embedder_ab.py`, table in `eval/README.md`). **English/VoxCeleb
ERes2NetV2 was never released** — 3D-Speaker #208: not open-sourced, no plan
to — so the 1.48 % @ 2 s promise belongs to a model that does not exist; the
only V2 is Chinese-200k-trained. Measured on the full matrix (each candidate
diarizes AND names, both seats):

| arm | loop DER | loop attribution | DIR@FAR0 full | 3 s | 2 s |
|---|---|---|---|---|---|
| ERes2Net-base (shipped) | **18.0 %** | **89.7 %** | 88.7 % | **79.2 %** | **49.1 %** |
| ERes2NetV2 zh-200k | 23.8 % | 83.1 % | 91.5 % | 10.6 % | 12.8 % |
| ResNet34-LM (pyannote's) | 24.8 % | 82.1 % | 70.0 % | 0.0 % | 0.0 % |

Both candidates regress clustering ~6–7 pts of word attribution, and the
short-turn axis — the step's whole motivation — *collapses* instead of
improving: their cosine distributions compress upward on our domain (FAR0
thresholds 0.820/0.878 vs 0.605), pushing short-cluster knowns under the
stranger ceiling. ResNet34-LM's "known-good pairing" is a pyannote-pipeline
fact that does not survive sherpa's clustering. The shipped
ERes2Net-base stays; the full-duration naming ceiling (92.5 %) is clustering
confusion, not embedding quality — steps 4–5 territory.

Watch triggers (unchanged in the declined list): ReDimNet2 with a CC-BY
vox2 checkpoint and working ONNX export; additionally recorded from the
2026-08-07 hunt: `iic/speech_eres2net_large_sv_en_voxceleb_16k` (EN, EER
0.57 %, Apache-2.0 claim, PyTorch-only, absent from every supported export
path) — worth an export attempt only if a future step is actually limited by
full-duration embedding accuracy, which today none is. CAM++ stays rejected
(`assets.py`).

## Step 4 — own the diarization loop (clustering upgrade + stride)

sherpa's `OfflineSpeakerDiarization.process()` is a monolith: to change
clustering or stride we take over the loop, keeping the *models* and the
`SpeakerEmbeddingExtractor` — run segmentation-3.0 via onnxruntime directly,
then our clustering. Reference semantics extracted line-by-line into
`eval/diarization-loop-spec.md` (2026-08-07 — windowing, powerset order,
embedding gates, FastClustering cut semantics, the two-frame-scale gotcha,
all cited and falsification-tested).

**Phase A — the owned loop — BUILT 2026-08-07, parity gate PASSED**
(`stenograf/diarization/loop.py`, `eval/loop_parity.py`): raw `diarize(k+1)`
on all 20 loop channels, mean ΔDER **−1.41 %** (own slightly ahead), 14/20
within ±0.6 pts, runtime equal. The six divergers swing both ways (own −23.6
on TS3010a … +8.2 on TS3010d): AHC merge fragility on confusion-heavy
channels, where tiny embedding deltas (our stream concatenates audio, sherpa
concatenates feature frames) flip discrete merges — the same fragility
Phase B exists to fix, not a systematic bias.

**Phase B — clustering + fold SHIPPED 2026-08-07: ward linkage with the
similarity-gated fold** (`OwnDiarizer` default `ward`;
`pipeline.FOLD_PAIR_SIMILARITY = 0.8` in `fold_excess_clusters`). Loop DER
16.5 → 13.2 %, attribution 91.3 → 94.8 %, enrolled speakers >50 % misnamed
5 → 1 of 35, worst per-channel regression +0.5 pt; the full evidence chain
(partitioner sweep, fold matrix, gate audit, k=2 duo channels, by-reference
naming) and both adversarial reviews are in `eval/README.md`. The decisive
finding: **the partitioner and the fold rule are one decision** — the
duration-spare fold was co-tuned to complete-linkage clusters and both
manufactured ward's only bad regression (Bmr025 +21.3, a split dominant
talker the smallest-cluster rule cannot repair) and discarded 2.6 pts of
ward's oracle-fold bound. Declined with numbers: VBx-as-assumed (no
shippable model/PLDA — spec §3), NME-SC (criterion inversion, structural
failure on ES2007d; best-fold 14.4 % vs 13.2), centroid (linkage inversions
degenerate maxclust: 54.3 %), average (40.5 %), ungated max-pair (24.7 % on
complete), ward@k (17.6 %, the split persists without the spare), the 0.6
gate (4 cross-speaker admissions, caught by the fold-gate audit). Heavy
fallback (two-covariance PLDA via WeSpeaker + stock VBx) stays recorded,
re-open trigger = a Phase-B-style residual the gate audit can't explain.

**Stride DECLINED 2026-08-07 at 2 s and 3 s, measured** (the research
record's "8.5× at ≈0 DER cost" did not survive our harness): DER
13.2 / 13.5 / 14.4 % and attribution 94.8 / 94.4 / 93.5 % at strides
1/2/3 for 2.1× / 3.1× speedups — and the FAR0 naming threshold jumps
0.592 → ~0.78 at both larger strides (coarser boundaries contaminate every
cluster embedding), silently invalidating step 2.4's calibration (trial FAR
at 0.62: 0 → 5.0 → 13.6 %; by-ref stranger misnaming 12.8 → 12.9 → 18.5 %).
Evidence in `eval/README.md`. Re-open trigger: finalize latency becomes
binding or step 5 needs the budget — then per-chunk embedding first (mask at
stat-pooling, attacks the contamination mechanism), and any stride change
re-runs the threshold calibration as part of its gate. `OwnDiarizer(shift=…)`
stays a parameter; the default is the reference 1 s.

**The production swap SHIPPED 2026-08-07 — step 4 is COMPLETE.**
`build_diarizer` constructs `OwnDiarizer` (every platform — the harness that
measured ward+gated-fold ran on this Mac, so scoping the win off-mac would
have declined a measured improvement for no reason); stenodiar/speakrs keeps
the estimate seat unchanged (`SpeakrsCliDiarizer` wraps the owned loop). The
old path is DELETED, not kept: sherpa's `OfflineSpeakerDiarization` monolith
is called nowhere in production — `sherpa.py` is now the embedding base
(extractor + `cluster_embeddings` + re-ID plumbing) that `OwnDiarizer`
subclasses, and sherpa-onnx stays a dependency for exactly that. The
stenodiar-less estimate fallback is the loop's threshold mode (reference
complete cut, documented in `_cluster`). The parity gate stays re-runnable:
`eval/loop_parity.py` constructs the reference from the installed sherpa
package directly. Real-backend coverage moved with the swap
(`tests/test_diarization_loop_real.py`; the unit aggregation tests drive
`cluster_embeddings` through the real `embed` with a fake extractor).
Deferred per-chunk embedding (mask at stat-pooling) stays recorded above as
the stride re-open's first move.

## Step 5 — the DiariZen gate — DECLINED 2026-08-09, measured in three arms

`BUT-FIT/diarizen-meeting-base` (MIT, ungated) was the one shippable model
clearly above the pyannote family on paper (AMI-SDM 15.6 vs 21.1). On the
harness it loses to the shipped stack (`eval/diarizen_arm.py`, decision rule
pre-registered in its docstring; full evidence in `eval/README.md`): at the
fair replacement boundary — DiariZen at k+1 folded by the production
`fold_excess_clusters`, since the fold lives in the pipeline half step 5
never replaces — loops go **+1.3 mean / +1.2 median ΔDER against it** (wins
4/20, sign p≈0.012, attribution −1.1 pts), and the k=2 duos, the near-field
go/no-go, go +2.0/+1.3 (wins 3/20). Exact-k (+11.7) and estimate-as-published
(17.1 % vs 14.5 % loop mean) are both worse framings for it; mcs swept —
the checkpoint's own hyperparameters are not the excuse; the 0.8 fold gate
transfers to its clusters (22/22 gate-admitted merges same-speaker, crosses
≤ 0.702). The ONNX-export program never opens. **Re-open trigger: the
product's meeting mix shifts toward many-speaker rooms — DiariZen won all
four ICSI loops (k=4–8, −0.4 to −2.5 DER, attribution +0.8 to +2.8; n=4, one
meeting series, but the ICSI duos flip back to production, so the signal
tracks speaker count, not corpus) while losing the 3-speaker AMI loops and
their duos 31/32.**

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
