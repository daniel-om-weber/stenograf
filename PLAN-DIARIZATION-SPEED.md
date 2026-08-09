# PLAN-DIARIZATION-SPEED — steps and gates

Scoped 2026-08-09, replacing the same-day draft discussion notes (`git log
-p` this file for them). Accuracy is settled — the closed diarization
program's verdicts stand; this plan changes *when* and *how fast* the same
computation runs. The draft's estimates were adversarially reviewed the same
day (three independent measurement sessions on the M4 Max, one channel of
real in-room audio each); every number below marked **measured** was taken
then unless it carries an older date, and each gate re-measures on its own
inputs — none of the review's scratch scripts are load-bearing.

## What the review corrected before any step ran

- **The 111.7 s baseline is the known-count path, and that is not the
  default** (measured, from code). `build_diarizer` returns
  `SpeakrsCliDiarizer`, which forks on speaker count
  (`diarization/speakrs.py`): known count → `OwnDiarizer` (the measured
  2 361-embed loop); auto-detect — the default, and what the setup form
  recommends — → the stenodiar subprocess, then ~one embedding per cluster
  span. Only when stenodiar is absent (the manylinux_2_28 low-floor wheel,
  i.e. the oldest hardware) does the owned loop take both cases. The
  "under a second per meeting-hour" warm-helper figure in `speakrs.py`'s
  timeout comment is **assumed**, never measured — hence step 0.
- **Parallelization's headroom was quoted against the wrong denominator.**
  The shipped loop already runs `min(8, cpu)` intra-op threads
  (`diarization/sherpa.py`), worth 1.50× over single-thread on the embed
  stage (measured). An 8-worker pool at intra-op 1 beats *shipped* by
  2.5–2.6× (measured, interleaved re-runs ±2 %), not the draft's 3–6×; a
  4-core mobile Intel projects to ~1.9× net (**inferred** — step 4 measures
  it).
- **Batching the embedder is a 3.8× pessimization** (measured, reproduced
  interleaved): ORT's conv path collapses above batch 1 on this model
  (conv time 8.6× for 2× work; 85.6 % of the profile is conv). Batching
  segmentation: 1.14× at batch 32. Pooling segmentation instead: 3.07× at
  12 workers with max|Δ| = 0.0 and zero powerset argmax flips (measured).
  The draft's "DiariZen drives batch 32" was a GPU argument imported into a
  CPU setting.
- **The draft's parity tolerance would have failed its own change for a
  spurious reason.** Changing intra-op 8 → 1 shifts embeddings 3.4e-7 —
  above the borrowed 1.9e-8 — by float reduction order alone; the pool
  itself, against an intra-op-1 sequential reference, is **bit-exact** at
  2/4/8/12 workers, shared or per-worker extractors, 318 real slices
  (measured). The gate in step 2 is built from that split.
- **Extractor thread-safety: safe as exercised.** One shared
  `SpeakerEmbeddingExtractor` across 8 workers is bit-identical to
  sequential and no slower than per-worker copies, which would cost
  8 × 26 MB (measured; one platform, one model — the gate re-checks on
  Intel). ORT `Run` is documented thread-safe; sherpa's per-call fbank
  state lives in `create_stream()`.
- **The structural redundancy is confirmed and is the bigger prize**:
  13 192 s of audio embedded for a 2 251 s channel — 5.9× the file, 9.8×
  its speech (measured on the frozen ES2003c.loop reference). Embed cost is
  affine in input length (6.5 ms + 5.9 ms/s, predicts the production 93.4 s
  to 0.2 %), so removing embedded seconds pays linearly and identically on
  every platform, where pooling pays less the weaker the chip. The
  algorithm needs one *vector* per (chunk, speaker) pair for the vote; it
  does not need one *trunk pass* per pair — the floor is ~10× under today.
- **A recorded re-open path was ambiguous, and one reading is backwards**
  (priced with the validated cost model, not run): the declined-stride
  bullet's "per-chunk embedding, mask at stat-pooling first" read as
  one-trunk-pass-per-10-s-window would embed 2 248 full windows = ~147 s,
  *slower* than today's 93 s, because `_pair_embeddings` already crops to
  sole-speaker audio (`diarization/loop.py`) — the 21× framing never
  applied here. SDBench's actual "per-chunk" method runs the trunk over
  the input audio once ("embedding exactly 30 seconds of audio for a
  30-second audio input") — that is step 5's L1, with a published accuracy
  prior of DER 0.255 → 0.257. Step 1 disambiguated the record
  (adversarially re-verified against the source text same day).
- **The online idea's costs were understated.** Notes consume speaker
  labels (`notes/prompt.py` renders `speaker [ts]: text` and instructs
  attribution), so diarization can never leave the notes critical path by
  running beside it. A worker's sustained load on a weak chip is the most
  plausible trigger of live-decode shedding, and a shed channel loses
  finalize ASR reuse entirely (`session.py`, `reuse_live_finalize`) — the
  failure mode is a net *slowdown*, not neutrality. Hybrid meetings
  diarize two channels (~2× the draft's duty cycle), fan noise during a
  meeting raises the floor of the very mic being diarized, and byte-parity
  across mixed online/offline runs pins session config (threads, EP)
  identically across all paths forever. Step 6 decides with gates that can
  kill the design before any worker code exists.

## Step 0 — path-cost split — DONE 2026-08-09, the split is 8.7×

Measured (M4 Max, ES2003c.loop 37.6 min, `diarize_with_embeddings` — what
finalize calls — timed directly, ASR/notes excluded):

| arm | time | RTF | notes |
|---|---|---|---|
| known-count (owned loop) | 126.2 s | 0.056 | probe passed k=3; production passes k+1 — the expensive stages are k-invariant (k only enters `_cluster` and the fold), so production is ≥ this and the split is conservative. First-arm lazy loads (ORT session, 26 MB extractor) push the other way; both effects are small against 126 vs 14 |
| estimate (process-cold, models cached) | 14.1 s | 0.0063 | NOT first-run cost — CoreML artifacts were already compiled and cached; a true first run takes minutes (`speakrs.py`) |
| estimate (2nd run) | 14.4 s | 0.0064 | ≈ 3.4 s helper + 10.0 s `cluster_embeddings` (5 clusters), decomposed in a separate run (~1 s transport unaccounted); run-to-run noise ±2 % |

Two facts fall out beyond the split itself: the helper is ~5.4 s/meeting-
hour (the "under a second" comment in `speakrs.py` was ~5× optimistic,
corrected), and **`cluster_embeddings` is a first-class ~10–15 s/channel
cost on BOTH paths** (the ~14.5 s known-count figure is inferred by
cross-run subtraction, ±2–3 s; the 10.0 s estimate-path figure is timed
directly) — step 2's pool must cover its embed calls too, and step 5 L1
would make it free.

Scope, therefore: this program's beneficiaries are (a) users who state
counts — which the product recommends because exact counts label speakers
better — and (b) the manylinux_2_28 / no-stenodiar tier, where the owned
loop takes both cases. On the default estimate path diarization is ~14 s
per channel (measured); whether notes then dominates that wait is
plausible but **unmeasured until step 4's split**. Every later step is
weighed against that scope. (The estimate arm found 5 speakers where the
reference has 3 — a real caveat on the cheap path, since notes consumes
those labels; the accuracy gap itself is the closed program's territory,
not this plan's.)

## Step 1 — correct the record (no code)

- PLAN.md declined-stride bullet and `eval/README.md`'s matching passage:
  replace "per-chunk embedding (mask at stat-pooling) first" with the
  shared-trunk formulation (one trunk pass over long blocks, masked
  statistics pooling per pair — step 5's L1, = SDBench's method with its
  DER 0.255 → 0.257 prior), and record the disambiguation: the
  per-*window* reading was priced at ~147 s vs 93 s here because our loop
  already crops to sole-speaker audio.
- `eval/diarization-sota-2026.md`: the 21× / DER 0.255→0.257 numbers are
  SDBench (arXiv:2507.16136), not arXiv:2606.08505 (which is the
  relative-minimum-cluster-size paper, on CAM++, containing neither
  number).
- `eval/diarization-loop-spec.md`'s "~2 active speakers per chunk":
  measured 1.05–1.08.

## Step 2 — the pool — DONE 2026-08-09, both gates green, 2.86×

Shipped: one shared `ThreadPoolExecutor` per diarizer (workers = physical
P-cores, `_pool_workers()`), every ORT session at intra-op 1, one shared
extractor; `_chunk_labels`, `_pair_embeddings` AND `cluster_embeddings`
submit into the same budget (step 0 showed the last is ~10–15 s/channel on
both paths). Measured (`eval/pool_parity.py`, table in `eval/README.md`):
111.7 → **39.1 s** on the reference channel (2.86× vs shipped; 4.7× vs
sequential intra-op-1). Both gate halves green: bit-exact vs the sequential
reference at 4 and 12 workers, and the full pooled pipeline byte-matches
`ami-loop-ward-sv08` on all 20 loop channels (max emb drift 1.08e-7 — the
intra-op 8 → 1 flip moved no decision). Gate lesson recorded in
`eval/README.md`: `ami-loop` was a stale baseline (predates
ward-by-default); the loop is deterministic run-to-run, so a mismatch vs
the RIGHT baseline is always real.

**Channel-parallel finalize: dropped as redundant, by design.** The shared
pool is work-conserving — one channel already saturates the P-cores, so
two channels' diarize running concurrently through the same pool buys
~nothing over serial channels through that pool (only the ~1 s serial
clustering and the 3.4 s estimate-helper subprocess could overlap).
Restructuring `session.py` for that margin fails the cost/benefit that
motivated it — the 2× claim assumed the pre-pool sequential regime.
Re-open only if a real finalize profile shows dead pool time between
channels.

Still owed from this step: finalize watts (race-to-idle check) — folded
into step 6's gate session, which needs the same quiet machine (the
desktop was loud when the gates ran); weak-Intel scaling is step 4's.

## Step 3 — warm the notes model during the meeting — DONE 2026-08-09, benefit smaller than projected

Shipped: `notes/warm.py` (`NotesWarmer`) — a dedicated thread started by
`MeetingRun.run` once capture and the ASR are up; after a 45 s grace and a
12 GiB free-memory gate it builds the exact backend the notes tail would
(`notes/run.py::resolve_backend`, same arguments) and runs `MlxBackend.
warm()` (load + one token, binding generation to that thread); the notes
tail then executes on the warmer's thread with the prebuilt backend. Cold
path is the fallback by construction (short meeting, low memory, warm
failure, cancel); Ollama is deliberately not warmed (server `keep_alive`
evaporates before a normal meeting ends). Ctrl-C semantics preserved from
the waiting thread. Gates green on a 150 s isolated-settings replay: warm
fired mid-meeting, both channels kept "reusing live decodes" (shed exactly
0), notes wrote through the warmer.

**Honest outcome: the projected win largely evaporated on measurement.**
Cold mlx load + first token is **2.3 s** page-cache-warm on the M4 Max
(mmap lazy loading; the "4.35 GB cold load" was an unmeasured projection —
the arch review's best benefit-per-risk ranking was wrong on this machine).
Ceiling is a truly cold cache (first meeting after boot), plausibly a few
seconds more, **unmeasured**. The thinking-mode generation dominates the
notes tail regardless. Kept because it is small, gated safe everywhere,
and the cold-cache first-meeting case is the realistic one — but this step
would not have been built at this priority had the 2.3 s been measured
first. (Measure-the-comparator lesson, same shape as step 0's.)

## Step 4 — the weak-box datapoint, reframed as the whole wait

One session on the CachyOS notebook or the GPD, one real (or `verify`-
skill) meeting: record the **entire** end-of-meeting wait split —
diarization (and which path), notes model load, notes generation — plus the
intra-op sweep and pool sweep from step 2's harness (<5 min compute) and a
batch>1 embed timing (the conv collapse is measured on ARM; x86's NCHWc
layout may behave differently — if batching *wins* there, step 2 grows a
per-platform branch). Suspicion to test, not assume: on a CPU-only box with
Ollama in thinking mode, notes may dominate the wait there too, which would
invert the draft's "weak Intel is the online worker's strong case".

## Step 5 — remove embedded seconds (accuracy-gated, the big lever)

Every arm here changes embedding values, so each costs a harness matrix
plus `threshold_pick.py` and `auto_update.py` re-runs; `loop_freeze.py`
cannot amortize any of it (it persists exactly what these arms change, and
is sha-stamped against stale reuse). Kill-tests come before matrices.

- **L3 first — embed each maximal contiguous sole-speaker run once.**
  ~241 calls / ~9.4 s instead of 2 361 / 93.4 s on the reference channel
  (priced with the validated cost model); ~30 lines in `_pair_embeddings`,
  no ONNX work, and the speedup is hardware-independent. Risk is the
  clustering unit: fewer, longer vectors, and a run spanning a missed
  speaker change becomes one contaminated vector (the TST-Bench failure
  mode). Kill-test: one channel — cluster count, DER, naming cosines vs
  reference; buy the matrix only if it holds.
- **L2 as the attribution probe** (cheap, do with L3's kill-test): embed
  every Nth chunk, propagate labels by frame overlap; `speakers_per_frame`,
  `starts` and the vote grid stay byte-identical to today, so the
  boundary-coarsening mechanism named in the stride kill is not engaged.
  The declined bullet's damage attribution (clustering vectors vs
  `cluster_embeddings()` naming vectors) is **inferred, never isolated** —
  at stride 2 the surviving offsets are a subset of stride 1's, so
  clustering vectors can only be removed, not contaminated. One N=3 run
  diffing naming cosines tests the attribution the declined bullet rests
  on; if confirmed, note it on the bullet (its scope is narrower than it
  reads).
- **L1 behind L3 — one shared trunk pass + masked stat-pooling** (=
  SDBench's "per-chunk" method, which carries a published accuracy prior:
  DER 0.255 → 0.257 on their benchmark — a measured data point for this
  exact architecture, not a guess). The embedder trunk is time-local
  (verified: the only global-over-time ops are the three ReduceMeans in
  `/pool/`; the graph splits cleanly and trunk = 97–103 % of full-model
  cost), so one pass over long blocks plus O(1) masked mean/std per pair
  from prefix sums reaches the same ~10× and also makes the naming-stage
  `cluster_embeddings()` calls free — measured at step 0 as ~10–15 s per
  channel on both paths. Two seams break value-equivalence honestly: CMN scope
  (sherpa subtracts a per-call global mean *outside* the graph — dropping
  it moves cosine 0.86 → 0.33, measured) and mask-edge receptive-field
  bleed. **Precondition before any matrix**: our own fbank front end must
  hit cosine > 0.9999 against sherpa's on ~100 clips; a day's attempt
  reached only 0.855 — if parity can't be reached, L1 dies on
  implementation risk at zero matrix cost. L1 is also the enabling move
  for Intel EPs (below); pursue it only if L3 moves the accuracy numbers
  or the EP need materializes. Blocked trunk processing required — a
  whole-file trunk output is ~576 MB.

## Step 6 — decide the online worker against the post-step-2/5 numbers

Structural facts stand: only clustering is global (verified — non-tail
chunks are pure functions of their window), `SessionStore` is
prefix-immortal so input byte-parity is free, and crash fallback is the
offline path by construction. Build it only if ALL of these hold on the
weakest supported box, on a real hybrid meeting:

1. **Share** — post-step-2/5 diarization ≥ 40 % of the total wait
   (diarization + notes, step 4's split).
2. **Absolute** — post-step-2/5 diarization ≥ 90 s there.
3. **Non-regression** — with a synthetic load at the worker's duty cycle
   beside a full `--replay` meeting, `shed_by_channel` stays exactly 0.0.
   Any shedding kills the design (shed = finalize re-decodes ASR).
4. **Thermal** — package watts and fan state during that loaded replay
   indistinguishable from unloaded, or the policy is AC-only — and an
   AC-only policy is itself evidence against building it (the policy layer
   and the started-then-stopped mixed state raise the second-code-path
   cost, they don't halve it).
5. **Path coverage** — step 0 established the owned loop actually carries
   enough real meetings to matter.

Gates 3 and 4 need no worker code: trickle-pace the existing `OwnDiarizer`
over a growing prefix beside a replay and read powermetrics +
`shed_by_channel`. One afternoon; run them first and let them kill it
cheaply. If declined, record the trigger: a measured weak-box wait split
where diarization clears gates 1+2.

## Non-levers and declines, measured this review (don't re-spend)

- **Embedder batching**: 0.26× (see corrections above). Re-open only if
  step 4 shows x86 inverts it.
- **EPs through sherpa**: unreachable — sherpa's bundled ORT accepts any
  `provider` string and silently falls back to CPU (verified for openvino,
  xnnpack, directml). Every EP lever for the embedder requires L1's
  sherpa-bypass; segmentation already runs in our own session and is open
  to EPs today.
- **CoreML embedder EP**: 1.50× measured through sherpa, mac-only, 2 s
  compile per process, likely non-composing with the pool (ANE is one
  shared resource — cheap to test if ever wanted). Helps the platform that
  needs it least; skip.
- **int8 embedder**: declined, with the reason corrected — the "int8
  ruined German ASR" transfer is weak (that damage compounded through an
  autoregressive decoder; this is one feed-forward pass into cosine). The
  real reasons: ORT CPU int8 conv pays only with VNNI, absent on exactly
  the arbitrary-mobile-Intel target, and the recalibration bill (0.56 +
  ward + fold gates) is certain while the win is unmeasured. Re-open
  trigger: a measured >1.5× on real target hardware, not "never".
- **fp16**: no x86 CPU fp16 conv kernels in ORT — cast-to-fp32 makes it
  neutral-to-slower. ARM-only curiosity.
- **Graph-opt level / saved optimized model / IOBinding / arena tuning**:
  quantified non-levers (opt already `ORT_ENABLE_ALL` with fused convs;
  session init 0.05 s; feature+copy share 6.7 %; arena only matters with
  per-worker sessions, which the shared extractor obviates).
- **Slice-length cap (truncate embed inputs at ~3 s, 1.91× measured on the
  embed stage)**: subsumed by L3 (which removes the same seconds and more);
  revisit only if L3's kill-test fails for reasons a cap wouldn't share.
