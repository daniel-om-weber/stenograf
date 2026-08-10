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
the waiting thread. Gates green on two isolated-settings replays: a 150 s
run at `--local 1 --remote 1` (no diarizer loaded — warm fired
mid-meeting, both channels kept "reusing live decodes", shed exactly 0,
notes wrote through the warmer) and a 50 s run at `--local 2 --remote 2`
(meeting ends as warm fires: the mlx load ran concurrently with the
diarizing 12-worker finalize, clean).

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

## Step 4 — the whole wait on x86 — DONE 2026-08-11; the estimate path inverts off macOS

One session on the GPD, one 37.6-min `--replay` meeting on the reference
channel pair plus the sweeps. Numbers and machine caveats are in
`eval/README.md`; what they decide:

- **The whole wait is 328.4 s** on the known-count path: finalize 158.6 s
  (mic 8.5 + diarizing system 150.1) + notes 169.8 s (ollama qwen3:8b,
  single pass, CPU). Diarization is **45.7 %** of it. The suspicion this
  step existed to test — that notes would dominate on a CPU-only Ollama box
  — is **not** what happened; on the default path diarization is the larger
  half.
- **The path split inverts.** stenodiar on ORT CPU costs 241.9 / 278.8 s
  against 3.4 s on CoreML, so the estimate path (256–293 s) is 1.7–1.9×
  *more* expensive than the owned loop (154.2 s) here, where macOS makes it
  8.7× cheaper. Step 0's scope sentence is macOS-only: off macOS every user
  is a beneficiary of this program, and the default path is the expensive
  one.
- **The pool holds at a smaller ratio** — 1.46× over the reconstructed
  pre-pool config against the M4 Max's 2.86×, because x86 intra-op
  threading is comparatively strong (2.43× at 8 threads) — and the
  committed gate passes bit-exact at 8 and 12 workers, so the shared
  extractor now holds on two architectures.
- **Batching does not invert on x86**, so step 2 grows no per-platform
  branch, and the reason is stronger than the ratio (uniform-length inputs
  are batching's best case; padded batching needs L1's masked pooling).
- **Measurement lesson, worth more than any single number here:** repeats of
  one config span 11 % interleaved and 21 % across sessions on this box —
  larger than most effects worth chasing. An increasing-order worker ladder
  manufactured an 8-worker optimum that an interleaved 12/8/12/8 re-run
  refuted; `_pool_workers()`'s 12 stands. Interleave every arm on a handheld,
  and treat anything under ~15 % here as unmeasured.

**What this step did NOT settle, and is now the open remainder:** this box is
a 12-core part under a handheld TDP cap, not the 4-core mobile Intel the
corrections section projects ~1.9× net for. That projection stays
**inferred**, and the estimate-path inversion above is a *direction* that
follows from the execution provider, with a magnitude that does not follow to
fewer cores. Both want a genuinely small x86 box; neither blocks anything.

## Step 5 — remove embedded seconds — L2 and L3 DECLINED at kill-test 2026-08-09; L1 is the surviving path

The kill-tests ran on all 20 loop channels (`eval/embed_units.py`,
freeze-based — no matrix bought, per-channel numbers in the harness
section of `eval/README.md`):

- **L3 (contiguous sole-speaker runs, ~10× less embedded audio) —
  DECLINED.** Mean DER 13.7 % vs baseline 13.2 %, and the worst channel
  regresses **+3.5 pt** (Bmr030, k=4 — inside the product-weighted 3–5
  speaker range) with +1.9/+1.8 on two more. The closed program's shipping
  bar (ward: worst-channel +0.5 pt) is missed by 7×. The savings were
  real (9.9× on the reference channel) — the accuracy price is not payable.
  Re-open trigger: none from here; the seconds-removal budget goes to L1.
- **L2 (cluster every Nth chunk's vectors, labels propagated) —
  DECLINED, with a finding that narrows nothing.** Mean 14.3 %, and
  TS3010d is catastrophic: **+17.1 pt** (0.344 vs 0.173) with min naming
  cosine 0.756. The attribution the probe was built for came back MIXED:
  on ES2003c thinning clustering evidence 3× left naming cosines at
  1.0000 (the declined-stride bullet's mechanism claim holds there), but
  TS3010d/Bmr024/IS1009a show cluster-evidence thinning ALONE corrupting
  clusters and naming vectors with boundaries byte-identical — so "L2 is
  safe where stride wasn't" is **refuted**, and the declined-stride
  bullet's scope stays as written (do not narrow it).
- **L1 — one shared trunk pass + masked stat-pooling — now the ONLY open
  seconds-removal path**, and the kill-test outcome strengthens its case:
  L3 failed by changing the clustering units, and L1 keeps the
  per-(chunk, speaker) units exactly while computing them ~10× cheaper (=
  SDBench's "per-chunk" method, published accuracy prior DER
  0.255 → 0.257). The embedder trunk is time-local (verified: the only
  global-over-time ops are the three ReduceMeans in `/pool/`; the graph
  splits cleanly and trunk = 97–103 % of full-model cost), so one pass
  over long blocks plus O(1) masked mean/std per pair from prefix sums
  reaches the ~10× and also makes the naming-stage `cluster_embeddings()`
  calls free — measured at step 0 as ~10–15 s per channel on both paths.
  Two seams break value-equivalence honestly: CMN scope (sherpa subtracts
  a per-call global mean *outside* the graph — dropping it moves cosine
  0.86 → 0.33, measured) and mask-edge receptive-field bleed.
  **Precondition before any matrix**: our own fbank front end must hit
  cosine > 0.9999 against sherpa's on ~100 clips; a day's attempt reached
  only 0.855 — if parity can't be reached, L1 dies on implementation risk
  at zero matrix cost. L1 is also the enabling move for Intel EPs
  (below). Blocked trunk processing required — a whole-file trunk output
  is ~576 MB. This is the next session's opening move for this plan.

## Step 6 — the online worker — DECLINED 2026-08-11 on gate 4, all five gates measured

Four of the five gates pass on x86 and the design still dies, on the one
the plan wrote for exactly this. Per-gate numbers are in `eval/README.md`;
the shape of the result:

- **Share (45.7 %, ~60 % on the default path) and absolute (150 s, 256–293 s)
  pass**, which is what made the decision worth taking rather than assuming.
- **Non-regression passes**, over 900 s of a 2262 s meeting: with the
  worker's duty cycle running beside the live pass, every channel still
  finalized "reusing live decodes" — which finalize grants only on
  `shed_by_channel == 0.0` — at zero late windows.
- **Thermal fails.** Fan state is indistinguishable, package power is not:
  **+43 %** (5.09 vs 3.57 W) over the same window, loading only the channel
  production actually diarizes — far outside this box's 11–21 % repeat
  spread. Measured duty is 0.44 core under live-pass contention. Priced
  from the power rows (derived, not measured), the worker holds +1.5 W
  across the whole meeting ≈ 3.5 kJ where the offline finalize spends
  ~2.2 kJ in 150 s: race-to-idle wins on energy as well as on the gate. An
  AC-only policy was pre-declared as evidence against building, not a way
  past this.
- **Path coverage passes**, and more strongly than expected: off macOS the
  owned loop is the *cheaper* path.

Structural facts that survive the decline (they were verified, and a future
attempt should not re-derive them): only clustering is global, non-tail
chunks are pure functions of their window, `SessionStore` is prefix-immortal
so input byte-parity is free, crash fallback is the offline path by
construction, and gates 3/4 need no worker code at all — trickle-pace the
existing `OwnDiarizer` beside a replay.

**Re-open trigger: L1 landing — and it is not a free pass.** L1 removes
*embedding* seconds only; segmentation is untouched and is 11.3 % of loop
cost here (60.7 of 536.2 s), so a 10× embed cut leaves 0.20× of the work:
duty 0.44 → **~0.09 core**, and the power delta +1.53 W → ~+0.31 W, i.e.
~+9 % over the capture floor — inside this box's repeat spread, so at that
point gate 4 becomes genuinely undecidable by this method and would need a
paired, interleaved load-on/load-off design to answer. Re-run gates 3 and 4
on this box after L1; nothing else about the design needs revisiting first.

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
