# PLAN-DIARIZATION-SPEED — draft discussion notes, not yet a plan

Opened 2026-08-09, immediately after the diarization program closed. Status:
**notes for a scoping session** — nothing here is committed work; the next
session turns this into steps with gates, or closes it. Accuracy is settled
(the closed program's verdicts stand); this is purely about *when* and *how
fast* the same computation runs.

## The measured baseline (2026-08-09, M4 Max, ES2003c.loop, 37.6 min)

Production `OwnDiarizer.diarize` at k+1, per stage:

| stage | time | share | shape of the work |
|---|---|---|---|
| segmentation | 17.5 s | 16 % | 2 248 windows, **sequential, batch 1** |
| embeddings | 93.4 s | **84 %** | 2 361 `embed()` calls, **sequential, batch 1** |
| clustering + assemble | 0.7 s | <1 % | global ward over all vectors |
| **total** | **111.7 s** | RTF **0.049** | ≈ 3 min per meeting hour |

Context for any latency discussion: this is a *finalize* cost — the live
pass never waits on it — and the end-of-meeting wait also contains the
notes-LLM pass, which needs the finished transcript and can never move
online. What the wait is actually made of on a real meeting is unmeasured
(see "Missing measurements").

## Lever 1 — parallelize + batch (engineering only, no accuracy trade)

The 2 361 embedding calls are mutually independent; onnxruntime's `Run`
releases the GIL, and models this small do not saturate intra-op threading
(`_num_threads` caps it today). A thread pool across the P-cores is
plausibly 3–6× on the 84 % stage. The segmentation model has a batch
dimension we run at 1; DiariZen drives the same architecture at batch 32 —
plausibly 2–4× on the 16 % stage. Projected combined: **~112 s → ~30–45 s**
per 40-min meeting (estimate, to be measured).

Why this is the first move regardless of everything below: it speeds up
every path that exists — today's finalize, `steno transcribe` (offline
forever), and every catch-up/recovery path of a future online worker — on
every platform, weak hardware most of all.

Gates and known traps for the plan:

- **Parity gate on the harness**: same inputs, same outputs. Execution-order
  float jitter is expected — the precedent tolerance is the swap gate's
  emb.json ≤ 1.9e-8 with *no merge decision moved*; define the tolerance
  before the change, byte-compare turns.
- **Extractor thread-safety is unverified.** sherpa-onnx's
  `SpeakerEmbeddingExtractor` may need one instance per worker thread;
  measure both layouts.
- **Thread budget interplay**: outer parallelism × `intra_op_num_threads`
  must not oversubscribe (today both sessions get `_num_threads()` intra-op
  threads); likely intra-op → 1 per worker once the outer pool exists.
- Power: batch/parallel is race-to-idle — expected watt-neutral or better,
  but the finalize watt number has never been taken; cheap to record while
  gating (passwordless powermetrics).

## Lever 2 — already declined, recorded with its re-open path

Stride ≥ 2 s measured 2.1×/3.1× end-to-end but contaminates cluster
embeddings and silently invalidates the 0.56 naming calibration (FAR at the
shipped threshold 0 → 5 → 13.6 %). Declined in the closed program; the
recorded re-open path (PLAN.md declined list) is per-chunk embedding masked
at stat-pooling *first*, and any stride change re-runs the threshold
calibration inside its gate. Note the redundancy it attacks is structural:
at stride 1 every second of speech lies in ~10 overlapping windows and is
embedded up to ~10×.

## Lever 3 — speculative, measure before believing

- CoreML EP for the ERes2Net embedder (CNN, ANE-friendly; sherpa-onnx
  exposes the provider knob). The segmentation LSTM is typically a poor
  CoreML fit. Unmeasured.
- int8 embedder quantization: treat as effectively declined — int8 already
  ruined German ASR in this repo, and quantizing the embedder shifts the
  cosine scale, forcing a full threshold recalibration for an unproven win.

## The online idea — compute during the meeting, cluster at the end

Feasibility is structural and favorable: of the four stages, **only
clustering is global over the meeting, and clustering costs 0.55 s.**
Segmentation windows and per-chunk embeddings depend only on audio already
heard — a worker trailing the recording by a few seconds amortizes the
112 s across the meeting (~5 % of one P-core here; 15–30 % on a weak Intel).
At meeting end: cluster accumulated vectors, fold, intersect, name — seconds.

Design properties worth keeping from the discussion:

- **Pure precompute, identical semantics.** Same window boundaries, same
  models, deterministic; only the last few tail windows need the meeting's
  end. State is small: (labels, pairs, vectors) per channel. If the worker
  dies or lags, finalize computes whatever is missing — the offline path is
  the fallback by construction, which is only comfortable because lever 1
  makes that fallback fast. Byte-parity with the offline loop is the gate.
- **Parallelization's role changes online**: steady-state arrival is ~1
  window + ≤3 embedding calls per second — sequential keeps up even on weak
  chips. The pool earns its place in catch-up (late start, load, lid-close,
  crash recovery) and in the forever-offline paths. Online is a scheduling
  layer over the same code, not a second algorithm.

Costs, to be weighed with numbers next session:

1. **Power during the meeting.** Moves ~112 s of compute into the live
   window — plausibly a few tenths of a watt sustained, same order as the
   entire live budget (~0.6 W) that was fought down from 6.1 W with
   efficiency explicitly prioritized over latency. Needs powermetrics, and
   possibly a plugged-in-only policy question.
2. **A second computation path.** `steno transcribe` keeps the offline loop
   forever, so the online worker duplicates the loop's chunking semantics
   in streaming form — every future loop change then has two homes. The
   one-path principle doesn't strictly forbid it, but it's a real
   maintenance cost.
3. **Benefit is hardware-dependent.** On this Mac after lever 1: saves
   ~30–45 s out of a wait that stays minutes long because of notes. On
   mobile Intel: saves 6–12 min (2–4 min after lever 1) — the strong case.
   The trigger candidate: *the post-lever-1 finalize number on weak
   hardware still bites.*

## Missing measurements (do these before or at the scoping session)

1. **Real-meeting finalize split**: ASR (should be ≈0 — live decode reuse),
   diarization, re-ID naming, notes-LLM, on a real meeting from
   `~/Documents/Meetings` (or the `verify` skill's synthetic session).
   The online decision is made against this split, not against the
   diarization number alone.
2. **Finalize watts** during the current sequential run vs the lever-1
   parallel run (race-to-idle check).
3. **An Intel datapoint.** Everything above marked for Intel is *inferred*
   (mobile 3–6× slower single-core ⇒ RTF ~0.15–0.3 ⇒ 6–12 min per 40-min
   meeting today). Directly measurable on the CachyOS notebook or the GPD;
   worth one run if weak-hardware numbers will carry the online decision.

## Suggested sequencing for the plan (to be debated)

1. Lever 1 with its parity gate and thread-safety measurements.
2. Missing-measurement runs (finalize split, watts, one Intel datapoint).
3. Decide online precompute against those numbers; if pursued, it is a
   scheduling layer over lever 1's code with byte-parity against the
   offline loop as the gate; if declined, record the trigger that reopens
   it (weak-hardware finalize pain).
