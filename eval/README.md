# Phase 0 — model evaluation harness

**Result (2026-07-06): Parakeet-TDT-0.6B-v3 is the default for finalize + live.**
Blind adjudication of 161 disagreement sites: Parakeet tied Whisper 42:42,
Voxtral lost to both, Canary lost every pairing ~1:2.
raw judgments in out/adjudication-results-2026-07-06.json (gitignored).

Goal: decide the default finalize-pass ASR model on *real* meeting audio, not
read-speech leaderboards. Candidates:

| Model | Runtime | Role | Status |
|---|---|---|---|
| Parakeet-TDT-0.6B-v3 | parakeet-mlx | **default candidate** (finalize *and* live) | wired up |
| Voxtral Small 24B (4-bit) | mlx-voxtral | max-accuracy challenger | wired up (text only — no timestamps), ~14 GB download |
| Whisper large-v3 | mlx-whisper | mature fallback | wired up |
| Canary-1B-v2 | NeMo on MPS | accuracy-ceiling reference only | wired up, needs `uv sync --group eval-canary` (slow; never shippable) |

Canary was demoted from expected default (July 2026 research): no MLX/CoreML
runtime emits its word timestamps — PyPI `canary-mlx` is an abandoned template,
mlx-audio's Canary returns hardcoded 0.0 timestamps, and onnx-asr's timestamp
support covers TDT/CTC/RNNT only. NeMo-on-MPS is the sole real path and is too
slow/heavy to ship, so it serves purely as the accuracy ceiling in this eval.

## Workflow

Everything under `eval/audio/`, `eval/out/`, and `eval/refs/` contains private
meeting content and is gitignored.

```sh
# 0. See what language/content each recording holds (Whisper-tiny probes)
uv run --group eval eval/scan_languages.py

# 1. Define eval segments in manifest.json (id, source file, start/end seconds,
#    language), then cut them to mono 16 kHz WAV:
uv run --group eval eval/extract.py
#    To pick bounds by listening, first extract a full recording:
uv run --group eval eval/extract.py --full "my-meeting-recording.mov"

# 2. Judge the models. Primary path — blind disagreement adjudication:
#    align all hypotheses, review only the spots where models disagree
#    (audio snippet + shuffled unlabeled variants, ~4s each):
uv run --group eval eval/adjudicate.py       # → eval/out/adjudication.html
#    Open in a browser, judge (keys 1–9, 0 = unsure), download results, then:
uv run --group eval eval/adjudicate.py --score ~/Downloads/adjudication-results.json
#
#    Secondary path — full hand-corrected references (slow, and beware anchor
#    bias: a reference corrected from model X's draft flatters model X; the
#    de-1 attempt measured Whisper at 1.3% WER for exactly that reason).
#    Fix eval/refs/<id>.draft.txt while listening, rename to <id>.txt;
#    only *.txt files are scored by score.py.

# 3. Run every candidate (one process per backend, for clean peak-memory numbers):
uv run --group eval eval/transcribe.py --backend parakeet
uv run --group eval eval/transcribe.py --backend whisper
uv run --group eval eval/transcribe.py --backend voxtral

# 4. Score:
uv run --group eval eval/score.py     # → eval/out/report.md
```

Target coverage: ~10 min hand-corrected reference per language (German +
English), including one in-room far-field sample.

## Short-utterance / window-length study (2026-07-19)

Tests whether the product's windowing hurts short, isolated utterances: the
finalize pass decodes VAD-packed windows and reuses them verbatim, so an
utterance with >5 s silence around it becomes its own tiny window with no
acoustic context. `windows.py` decodes eval segments through the *real*
pipeline path (VAD → `pack_windows` → Parakeet per window, byte-identical
slices) and records which window produced which words; `adjudicate.py` joins
disagreement sites (vs the Whisper pivot) against those spans, reports
disagreement density per window-length bucket, and carries the tags into
blind judging. Full recorded meetings (`--record-audio` WAVs, one manifest
entry per channel via the `channel` field) supply the short-window data the
300 s Phase 0 segments barely contain.

```sh
uv run --group eval eval/extract.py       # includes the meeting-channel cuts
uv run --group eval eval/windows.py       # → out/parakeet-win/<id>.json (+ window spans)
uv run --group eval eval/transcribe.py --backend whisper --segments <new ids>
uv run --group eval eval/adjudicate.py --backends parakeet-win,whisper --max-sites 20
#   → out/window-report.md (label-free density table) + adjudication.html
uv run --group eval eval/adjudicate.py --score ~/Downloads/adjudication-results.json
#   → adds a per-bucket correctness table for parakeet-win
```

**Label-free result (2026-07-19, 5.2 h of channel audio + Phase 0 segments):**
disagreement density falls monotonically with window length — **9.2 sites/100
words in <3 s windows, 5.1 in 3–8 s, 3.9 in ≥8 s** — and sites are ~3×
over-represented in the first 0.5 s of a window. Mic channels are the
short-window regime (de-0717s-mic: median window 1.8 s; system channels
~25 s).

**Causal confirmation without human judging (`context_ab.py`, 2026-07-19).**
Human adjudication was replaced by two automated experiments (model juries
were rejected: the Phase 0 jurors are ~coin-flips on contested sites, and
juror difficulty correlates with exactly the short-utterance audio under
test). Results in `out/context-ab.md`:

1. **Context A/B** — every <8 s window re-decoded by the *same* Parakeet with
   left context added, Whisper refereeing only the changed spans (same model +
   same audio ⇒ any change is windowing-caused; same-model variants ⇒ referee
   style-anchor bias cancels). With contiguous preceding audio ("raw", what a
   fix would decode): pivot sides with added context **23:9 in <3 s windows,
   29:15 in 3–8 s, and 26:29 (null, as predicted) in the ≥8 s control**.
   Splicing the previous window across the gap is weaker (15:6 / 21:17) — the
   seam costs accuracy, so the fix should extend windows into *contiguous*
   preceding audio (silence included) and drop the pre-span words, NOT splice
   distant speech. That changes only the decode step (pipeline `_decode` +
   `WindowedLiveDecoder._decode_window`, in lockstep for the reuse guarantee)
   — window spans and pack_windows stay untouched.
2. **VAD-drop check** — the 74 pivot-only spans decoded by Parakeet with no
   VAD gate: **20 confirmed drops of real speech (48.5 s lost)**, 18 Whisper
   hallucinations, 36 disputed — but the disputed examples are mostly real
   interjections both models hear ("ja", "okay", "gut") differing only in
   wording, so ~3/4 of the spans are genuinely lost speech. Only 1 drop is
   under 0.3 s: the `min_speech=0.25` floor is NOT the main culprit — the
   detector misses quiet short interjections at threshold level (mic channel
   almost exclusively; these are the user's own "ja/genau/okay" turns).

**Shipped fix (2026-07-19): context-carry decode, and nothing else.**
Windows under 8 s decode with up to 15 s of contiguous left context
(`vad.context_start`, mirrored by `pipeline._decode` and
`WindowedLiveDecoder._decode_window`; window bounds untouched).
Verification: byte-identity + full suite green, `live.py` at
**0.0 % WER** vs finalize (reuse guarantee intact), `context_ab.py` re-run
collapses the raw-context arm to **2–4 % changed with a null referee** on
the short buckets (the product now *is* the context decode; the splice arm
*loses* to it — contiguous-beats-splice reconfirmed), and a batch
re-transcription of an 87-min meeting lands within **−17 words of the
pre-fix decode** (pure churn) with the refereed changed regions favoring
the new text. One meta-lesson is recorded in `windows.py`: its decode loop
once re-implemented the pipeline's slice arithmetic and silently went stale
when the fix landed — it now imports `context_start`/`_clip_context` from
the package, so the eval measures the real decode path by construction.

**Two VAD edge retunes: tried and REVERTED (same day), each caught by the
systematic re-transcription comparison against the stored meeting
transcripts.** (1) threshold 0.5 → 0.4 recovers quiet interjections (drop
check: 56 likely-real lost spans → 39) but on busy channels removes the
`min_silence` closes — runs slam into `max_speech`, windows pack
wall-to-wall and cut mid-speech at the 30 s budget, losing Whisper-verified
sentences. (2) pad 0.15 → 0.3 s looked harmless but bisected to **−302
words on one meeting** (context-carry held constant; at pad 0.15 the same
code is −17). Root cause, directly observed: the greedy TDT decode is
**knife-edge unstable at a full window's tail** — the same ~30 s span
decodes completely or drops ~10 trailing words on a few-millisecond bound
shift — so ANY change to window bounds re-rolls every tail, and bounds that
end nearer to speech systematically lose. Standing rule: fix onset clipping
and interjection drops with decode-side changes, never by moving window
bounds. The tail instability predates this study and explains occasional
hard-split boundary losses in every earlier transcript.

**Shipped fix (2026-07-26): the pre-roll claim — the onset clipping, fixed
without touching a bound.** The onset words the study named lost ("Eine
Frage habe ich noch" → "Habe ich noch", leading "Ja,") were *decoded* all
along on short windows — the context-carry slice contains them — and then
discarded, because a word belonged to the window containing its midpoint and
theirs fell before the padded start. `vad.claim_start` moves that keep-rule
to `max(start − 0.3 s, previous window end)`: the window owns 0.3 s of the
silence ahead of it (0.45 s behind the *reported* onset once the 0.15 s pad
is counted — Handy's `VAD_PREFILL_FRAMES=15`), floored so no word is ever
claimed twice. Bounds, packing and every decode slice are byte-identical to
before; only ownership changes.

Measured by re-transcribing the six study meetings on **both code arms**
(`retranscribe_compare.py --old-dir`, added for exactly this control):
**+147 words / 61 k, 133 added regions (131 ≤3 words), 1 removed, 2
changed** — the first change in this study that is purely additive, as it
must be when no decode moves. A VAD-window probe over the same audio pins
the mechanism: 1426 of 1450 windows have a non-empty pre-roll, 442 words are
claimed from one, and **227 are text the old arm had nowhere within ±1 s** —
sentence heads, "Ja," / "Genau," / "Okay," / "I think not." / "Timo, kannst
du mir sagen". `live.py` stays at **0.0 % WER** (reuse
guarantee intact).

**Long windows cannot claim, and buying them the ability was rejected.** A
window ≥8 s decodes exactly its span, so nothing precedes its start to claim.
Giving them a 2 s pre-roll *read* was tried in the same session: it recovered
**89 more** heads (227 → 138 short-window-only) and re-rolled the greedy
decode of every long window, taking the corpus to **−59 words with 324
removed / 495 changed regions** and a referee that could not tell the arms
apart (71:58). The same lesson as the two retunes, one level down: it is not
only *bounds* that must not move — anything that re-rolls a long window's
decode pays coin-flip churn on ~1000 windows for a handful of words.

**Cut-overlap decoding: shipped and REVERTED 2026-07-19** (design record +
revert rationale in git history). The decode-side
cut repair (edge classification, 2.5 s overhang, midpoint keep-rules,
speech-coverage retry) measured slightly positive in aggregate (+24
non-filler words / 61 k arm-to-arm, net −235 vs −272) but per-site ear
adjudication showed a coin flip (6 gains : 5 losses, referee 61:61) and a
new language-flip failure mode invisible to the coverage chooser — healthy
German windows re-rolling into English. Reverted whole for simplicity;
`windows.py` and `context_ab.py` (experiment 3, the jitter probe) went
back with it to the pre-change versions matching the restored pipeline.

Measures the *diarizer*, not the ASR: **DER** (Diarization Error Rate) and
**word attribution** (of the finalized words, the fraction placed on the right
speaker). Nothing speaker-centric — re-ID threshold tuning, clustering/embedding
upgrades — is measurable without this, so it is the gating prerequisite.

References are **hand-labelled** speaker turns in NIST RTTM. The audio is a mono
downmix of every speaker (`extract.py`'s `-ac 1`), so one RTTM per segment labels
all distinct voices in it.

```sh
# 1. Seed a draft from the current diarizer to correct while listening (much
#    faster than labelling boundaries from scratch). Pass the real speaker count
#    you remember — unconstrained estimation over-clusters badly (de-1 → 13
#    speakers), which is exactly the problem this measures.
uv run eval/diarize.py --bootstrap --segments de-1,de-2,en-1 --num-speakers 3
#    → eval/refs/<id>.draft.rttm  (each line: SPEAKER <id> 1 <onset> <dur> <NA> <NA> <spk> <NA> <NA>)

# 2. Fix boundaries + merge/rename speakers against the audio, then rename to
#    the scored name (only <id>.rttm is scored; never score a .draft — it
#    flatters the model that produced it):
mv eval/refs/de-1.draft.rttm eval/refs/de-1.rttm   # after correcting

# 3. Produce hypotheses (raw diarizer turns + finalized word labels):
uv run eval/diarize.py --segments de-1,de-2,en-1   # → eval/out/diar/<id>.{rttm,words.json}

# 4. Score (DER + word attribution, optimal speaker mapping, 0.25 s collar):
uv run eval/der.py                                 # → eval/out/diar-report.md
```

`der.py`/`rttm.py` are pure (numpy + scipy) and unit-tested in
`tests/test_eval_der.py` against hand-computed cases; `diarize.py` drives the
real stenograf backends. Everything under `eval/refs/` and `eval/out/` stays
gitignored (private content).

### The corpus harness (AMI/ICSI, no hand labels)

> **The owned loop, the sherpa diarizer, stenodiar and the fold/collapse rules
> were deleted on 2026-11-15** (Nemotron 3 is the one diarizer). The numbers
> below stand as measured; the scripts that measured only those pieces —
> `loop_arm`, `loop_freeze`, `loop_parity`, `fold_sweep`, `fold_gate_audit`,
> `linkage_sweep`, `pool_parity`, `embed_units`, `kplus1`, `cluster_ab`,
> `collapse_probe`, `nemotron_collapse`, `diarizen_arm` — live in git history
> (`git log --diff-filter=D -- eval/<name>.py`). `diarize.py`, `solo_arms.py`
> and `split_recovery.py` now run Nemotron, so their published numbers
> describe the diarizers they ran then, not a rerun today.

The `PLAN-DIARIZATION.md` step-0 harness sidesteps hand-labelling: `ami.py`
downloads per-speaker headset audio for five recurring groups — AMI ES2003,
ES2007, IS1009, TS3010 (all three recording sites; ES2007/TS3010 joined
2026-08-07 after a vetting pass that also *measured* clipping the corpus
data-problems table doesn't list — `AMI_GROUPS` docstring) and the ICSI Bmr
series as a fifth group with real attendance churn (Bmr021/024/025/030 mapped
onto session letters; `ICSI_SESSIONS` docstring) — and synthesizes
stenograf's topology per meeting: one participant's headset is the **mic**
channel, the other N−1 mixed are the **loopback**, references from the human
annotations under corpus-global speaker names (`eval/refs/ami/`, generated,
still gitignored). The whole matrix (40 channels) is one command:

```sh
uv run --group eval eval/ami.py run   # fetch → diarize → DER + naming reports
```

which chains `diarize.py --ami` (known per-channel counts; also writes each
cluster's embedding), `der.py` (now also covering `refs/ami/`), the re-ID
trial builder (`ami.py trials`: enroll each group's session *a*,
alphabetically-last participant left out as the stranger; Bmr adds natural
strangers — attendees absent from Bmr021), and `reid_score.py` — DIR @ FAR
with the FAR/FRR curve, unit-tested in `tests/test_eval_reid.py`;
parsing/mixing math in `tests/test_eval_ami.py`. Corpus facts (URL patterns,
identity mapping traps, format gotchas) are in `ami.py`'s docstrings;
calibration caveats in `eval/diarization-sota-2026.md`'s scoring-family
table (published AMI numbers are family A, 0 s collar; ours are family B,
0.25 s — compare our numbers only to our numbers).

**Trial convention (2026-08-07): the headline pool is same-group only.** A
cluster scores against its *own* group's gallery (`out/reid/trials.json`) —
an unenrolled voice in your own meeting, the product's hard case. Scoring
every cluster against every foreign gallery manufactures easy negatives whose
count scales with the group count and dilutes the FAR denominator (measured
at five galleries: 280 easy vs 20 hard negatives pooled — FAR ≤ 5 % would
tolerate 15 hard accepts where the 2-gallery harness tolerated 2, silently
lowering any threshold picked from the curve). Cross-group scores live on as
a separate big-store diagnostic (`trials-crossgroup.json`, reported at the
headline thresholds). Numbers recorded before this date pooled both
populations and are not comparable to their successors.

**Grown-harness baseline (2026-08-07, 41.8 min matrix, gate ≤ 1 h):** 70
same-group trials (53 known, 17 strangers — FAR granularity 5.9 %). DIR
**88.7 % @ FAR 0 %** (threshold 0.605); DIR flat at 88.7 % from threshold
0.322 up while FAR falls 41.2 → 0 %; the known-trial ceiling is 92.5 % (4/53
top-scored by a wrong profile at any threshold — clustering confusion, not
threshold territory). At the shipped 0.5 default, hard-stranger FAR is
**11.8 %** — the first direct evidence the default is too permissive (step
2.4's input). Cross-group diagnostic: highest foreign score 0.531, FAR 0 % at
every headline threshold. DER/word-attribution: mic channels 100 %
attribution (20/20); new-group loops in the established character (ES2007
94.0–96.9 %, TS3010 68.9–98.8 % with one short confusion-heavy session —
target material like IS1009); ES2003 b/c reproduce the 2026-08-06 baseline
(4.9/5.8 % DER vs 4.8/5.8 %).

#### Why a single-speaker channel never sees the diarizer (2026-08-06)

`finalize_channel` labels everything `S0` when the channel's speaker count is
1. `solo_arms.py` scores that against the two arms it replaces, on all ten
single-speaker corpus channels (`out/diar-solo-arms.md`):

| arm | DER | word attribution | clusters |
|---|---|---|---|
| shipped (no diarizer) | 24.8 % | **100 %** | 1 |
| diarizer forced to one cluster | 11.4 % | **100 %** | 1 |
| diarizer estimating the count | 30.5 % | 77.7 % | 2–3 |

Two things follow, and only one of them is the obvious one.

**Running the diarizer at k=1 halves DER and changes nothing the user reads.**
Word attribution is 100 % in both arms on all ten channels, because with one
speaker every attribution is trivially right. The 13-point DER gap is speech
the segmentation model marks and the transcript never contains — backchannels
and soft tails the ASR path does not decode — so it buys a better activity
score for ~105 s of diarization per channel and no better transcript. Declined
on those numbers, not skipped.

**Estimating the count on a solo channel splits it, every time.** All ten
channels come back as 2–3 speakers and word attribution falls to 77.7 % mean
(45.6 % worst) — one person's monologue chopped across "Speaker 1/2/3". That
is the cost of *not* stating a count, it is the far-field over-splitting
complaint measured on the local channel, and it is what the shipped recovery
pair — merge-at-naming for profiled speakers, `collapse_single_voice` for
everyone else — closes (the split-recovery section below). The arms here
measure the raw estimator; the pipeline no longer passes its splits through.

#### Split recovery: merge-at-naming, the collapse rule, and k+1-fold (2026-08-06)

Three measurements behind the shipped over-split recovery
(`PLAN-DIARIZATION.md` step 1.3), all on estimate-mode or k+1 diarization of
the corpus channels with word times reused from the matrix:

**`split_recovery.py`** — recovery arms + the discriminator. Estimating splits
every solo channel 2–3 ways; the arms measure what gets it back. One-to-one
resolve recovers *nothing* (by construction: one cluster keeps the profile,
the rest stay split) and once *forced a wrong name* — the over-split cluster's
true profile was claimed, so greedy exclusivity handed it the next-best wrong
one. Many-to-one resolve (merge-at-naming, now shipped) recovers all six
profiled solo channels to 100 % word attribution and never regresses a loop
(+0–2.1 pts there). The discriminator stats decide the unprofiled case: min
pairwise cluster-embedding similarity is 0.74–0.98 on split solo channels vs
≤ 0.18 on the 3–7-speaker loops, while dominance share overlaps (46 % on both
sides) and unrestricted pairwise self-merge is catastrophic on loops
(cross-speaker cluster means reach 0.95 cosine; IS1009b.loop chain-merged
7 → 1, 93.2 → 44.3 %). Only the all-pairs-similar → collapse-to-one form
survives.

**`collapse_probe.py`** — the falsification test for that collapse on the case
the corpus lacks: genuinely-2-speaker channels (production's usual remote
channel). All loop-participant pairs per AMI meeting, masked and mixed exactly
like the loop channels (24 channels): min sim 0.00–0.45, **0/24 falsely
collapse**. The empty band [0.45, 0.74] puts `pipeline.COLLAPSE_SIMILARITY`
at 0.6, mid-gap.

**`kplus1.py`** — known counts diarized one over the stated k. Raw k+1 is a
wash on 8/10 loops but +20.8 pts on IS1009c (exact-k clustering had fused two
voices) and +2.6 on Bmr025, with losses bounded at −0.4 — the measured shape
of "splits are recoverable, merges are not". Folding the most-similar cluster
pair back (`pipeline.fold_excess_clusters`, now the shipped path) keeps the
win: mean 87.8 → 90.1 %, worst −0.3, output always exactly the stated count
(no phantom speaker), profiles not required. The fold did *not* re-fuse
IS1009c's separated voices (its max-sim pair was elsewhere, 0.45). Recovery
via naming alone is declined as a shipped form: it needs profiles and folded
wrongly once (IS1009d, −1.3).

The re-baselined matrix (same day, folded pipeline end to end) reproduces the
fold arm exactly on all ten loop channels — IS1009c's confusion drops
24.9 → 6.2 % DER — and the re-ID trials, now built from folded clusters, hold
the same operating points (DIR 88.2 % @ FAR 0, 94.1 % @ 3.4 %); only the
FAR-0 *threshold* moved (0.609 → 0.809), which belongs to step 2.4's sweep.

#### Overlap-clean embeddings, and the fold criterion they exposed (2026-08-06)

`cluster_embeddings()` now excludes spans where another cluster is active
before embedding (a fully-overlapped cluster falls back to its raw turns —
absent embeddings would block naming, collapse and fold). The literature
motive: overlap in the embedding measurably hurts (12.84 → 14.11 % EER), and
frame-level identity inside overlap is near-chance.

**The first matrix gate failed**, and the diagnosis changed the fold. Loop
attribution fell 90.1 → 83.1 % and DIR@FAR0 88.2 → 66.7 %, all of it in
channels where the k+1 fold had fused two large real clusters. On the cached
k+1 turns the max-similarity pair had *never* been the semantically right
pair — under overlap-included embeddings it merely happened to involve the
tiny spare cluster, whose similarity was inflated by audio it shared with a
real cluster, so the wrong fold cost single-digit seconds. Cleaning removed
exactly that inflation, and the max pair became two large real speakers
(ES2003b: 182 s + 485 s fused). The four-arm comparison on all ten loops:

| fold criterion | raw embeddings | clean embeddings |
|---|---|---|
| most-similar pair | 90.1 % | 83.1 % |
| smallest → most-similar partner | 90.1 % | **90.1 %** |

So the shipped fold picks the *spare by duration* and only its partner by
similarity — a wrong partner can never cost more than the spare's own speech
— and that criterion is embedding-insensitive where max-pair was luck.
IS1009c's +20.8 survives under it. The collapse discriminator re-measured
*cleaner* with overlap-clean embeddings: split solos 0.73–0.98, synthesized
2-speaker channels ≤ 0.39 (still 0/24 false collapses), loops ≤ 0.16;
`COLLAPSE_SIMILARITY` stays 0.6, now nearer the solo edge so the
unrecoverable direction (two real speakers merged) keeps the bigger margin.

The second matrix gate passed: loop attribution back at 90.1 % mean
(channel-identical to the offline clean/smallest arm), and the re-ID curve at
the strict end is unchanged-to-better — DIR 88.2 % @ FAR 0 at threshold 0.605
(baseline 0.609), FRR there 11.8 → 5.9 %. The loose-end ceiling dropped
94.1 → 88.2 %: exactly one of 17 known trials, single-trial granularity. Both
wrong-name trials sit on clusters from the two most confusion-heavy loops
(ES2003d, IS1009b) — clusters the diarizer already mixed, a clustering
problem (steps 4–5), not an embedding one. Re-judge when the trial set grows
(the step-2.4 precondition).

#### A boundary margin at word intersection is a no-op here (2026-08-06)

The literature's ~0.1 s turn padding (TST-Bench +0.26 DIR) presumes an
intersection that drops or misplaces words outside every turn. Ours cannot:
`merge_words_turns` snaps an uncovered word to the *nearest* turn, which
already splits every inter-turn gap at its center, and a symmetric pad clamped
at neighbors provably cannot move a center split — the only leak is
largest-overlap decisions where turns genuinely overlap. `margin_sweep.py`
(cached matrix turns + word times, real merge, margins 0.05–0.25 s, ten loop
channels) confirms it: mean attribution 87.77 % at margin 0 and 87.74–87.77 %
everywhere else; at the widest margin only 172 of 32 879 words move at all,
best single channel +0.27 pts, worst −0.33 pts, no margin consistently signed.
Padding the *output* turns is no better — DER of padded turns worsens on 8 of
10 channels (IS1009a +10.5 pts at 0.25 s; best improvement −0.9 pts,
Bmr021) — so the pad earns no place anywhere in the pipeline. Declined in
`PLAN-DIARIZATION.md` step 1.2 on these numbers. Re-open trigger: an
intersection rule that no longer assigns every word — the no-op proof rests
on nearest-turn snapping.

#### A minimum-duration gate on naming has nothing to gate (2026-08-06)

The literature's short-turn cliff (EER at 2 s ≈ 2.4× full-duration;
ERes2Net-base 3.28 % @ 2 s) motivated barring clusters with < ~3 s of clean
speech from profile matching. `naming_gate.py` measured the proposal three
ways, and the cliff is real but lands entirely on the *miss* axis:

- **Truncation sweep** (every matrix cluster's clean audio cut to 1–8 s,
  production embedding, session-a galleries): known speakers' top-correct
  score falls 0.90 (full) → 0.63 (3 s) → 0.40 (1 s), so DIR@FAR0 drops
  88.2 → 41.2 %. But strangers' top scores *barely move* — max 0.37 at 2 s,
  0.34 at 3 s across 59 stranger trials per arm — so at the shipped 0.5
  threshold FAR is 0.0 % in every truncated arm. A short cluster fails to be
  named; it does not get falsely named. The one false accept in the whole
  table is a *full-duration* cluster (0.594).
- **Shipped known-count path**: the k+1-fold leaves no small clusters (min
  clean duration 3.2 s over all matrix clusters); a 3 s gate gates zero
  trials, and 5 s moves no operating point.
- **Estimate mode** (collapsed clusters, threshold 0.5): 22 clusters get
  named; the only one under 5 s clean is named *correctly* (3.4 s, 0.669),
  while both wrong namings sit at 9.5 s and 75.1 s — clustering confusion the
  steps-4/5 work owns, unreachable by any duration cutoff. A 5 s gate:
  0 wrong namings prevented, 1 correct naming lost (and its merge-at-naming
  recovery with it).

Declined in `PLAN-DIARIZATION.md` step 1.5 on these numbers: the threshold
already does the gate's only useful job. The result is threshold-dependent —
re-measure if step 2.4's sweep lands the operating point below ~0.4, where
the short-stranger maxima live.

#### Multi-meeting profiles: score averaging measured before shipping (2026-08-06)

`store_v2.py` gates the profile store's move from one running mean to a
per-meeting embedding set matched by mean cosine (`PLAN-DIARIZATION.md`
step 2.1). Enrollment from AMI sessions a **and** b, trials from the cached
matrix clusters of c–d + ICSI (60 trials, 11 known — one known trial = 9.1
pts, so only large effects can register). Five arms on identical trials:
single-a, single-b, v1's running mean `l2(a+b)`, mean cosine (v2), max cosine
(control).

At full trial duration every arm is identical (DIR 90.9 % at FAR 0 *and*
FAR ≤ 5 %; the same one known trial fails everywhere). The arms separate only
where matching is hard — trials truncated to 2 s of clean audio: **single-a
collapses to 81.8 % DIR@FAR0 while both combined-mean forms hold 90.9 %**, and
max-cosine collapses with it (81.8 % — max rides the single worst gallery
member). So multi-meeting enrollment is measured insurance against one
enrollment being wrong for a hard cluster, the mean beats the max, and score
vs embedding averaging stays within single-trial noise here (score-avg's 3 s
EER point is one trial better: FAR 4.1 %/FRR 0 vs 6.1 %/9.1 %). Score
averaging ships on the research record's at-scale numbers (2.05 % vs 2.85 %
EER, `PLAN-DIARIZATION.md` step 2.1) plus measured never-worse; single-entry
profiles score identically under both, so the shipped matrix operating points
are unchanged by construction.

#### Rename-once enrollment: the meeting's own cluster, measured (2026-08-06)

`rename_once.py` gates `PLAN-DIARIZATION.md` step 2.2 — enrolling a speaker
from the diarized cluster the user just corrected, instead of demanding a
clean sample. Arms differ only in what session *a* enrolls (trials identical:
all 76 matrix clusters of sessions b–d + ICSI, full-duration plus 3 s/2 s
truncation): `headset` = reference spans on the raw headset (the trial
convention), `clusters` = the session-a matrix cluster embeddings mapped to
speakers by reference majority (exactly what `steno profiles assign` stores),
`both` = one profile holding both (enroll-then-assign), and `headset ∩` =
headset restricted to cluster-covered names — the control that separates
enrollment *source* from enrollment *coverage*.

At equal coverage, cluster enrollment matches clean-headset enrollment at
every practical operating point (identical EER 92.9 %/6.5 %/7.1 % at full
duration, DIR@FAR≤5 % equal at full and 3 s) and **beats it on 2 s trials**
(85.7 % vs 78.6 % DIR@FAR≤5, EER likewise) — the research record's
channel-match claim (+18 % rel) reproduces in sign on our stack. Reinforcing
an existing profile (`both`) is never worse than the clean enrollment alone
and better at 2 s (DIR@FAR0 82.4 % vs 76.5 %). Two costs, both measured and
both owned by the *enrollment meeting's diarization*, not by the flow:

- **An impure cluster enrolls an impure profile.** IS1009a's known-count
  clustering fused quiet FIO084 into FIO087's cluster; the resulting profile
  leaks 0.789 cosine to FIO084's clean headset (every clean profile's
  off-diagonal is ≤ 0.344) and pulls FIO084's later clusters to a false
  "FIO087" at up to 0.860 — which is what pushes the cluster arm's strict
  FAR-0 point down (78.6 % vs 92.9 % for `headset ∩`; FAR ≤ 5 % and EER are
  unaffected). No threshold fixes an impure enrollment; the confusion is
  IS1009's known clustering problem (steps 4–5), and the leak table in the
  report is the probe to re-run after those steps.
- **A fused-away speaker is unassignable from that meeting** (FIO084 has no
  majority cluster on IS1009a — 1 of 6 enrollable names). The coverage hole,
  not the embedding source, is most of the naive headset-vs-clusters gap:
  the same hole cut into the headset arm moves its top stranger score
  0.597 → 0.692.

Step 2.2 ships on these numbers: enroll-or-reinforce from the corrected
cluster, with the impure-enrollment caveat recorded here rather than gated —
the failing case is exactly the meeting whose transcript was already wrong.

#### Automatic profile updates: measured, declined (2026-08-07)

`auto_update.py` closes `PLAN-DIARIZATION.md` step 2.3 — may an auto-matched
cluster reinforce its matched profile without a user correction? The deployed
loop is replayed on the harness: rename-once enrollment from session a,
sessions b+c resolved at the shipped 0.5 threshold with each policy deciding
which named clusters append their embedding, then naming evaluated on
held-out session d + ICSI (44 trials, 5 known — single-trial granularity is
20 pts, so only large effects can register). Policies: no-update baseline,
ungated, score margin (.10/.20 above threshold), margin + 5 s minimum clean
duration, and an oracle that appends only correctly-named clusters — the
upper bound of any gate.

- **The benefit is invisible where it matters.** Even the oracle ties the
  baseline at every held-out full-duration and 3 s operating point (DIR@FAR0
  80.0 % in all six arms). The sole separation in the whole table is one
  2 s-truncated trial at the shipped threshold (20 % vs 0 % DIR for
  margin .20/oracle) — an artificial regime, and one trial.
- **The poison passes every implementable gate.** All three wrong updates are
  IS1009's FIO084 — the fused-away, unenrolled quiet speaker — absorbed into
  FIE088/FIO087 profiles at scores 0.609–0.860, top-minus-second gaps up to
  0.855 (*higher* than most correct updates' gaps), and 40–721 s of clean
  speech (the wrong clusters are among the longest, as `naming_gate.py`
  already measured for wrong namings). Absolute score, relative margin, and
  duration cutoffs all admit them; only the oracle doesn't. The attractor is
  the impure FIO087 enrollment (`rename_once.py`'s known problem) —
  diarization confusion, which no store-side statistic can see.
- **Absorption blurs rather than breaks at this scale.** The one baseline
  stranger accept (FIO084's session-d cluster → FIO087, 0.622) *drops* to
  0.531 under updates — mean-cosine dilution — but known-trial thresholds
  fall in parallel and the EER point's FAR doubles (5.1 → 10.3 %). Score
  distributions converge; no operating point improves.

Declined in `PLAN-DIARIZATION.md` step 2.3 on these numbers: profile growth
stays user-confirmed (rename-once assign, measured never-worse). Re-open
triggers: steps 4–5 fix the IS1009-class clustering confusion — every wrong
update is downstream of it, so re-run this harness then — or a grown trial
set that can resolve sub-20-pt benefits.

#### The operating threshold: 0.5 → 0.62, picked from measured curves (2026-08-07)

`threshold_pick.py` closes `PLAN-DIARIZATION.md` step 2.4 on the grown
same-group harness, sweeping candidates over *both* enrollment arms (clean
headset = the matrix trials; session-a cluster enrollment = what `steno
profiles assign` stores) at full duration and 3 s / 2 s truncation, plus the
solo-channel margins. What decides it:

- **At 0.5 the store names strangers**: 11.8 % hard-stranger FAR on the
  headset arm, 25 % on the cluster arm. DIR is flat from 0.322 to the FAR0
  point in both arms, so the FAR was pure loss.
- **0.62 rejects every non-pathological stranger in every arm and regime**
  (clean strangers top out at 0.597, cluster-arm clean strangers at 0.505;
  the 2 s cluster-arm FAR0-strict point is exactly 0.620) at zero
  full-duration DIR cost (headset 88.7 %, cluster 90.0 %, same as 0.5) and
  with solo/1:1 naming untouched (known solos score 0.870–0.970). It keeps a
  0.023 margin over the nearest clean stranger where 0.60 kept 0.003.
- **What survives above 0.62 is not threshold territory**: the IS1009
  impure-enrollment leak (a fused enrollment cluster attracting its absorbed
  speaker at 0.860/0.681/0.622). Strict FAR0 on the cluster arm would need
  0.866 and cost 26 pts DIR. That residue is clustering confusion (steps
  4–5); re-run this script after them.
- **The measured cost sits in truncated regimes** (cluster arm 2 s: DIR
  42 → 30 %) — artificial for the shipped k+1-fold path, which leaves no
  sub-3.2 s clusters; real only for estimate-mode short clusters, where the
  one recorded sub-5 s correct naming (0.669) still clears 0.62.

Shipped as `DEFAULT_THRESHOLD = 0.62` (`voiceprints.py`, curve numbers in its
docstring). Step 1.5's re-open trigger (an operating point below ~0.4) did
not fire; AS-Norm/QMF stay un-adopted — the residual FAR is a purity problem,
not a calibration problem.

#### Owning the diarization loop: parity measured (2026-08-07)

`loop_parity.py` gates `PLAN-DIARIZATION.md` step 4 Phase A: the owned loop
(`stenograf/diarization/loop.py` — segmentation-3.0 via onnxruntime, powerset
decode, per-(chunk,speaker) embeddings, complete-linkage AHC, all built from
the line-cited reference spec `eval/diarization-loop-spec.md`) against
sherpa's `OfflineSpeakerDiarization`, raw `diarize(k+1)` on all 20 loop
channels. Mean ΔDER **−1.41 %** (own slightly ahead), 14/20 within ±0.6 pts,
runtime equal (see `out/diar-loop-parity.md` for the table). Six channels
diverge in both directions (own −23.6 TS3010a, −9.4 ES2003d, −5.7 Bmr030 vs
+8.2 TS3010d, +4.5 IS1009d, +1.5 ES2007c): AHC merge-decision fragility on
confusion-heavy channels — our embedding streams concatenate audio where
sherpa concatenates feature frames, and the tiny vector deltas flip discrete
merges. Bit-parity was never the gate (the spec records where the reference
is internally inconsistent, §1.5); score parity holds.

#### Embedding-model A/B: both candidates lose both seats (2026-08-07)

`embedder_ab.py` closes `PLAN-DIARIZATION.md` step 3. The embedding model
sits in two seats — clustering inside sherpa's diarization and re-ID scoring
— so each candidate re-ran the full 40-channel matrix (own out dir, child
process per loop channel) and was scored in both: DER/word attribution
against the refs, then same-group naming trials re-enrolled with the
candidate, full + 3 s/2 s truncation, compared at FAR-anchored operating
points (raw thresholds are model-specific). Candidates were the only two
sherpa-compatible options standing after the model hunt: English ERes2NetV2
was never released (3D-Speaker #208), leaving the Chinese-200k V2
(`v2zh`) — a language-domain change measured, not assumed — and WeSpeaker
ResNet34-LM (`resnet34lm`, dim 256, pyannote-3's own embedding).

The table in `PLAN-DIARIZATION.md` step 3 has the verdict: both regress loop
DER by ~6–7 pts (18.0 % → 23.8/24.8 %) and word attribution by ~7 pts
(89.7 % → 83.1/82.1 %), and the short-turn axis the step existed to fix
collapses — 3 s DIR@FAR0 79.2 % → 10.6 % (v2zh) / 0.0 % (resnet34lm), 2 s
49.1 % → 12.8 % / 0.0 %. Mechanism, visible in the operating thresholds:
both candidates' cosine distributions compress upward on our meeting domain
(FAR0 at 0.820/0.878 vs base 0.605), so stranger scores ride high and
short-cluster known scores fall below them. v2zh's one win — full-duration
DIR 91.5 % vs 88.7 % — is real but pays 5.8 DER points and the short-turn
collapse for it. Declined; ERes2Net-base stays. The recorded leads (EN
eres2net-large self-export, ReDimNet2) carry their triggers in the plan.

#### The known-count partitioner and the fold are one decision (2026-08-07)

Step 4 Phase B's verdict, shaped by two adversarial reviews whose blocking
findings each changed the answer. Shipped: **ward linkage**
(`OwnDiarizer` default) **plus the similarity-gated fold**
(`pipeline.FOLD_PAIR_SIMILARITY = 0.8` in `fold_excess_clusters`). Headline,
40 channels, known-count k+1: loop DER **16.5 → 13.2 %** (confusion
5.6 → 2.2 %), word attribution **91.3 → 94.8 %**, by-reference naming
own-name **80.0 → 84.0 %** time-weighted / **68.1 → 76.1 %** macro,
enrolled speakers >50 % misnamed **5 → 1** of 35. Worst per-channel DER
regression **+0.5 pt** (6 wins / 0 losses >0.5 pt, sign test p = 0.031;
every leave-one-group-out stays ahead). The evidence chain, all on the
frozen-artifact harness (`loop_freeze.py` + `loop_arm.py`, whose `--check`
parity gate byte-reproduced the full production matrix before any candidate
was trusted):

- **Partitioner sweep** (`linkage_sweep.py`, `out/diar-linkage-sweep.md`):
  ward 15.3 % raw-k+1 mean vs complete 16.9; **centroid declined with
  mechanism** — 142–213 linkage inversions per channel degenerate scipy's
  maxclust cut into one giant cluster + singletons (54.3 % mean); nmesc
  declined earlier (criterion inversion on Bmr021, structural failure on
  ES2007d — `cluster_ab.py`).
- **The fold is not neutral** (`fold_sweep.py`, `out/diar-fold-sweep.md`):
  the duration-spare fold (2026-08-06, co-tuned to complete clusters where
  the spare is junk) discards ward's advantage — under ward the spare is
  half a dominant talker (Bmr025: me011's 12.5 min split 6.6 + 5.6; the
  smallest-cluster rule can't repair a split of the *largest*, +21.3 pts).
  Oracle fold bounds: ward 12.4 % vs realized 15.0. Ungated max-pair stays
  declined: 24.7 % on complete. **ward@k (no spare) falsified**: 17.6 %
  mean, and the split persists at k (36.4 % Bmr025).
- **The gate** (`fold_gate_audit.py`, `out/diar-fold-gate-audit.md`): at
  every k+1→k fold step across 40 loop+duo channels × complete+ward,
  cross-speaker max pairs reach **0.697**; the merges the gate exists to
  admit (split dominant talkers) sit at **0.846–0.923**. The first candidate
  gate (0.6, reused from `COLLAPSE_SIMILARITY`) admitted 4 cross-speaker
  merges — review-caught: that constant's measurement (min-over-pairs,
  estimate-mode) does not transfer. 0.8 sits in the empty band, larger
  margin to the unrecoverable side; DER flat 0.45–0.80 (review's sweep).
- **k=2 rooms** (synthetic `.duo` channels, `ami.py`, opt-in): gated fold ≡
  duration fold on all 20 duos × both partitioners — the
  participant-deletion scenario never fires; ward also wins duos
  (11.4 vs 12.8 % mean).
- **Naming per reference speaker** (`naming_byref.py`,
  `out/diar-naming-byref.md`): per-cluster DIR/FAR is not comparable across
  partitioners (trial sets are partitioner-dependent — review finding), so
  denominators here are reference speech time, identical across arms.
- **Production parity**: the shipped `fold_excess_clusters` byte-matches the
  measured ward arm (`ami-loop-ward-sv08`) through the harness, and on
  complete clusters changes no turn or word on any of 40 channels (emb.json
  floats differ ≤1.9e-8 — ONNX extraction jitter in the emb caches, no merge
  decision moved). macOS/sherpa path: measured unchanged, worst cross pair
  0.595 vs gate 0.8.

Supersession made explicit: `test_spare_is_chosen_by_duration_not_similarity`
(2026-08-06's duration-only rule) is replaced by the below-gate/above-gate
pair in `tests/test_pipeline.py`; `COLLAPSE_SIMILARITY`'s docstring now
scopes its 0.95 cross-speaker figure to estimate-mode collapse, a different
regime from the fold gate.

#### The swap: OwnDiarizer is production, everywhere (2026-08-07)

`build_diarizer` constructs the owned loop on every platform (the numbers
above were measured on macOS, so the win ships there too); stenodiar/speakrs
keeps the estimate seat, sherpa-onnx stays for the embedding extractor, and
sherpa's `OfflineSpeakerDiarization` is called nowhere — `loop_parity.py`
constructs it directly from the installed package so the parity gate remains
re-runnable against the true reference. The stenodiar-less estimate fallback
is the loop's threshold mode (reference complete-linkage cut) — measured,
not asserted (2026-08-07 review): it diverges from sherpa's estimate mode at
turn level (5–12 % less emitted speech) yet lands within 0.5 pt DER on four
AMI channels, both over-splitting a 3-speaker channel into 66–105 clusters,
so the swap neither fixes nor worsens the degraded path stenodiar exists to
replace. Real-backend
structural coverage: `tests/test_diarization_loop_real.py` (gated on cached
models + eval audio).

#### The pool: 2.86× at bit-exact parity, every ORT session intra-op 1 (2026-08-09)

The loop's two model stages and `cluster_embeddings` submit into one shared
`ThreadPoolExecutor` sized to physical performance cores
(`embedding.py::_pool_workers`), every ORT session at ONE intra-op thread, one
shared extractor (measured bit-exact and as fast as per-worker copies at 8
workers; per-worker copies cost 26 MB each). `eval/pool_parity.py`, M4 Max,
ES2003c.loop 37.6 min:

| config | segmentation | embeddings | total |
|---|---|---|---|
| workers=1 (sequential, intra-op 1) | 38.7 s | 145.2 s | 183.9 s |
| workers=4 | 12.6 s | 52.0 s | 64.6 s |
| workers=12 (= P-cores, shipped) | 5.4 s | 33.7 s | **39.1 s** |

Against the pre-pool shipped config (111.7 s at intra-op 8, sequential):
**2.86×**. Gate half 1: labels, pairs, vectors and cluster embeddings
**bit-exact** vs workers=1 at 4 and 12 workers — exact equality, no
tolerance. Gate half 2: the full pooled pipeline byte-matches
`ami-loop-ward-sv08` RTTMs on all 20 loop channels (max emb drift 1.08e-7,
no merge decision moved) — the intra-op 8 → 1 flip moved nothing.
Two traps for re-runs: `ami-loop` is NOT a valid baseline (it predates
ward-by-default by six hours; pre-pool HEAD reproduces ward-sv08
byte-exactly but not ami-loop — verified before trusting the gate), and the
pre-pool loop is fully deterministic run-to-run even at intra-op 8, so any
byte mismatch vs the right baseline is real, never "threading noise".
Finalize watts (race-to-idle check): measured on x86 in the section below —
18.4 W across 150 s of diarizing finalize against a 3.5 W live-capture floor.
Reproducing the 111.7 s denominator needs pre-pool code (checkout 9a13e6c;
`OwnDiarizer` now hardwires intra-op 1) — the committed gate's own arms
support 183.9 → 39.1 s = 4.7×. Both parity runs executed niced
(backgrounded); their totals match same-day foreground component
measurements within noise, and half 1 runs its slowest arm first, so
thermal drift biases against the pool.

#### The x86 datapoint: the pool holds, the estimate path inverts, the online worker dies (2026-08-11)

Machine for every number here: GPD G1617-02, Ryzen AI 9 HX 370 (4 Zen5 +
8 Zen5c = 12 physical / 24 logical), 23 GB, CachyOS, **on AC, governor
`powersave` + EPP `power` as found** — ratios are under that power policy, and
this is a 12-core part under a handheld TDP cap, not the 4-core mobile Intel
the projections in `PLAN-DIARIZATION-SPEED.md` aim at. Same reference channel
as the M4 Max table above (`ES2003c.loop`, sha256
`af5f73cbc607eb6a5e1fbd7143880163af8deff5580255772be8f99208a607f6`,
2256.93 s). The loop finds **2361 pairs, matching the M4 Max exactly** —
strong but not conclusive evidence of the same channel, and the residue says
so: embedded audio measures 13509 s here against the 13192 s recorded
2026-08-09 (**+2.4 %**), and the plan called the channel 2251 s against
2256.93 s here. Run lengths therefore differ slightly while the pair count
does not, so the x86 embed stage does ~2.4 % more work than the M4 Max's and
cross-machine ratios are like-for-like only to that tolerance. The mixing is
deterministic per meeting (no RNG, int32 sums, per-meeting masks) but nothing
hashed the result until now; settle it in one command by comparing that
sha256 on the Mac.

**Repeats of one config span 11–21 % on this box, and that is the first
thing to design around.** Interleaved 12/8/12/8 worker arms: 145.6, 149.9,
161.0, 157.3 s — two identical 12-worker arms **10.6 %** apart, rising with
arm index; across all sessions the same `workers=12` config measured 133.2,
145.6, 150.1 and 161.0 s, a **21 %** spread. (Cause unattributed: no power or
temperature was sampled *during* the ladders, so thermal soak, clock ramp and
ORT arena reuse are all consistent with the ordering.) A ladder run in
increasing worker order therefore *manufactured* an 8-worker optimum (136.7 s
at 8 vs 150.1 s at 12) that the interleave refutes: 8 and 12 are
indistinguishable and `_pool_workers()`'s pick of 12 stands. Interleave every
arm on a fanless or handheld box, and treat any difference under ~15 % here
as unmeasured.

**Watt instrument** for every power number below: the amdgpu hwmon
`power1_average` (SMU PPT) rail, sampled at 1 Hz, averaged over the window
named with each figure. RAPL's `energy_uj` is root-only on this kernel, so
the rail was **not** cross-checked against it or against turbostat — the
*ratios* are what the conclusions rest on, the absolutes are single-rail.
Windows differ per row, which is why the same live-capture phase reads
1901 rpm over 60–2262 s and 1919 rpm over 60–900 s.

Worker ladder, segmentation + embeddings at intra-op 1 (same units as the M4
Max table; increasing order, so read only the shape):

| workers | segmentation | embeddings | total |
|---|---|---|---|
| 1 (sequential, intra-op 1) | 60.7 s | 475.5 s | 536.2 s |
| 4 | 21.1 s | 160.9 s | 182.0 s |
| 8 | 15.1 s | 121.6 s | 136.7 s |
| 12 (= physical cores, shipped) | 12.5 s | 137.6 s | **150.1 s** |
| 24 (= logical, SMT) | 10.9 s | 165.9 s | 176.8 s |

**The two stages want different pool sizes here, and one shared pool cannot
give it to them:** segmentation improves monotonically to 24 workers
(10.9 s, i.e. past the physical core count onto SMT siblings) while
embeddings bottom out at 8 (121.6 s) — both read from the same arms, so the
ordering bias is common to them. A split-optimal 132.5 s against the shipped
150.1 s prices the one-pool invariant at ~1.13× on this chip. That invariant
is deliberate (concurrent stages and channels can never oversubscribe the
machine) and 1.13× is inside this box's own repeat spread, so this is a
recorded observation, not a proposed change.

**The committed gate passes on x86** (`pool_parity.py --half 1 --workers
8,12`, own arms 480.5 → 135.7 / 133.2 s): labels, pairs, vectors, turns and
cluster embeddings **bit-exact** vs workers=1 at both counts. The one shared
`SpeakerEmbeddingExtractor` across 12 workers had been exercised on one
platform and one model only; it now holds on two architectures. Half 2 is
unrunnable here — it needs `out/diar/ami-loop-ward-sv08/`, which only the
machine that built the matrix has.

**The pool's win over pre-pool is ~1.4–1.5× here against the M4 Max's
2.86×,** and the difference is in the denominator: intra-op threading scales
2.43× at 8 threads on this chip (32.3 → 13.3 s, regressing at 16), so the
sequential-at-intra-op-8 config the pool replaced is comparatively strong.
Reconstructed pre-pool arm (sequential, both sessions at `_num_threads()` =
8): **219.3 s** — 1.46× against the ladder's 150.1 s, 1.43× against the
interleave mean of 153.3 s. Against sequential-at-intra-op-1: 3.6×. Two
caveats the ratio does not carry: the pre-pool arm ran **last**, after ~25
minutes of continuous load, which on this box biases *for* the pool (the M4
Max run above deliberately ordered the other way); and the 2.43× intra-op
figure comes from a single increasing-order pass over the meeting's **first**
200 slices (4.96 s mean against the channel's 5.72 s), i.e. exactly the
protocol the batching paragraph below shows to be unsafe here. Treat it as
"intra-op scales materially better on x86 than the 1.5× recorded for ARM",
not as a calibrated ratio — the ARM figure's own protocol is not recorded
anywhere.

**Batching still loses — no per-platform branch.** Per-item cost at intra-op
1, warmed 5 s, timed in two passes of opposite order (spread ≤ 1.7 %):
embedder 0.92 / 0.86 / 0.83 / 0.79× at batch 2 / 4 / 8 / 32; segmenter 1.13×
at batch 2, flat after. ARM's conv collapse (0.26×) is much milder here, same
sign. Two things make the verdict stronger than the ratio: the timed input is
*uniform*-length, which is batching's best case (real pair slices vary, so a
batch pays padding), and padded batching needs masked statistics pooling the
ERes2Net graph does not have — i.e. it is L1, not a cheap branch. **Trap:** a
first run of this arm said batching *won* 1.9×; a cold single-threaded ORT
call on this chip measures 1.8× a warm one (249 vs 140 ms), so an unwarmed
increasing-batch sweep reads clock ramp as a batching win.

**The path split inverts off macOS**, which changes who the speed program is
for:

| arm | this box (ORT CPU) | M4 Max (CoreML) |
|---|---|---|
| known count → owned loop | **154.2 s** (RTF 0.068) | 126.2 s |
| estimate → stenodiar + naming | **256.1 / 292.6 s** (RTF 0.11–0.13) | 14.1 s |

The helper alone is 241.9 / 278.8 s here against 3.4 s on CoreML, so the
estimate path costs **1.66–1.90× more than the owned loop** where macOS makes
it ~9× cheaper (8.95× from this table's own arms; 8.7× is step 0's second
run). `cluster_embeddings` is 14 s on both paths, matching the M4 Max's
10–15 s. The estimate arm finds 5 speakers where the reference has 3 — the
same accuracy caveat the M4 Max arm carried. Two limits on how far this
travels: the M4 Max known-count arm passed k=3 where this one passes the
production k+1=4 (k-invariant per step 0's decomposition, but not the same
call), and the **direction** generalizes off macOS because the execution
provider does (`speakrs.py` picks CoreML only on darwin) while the
**magnitude** does not — the owned loop here rides a 12-worker pool, and
stenodiar's own threading was never characterized, so on a 4-core box the
inversion could narrow or vanish.

**The whole end-of-meeting wait, one real replay** (both reference channels at
meeting cadence, `--local 1 --remote 3 --notes`, 37.6 min, ollama qwen3:8b on
CPU):

| phase | wall | package | Tctl | fan |
|---|---|---|---|---|
| live capture | 2262 s | 3.5 W | 54.5 °C | 1901 rpm |
| finalize mic (solo, live decodes reused) | 8.5 s | 17.5 W | 61 °C | 1875 rpm |
| finalize system (3 speakers, diarizing) | **150.1 s** | 18.4 W | 74 °C | 3861 rpm |
| notes, single pass | **169.8 s** | 15.6 W | 74 °C | 4002 rpm |
| **total wait** | **328.4 s** | | | |

Diarization is **45.7 %** of that wait on the known-count path and ~60 % on
the default estimate path (≈ 256–293 s + 170 s). Both channels finalized
"reusing live decodes": the live pass shed exactly 0 across 37.6 minutes.
On the suspicion this run existed to test — that notes would dominate the
wait on a CPU-only Ollama box — the answer here is no, but hold it loosely:
n=1 meeting, one model, one language, a 33 KB transcript, and the run sent
no `think` field, so qwen3 ran at the server default rather than a pinned
thinking mode. Ollama on this box is genuinely CPU-only (the package ships
only `libggml-cpu-*.so`). The notes output was clean and template-complete,
which is incidental evidence for `PLAN.md`'s unrun Ollama gates, not a
substitute for running them.

**The online worker's gates, measured without writing one.** `OwnDiarizer`'s
real stages are trickle-paced one 10 s window per second of wall clock —
production's stride, so this is meeting cadence, not an accelerated proxy —
beside a 15-minute `--replay` of the same meeting. Only the 3-speaker channel
is loaded: a 1-speaker channel never reaches the diarizer (`session.py` drops
it), so trickling both would impose ~1.6× the duty a worker actually would.

| gate | bar | measured | verdict |
|---|---|---|---|
| 1 share | diarization ≥ 40 % of the wait | 45.7 % (known count), ~60 % (estimate) | pass |
| 2 absolute | ≥ 90 s | 150.1 s (known count), 256–293 s (estimate) | pass |
| 3 non-regression | `shed_by_channel` exactly 0.0 | both channels finalized "reusing live decodes", which `session.py` grants only on shed == 0.0 | pass, over 900 s |
| 4 thermal | watts **and** fan indistinguishable from unloaded | fan indistinguishable; package power **+43 %** at the real duty (below) | **fail** |
| 5 path coverage | the owned loop carries real meetings | off macOS it carries the *cheaper* path, and the default path is dearer | pass |

Duty cycle is the number gate 4 is really about: **0.44 core** on the
3-speaker channel with zero late windows, measured *under* the concurrent
live pass (0.45 when both channels were loaded — the second channel's work
barely moves the first's). That is ~2× this box's own uncontended cost for
the same work
(the sequential ladder arm is 536.2 s of CPU for 2256.9 s of audio =
0.238 core); the difference is contention plus clock policy, and the
contended figure is the one a worker would actually pay. The comparable M4
Max number is 0.081 core (183.9 s / 2257 s) from the sequential arm above —
a ~3× gap, not the "~5 % of one core" the plan carried, which was never
measured.

Gate 4's power comparison: 1 Hz PPT samples over the **same** 60–900 s
window of each run, n=839 in all three arms.

| arm | package | Tctl | fan |
|---|---|---|---|
| unloaded capture | 3.57 W | 56.7 °C | 1919 rpm |
| + worker on the diarized channel (the real duty) | **5.09 W (+43 %)** | 54.6 °C | 1886 rpm |
| + worker on both channels (upper bound, ~1.6× duty) | 5.77 W (+62 %) | 59.3 °C | 1994 rpm |

**Only the power rail carries the load signal; temperature and fan do not
survive a between-run comparison here at all** — the honestly-loaded arm ran
*cooler and quieter* than its unloaded comparator, because each run's thermal
state at minute one dominates a 2 W difference. That is also the method
caveat: these arms are separate runs rather than interleaved, which is the
discipline this section otherwise preaches, so read the fan/temp columns as
noise and the watt column as the measurement. Priced from the watt rows and
stated as derived, not measured: +1.53 W held across a 37.6-min meeting ≈
**3.5 kJ**, against the offline finalize's 150.1 s × (18.4 − 3.5 W) ≈
**2.2 kJ** for the same work — race-to-idle wins on energy, and +43 % is far
outside the 11–21 % repeat spread.

#### Owning the embedder's front end: the config, and how to check one (2026-08-11)

The shared-trunk route needs to feed the embedding model ourselves, which
needs our own fbank to match sherpa's. `eval/fbank_parity.py` gates that, and
passes with room: embeddings reach **worst 1−cos = 3.0e-11** (float32 front
end 1.3e-8) against a 1e-4 bar, over 0.5/1/4/10/30 s clips and over
production's real pair-slice shapes — 10–40 pieces of ~272 samples each, an
order of magnitude shorter and more fragmented than contiguous speech.
Features agree with sherpa's own `get_frames()` to **4.9e-4 in the log
domain**, which is float32-vs-float64 rounding rather than a difference: a
float32 front end lands in the same band and still passes.

**Four options differ from Kaldi's defaults**, and none is guessable from the
model: `num_bins` 80, `dither` 0, `snip_edges` false, and `high_freq`
**7600 Hz rather than Nyquist**. Get any single one wrong and cosine lands in
0.66–0.98 — measured one-at-a-time against the correct baseline: log floor
not at float32 epsilon 0.67, `high_freq` at Nyquist 0.76–0.93, no preemphasis
0.92, `low_freq` 0 0.92, `snip_edges` true 0.97, FFT size 1024 0.98,
zero-padded edges instead of Kaldi's mirroring 0.999. **That band is the
lesson:** a wrong front end does not announce which constant is wrong, and
several plausible single mistakes are indistinguishable by cosine alone. The
per-call mean subtraction sherpa applies outside the graph is part of the
front end too — without it embeddings collapse to cosine 0.10–0.45.

So check against ground truth rather than inferring from cosine. Two facts
make that cheap, and are the reusable part of this entry: sherpa's
`OnlineStream.get_frames(index, n)` is documented "for comparing FBANK
features across pipelines" — but it aborts the **process** on an
out-of-range request, so compute the frame count rather than probing for it;
and `kaldi-native-fbank` on PyPI is the same k2-fsa library sherpa bundles,
so sweeping option sets against it in a throwaway env (never a dependency)
names the config in one pass.

One trap the gate itself had to fix: the reference channel is ~45 % exact-zero
samples, and silent clips reproduce under *any* config (cosine > 0.999 even
with `high_freq` wrong), so a uniform sampler quietly filled a third of the
short regimes with clips that cannot fail. The harness now draws until each
regime holds live audio.

#### Fewer/longer embedding units: L2 and L3 declined at kill-test (2026-08-09)

`eval/embed_units.py` (freeze-based, all 20 loop channels, production fold;
DER family B): **L3** — embed each maximal global sole-speaker interval
once, assign pairs by overlap — cuts embedded audio 9.9× but lands mean
DER 13.7 % vs the ward baseline's 13.2 % with worst-channel **+3.5 pt**
(Bmr030, k=4), 7× over the ward-arm shipping bar (worst +0.5 pt).
**L2** — ward only every-3rd-chunk's frozen vectors, propagate — mean
14.3 % with a catastrophic **+17.1 pt** on TS3010d (min naming cosine
0.756 vs baseline). Declined without buying a matrix — the kill-tests
exist to fail this cheaply. One attribution finding: on ES2003c, thinning
clustering evidence 3× leaves naming cosines at 1.0000, but
TS3010d/Bmr024/IS1009a corrupt clusters with boundaries byte-identical —
so cluster-evidence thinning damages naming *by itself* on some channels,
and the declined-stride bullet's mechanism reading must NOT be narrowed to
boundary-coarsening-only. The surviving seconds-removal path is the
shared trunk (`PLAN-DIARIZATION-SPEED.md` step 5 L1), which keeps the
clustering units and their values.

#### Stride stays at 1 s: the speedup is paid in naming purity (2026-08-07)

`OwnDiarizer(shift=…)` is the knob the owned loop unlocked; strides 2 s and
3 s ran the full ward+gated-fold arm each (`loop_freeze.py --shift-s`,
per-stride freeze dirs). Measured, strides 1/2/3: loop DER
**13.2 / 13.5 / 14.4 %**, attribution **94.8 / 94.4 / 93.5 %**, diarization
stage **1× / 2.1× / 3.1×** faster. The kill is naming: the FAR0 operating
threshold jumps 0.592 → ~0.78 at both larger strides (coarser turn
boundaries put more cross-speaker audio in every cluster embedding, so
stranger scores ride up), which silently invalidates the step-2.4
calibration — at the shipped 0.62, trial FAR goes 0 → 5.0 → 13.6 %, and
by-reference stranger misnaming 12.8 → 12.9 → 18.5 % with catastrophic
speakers 1 → 2 → 4. The research record's "8.5× at ≈0 DER cost" did not
survive contact with our stack (cost scales ∝ chunks, and the quality cost
is real). Declined both; re-open trigger = finalize latency becoming binding
or step 5 needing the budget, and the first move then is the shared-trunk
route (one trunk pass over long blocks, masked statistics pooling per pair
— attacks the contamination mechanism instead of paying it; SDBench's
"per-chunk" method, = `PLAN-DIARIZATION-SPEED.md` step 5 L1, with a
published accuracy prior of DER 0.255 → 0.257. Disambiguated 2026-08-09:
read as one-pass-per-10-s-window instead, the variant is backwards for our
sole-speaker-cropping loop — ~147 s vs 93 s at 1.05 pairs/chunk, flipping
only above ~1.7), with any stride change re-running the threshold
calibration as part of its gate.

#### Post-swap recalibration: 0.62 → 0.56, auto-updates stay declined (2026-08-07)

The first full matrix on the shipped OwnDiarizer stack (43.7 min, gate ≤1 h
holds) reproduces the ward arm's numbers through production — loop DER mean
13.2 %, attribution 94.8 % — and moves the naming headline to **DIR 94.1 % @
FAR 0 %** (FAR0 threshold 0.592; known ceiling 3/51 wrong-profile, was 4/53).
Trial composition shifted with the new clusters (51 known / 19 unknown), so
pre-swap naming numbers are not comparable to successors. Two owed re-runs on
this matrix:

- **`threshold_pick.py` (supersedes the 0.62 section above).** Ward removed
  the very strangers 0.62's margin was bought against: the impure-enrollment
  leak (0.860/0.681/0.622) is gone, and every measured stranger now tops out
  at 0.529 same-group / 0.531 cross-group — the survivors are TS3010's
  MTD040ME pulling toward MTD039UID at ~0.52. Both arms hold FAR 0 % / FRR
  0 % at full duration on the whole 0.53–0.56 plateau; **0.56** (the
  cluster-arm FAR0-strict point) keeps a 0.029 margin over the highest
  stranger anywhere — the same class as the 0.023 that justified 0.62 — and
  recovers what 0.62 was paying on the new curves: +13.7 pts DIR at 3 s of
  clean speech, +5.9 at 2 s, and one known trial per arm at full duration,
  for zero measured FAR. Shipped as `DEFAULT_THRESHOLD = 0.56`
  (`voiceprints.py`). Step 1.5's re-open trigger (<0.4) still does not fire.
- **`auto_update.py` (step 2.3's re-open trigger fired: the IS1009 confusion
  behind all three wrong updates is fixed).** The decline survives with the
  poison's shape changed: FIO084 now pulls 0.655/0.688 — below every margin
  gate — but ward-era clusters supply new ungateable wrong updates: Bmr024's
  13 s S5 enrolls into fe008's profile at **0.838 with a 0.603
  top-minus-second gap** (survives margin .20; higher margin than most
  correct updates), Bmr025's S2 into me001 at 0.732 (survives margin .10).
  On the benefit side the oracle still ties no-update at the shipped 0.56 at
  every duration (≤1 trial of separation anywhere; the FAR0-at-lower-
  threshold gains at 3 s exist but sit below the operating point). Profile
  growth stays user-confirmed; the next diarization change re-runs this
  harness as part of its gate.

#### DiariZen meeting-base: measured on the harness, declined (2026-08-09)

The one shippable model the research record put clearly above the pyannote
family (`eval/diarization-sota-2026.md` §1.1: AMI-SDM 15.6 vs 21.1, MIT,
meetings-only training) loses to the shipped OwnDiarizer stack on our
topology. `eval/diarizen_arm.py` ran it as exactly the replacement the plan
described — DiariZen's turns, our words/embeddings/scoring — in three arms
(reports: `out/diar-ami-diarizen{,-exactk,-est}.md`), with the decision rule
pre-registered in the arm's docstring *before* the matrix landed:

| arm | loops ΔDER (mean/median, n=20) | DZ wins | duos ΔDER | DZ wins |
|---|---|---|---|---|
| **k+1 + production fold** (fair) | **+1.3 / +1.2** | 4/20 | **+2.0 / +1.3** | 3/20 |
| exact k (no fold) | +11.7 / +12.0 | 2/20 | +16.7 / +6.9 | 1/20 |
| estimate (as published) | loop DER mean 17.1 % (vs 14.5 % k+1+fold — its own best framing) | — | — | — |

Positive Δ = DiariZen worse. Loop DER mean 14.5 % vs production 13.2 %; word
attribution mean **−1.1 pts** (wins 6/20); sign tests p≈0.012 (loops) and
p≈0.003 (duos); every leave-one-group-out keeps production ahead (+0.8 to
+2.0). The duo arm is also the near-field go/no-go the plan required
(training data exclusively far-field): 13.3 % mean vs 11.4 % — no. Declined
on the pre-registered rule, which required DiariZen to *win* both mean and
median by ≥2 pts.

What the arms establish beyond the verdict:

- **The fair frame was the 2026-08-09 review's catch, and it mattered ~13
  pts**: the first smoke ran DiariZen at exact k (26.1 % on ES2003a.loop)
  when the k+1 + `fold_excess_clusters` layer it would ship under cuts that
  to 12.0 % — the fold lives in the pipeline half step 5 never replaces, and
  exact-k is a configuration our own stack loses 4.4 pts with. The exact-k
  arm stays as the ablation: the fold is worth ~10 pts mean on DiariZen
  clusters too.
- **The 0.8 fold gate transfers to DiariZen clusters, measured**
  (fold-audit table in `out/diar-ami-diarizen.md`): all 22 gate-admitted
  merges (cosine ≥ 0.8) are same-speaker by reference; every cross-speaker
  fold (7, all forced through the duration-spare path by the stated count)
  sits at ≤ 0.702 — the same empty band the gate was calibrated in on ward
  clusters.
- **The known count helps DiariZen only through the fold.** The raw forced
  count *reproduces* the research record's warning (exact-k 24.9 % loop mean
  vs estimate's 17.1 %); count-plus-fold beats both. Estimate mode (its
  published configuration) over-counts 15/16 of the k=3 channels (bound
  3–6) and lands 2.6 pts behind the k+1+fold arm.
- **`min_cluster_size` is not the excuse**: the checkpoint's AMI-SDM-tuned
  mcs=30 swept {30, 15, 5, 1} at k+1+fold on ES2003a/IS1009a/Bmr025 loops
  (`out/diar-diarizen-mcs.md`) — DER identical at 30/15 on all three
  (12.0/46.6/13.8 %; still flat at 5 on the k=3 channels), collapsing below
  (mcs=1: 55.0/52.5/42.8 %). The loss is the model's on this topology, not
  a transplanted hyperparameter's.
- **The counter-signal, recorded honestly**: DiariZen wins all four ICSI
  loops (k=4–8: DER −0.4 to −2.5, not monotone in k; attribution +0.8 to
  +2.8) while production wins the 3-speaker AMI loops and their duos
  **31/32** — the product's common case per the meeting-mix weighting. n=4
  and one meeting series, so "many speakers" is confounded with "ICSI"
  (different corpus, recording chain, reference granularity); what partly
  de-confounds it is within-corpus: the four ICSI *duos* (same audio, k=2)
  go to production at mean +0.7 while the ICSI loops go to DiariZen at
  −1.6 — the flip tracks speaker count, not corpus. A lead, not a mechanism.
  If the product's meeting mix shifts toward many-speaker rooms, this is
  the number to re-open on.
- **Harness economics** (both shortcuts parity-checked before use, both in
  the arm's docstring): PyTorch-on-MPS runs the pipeline at RTF 0.06–0.08
  vs 1.22 CPU (estimate-mode turns byte-identical; known-count cross-DER
  0.064 %, ΔDER 0.019 pts on IS1009a.loop), and caching the k-independent
  stage 1 (segmentation + chunk embeddings) makes every arm after the first
  nearly free (0.6 s vs 126 s per channel) — three full 40-channel arms for
  ~2.5 h of compute instead of ~75 h naive.

Even a win here would only have opened the adoption program — ONNX export
of WavLM-Base+ + Conformer as the first task, then the step-2.3/2.4 re-runs
and fold/collapse gate recalibrations. The loss closes step 5 with nothing
owed: production diarization is unchanged, so the 0.56 threshold and the
auto-update decline stand unmodified.

#### The estimate path, and Nemotron 3 Diarization as the one diarizer, measured on x86 (2026-11-12 – 11-15)

Box: Ryzen AI 9 HX 370 (4 Zen 5 + 8 Zen 5c, 24 threads), Radeon 890M iGPU
(RDNA 3.5, Mesa RADV Vulkan, no ROCm), Linux, mains, `balanced` profile.
Single runs unless a repeat is given — the 11–21 % repeat spread of the
2026-08-11 x86 datapoint applies. The held-out comparison reproduces with
`eval/heldout.py` (fetch, production's arms, scoring) and
`eval/nemotron_arm.py` (the Nemotron arm and its parity check against the
PyTorch reference); the other measurements' scripts were throwaway, so their
decision-relevant numbers are all here.

**The estimate path is the less accurate one, not only the slower one.** ES2003c
through production code on `main` (stenodiar estimate + `collapse_single_voice`
vs the owned loop at k+1 + fold), DER at the 0.25 s collar:

| channel (k) | estimate: speakers / DER | known count: speakers / DER |
|---|---|---|
| loop (3) | 5 / 13.3 % | 3 / 5.5 % |
| duo (2) | 4 / 11.9 % | 2 / 5.0 % |
| mic (1) | 2 → collapsed to 1 / 5.7 % | (count 1 skips the diarizer) |

The extra error is speaker confusion (7.5–7.9 pts vs 0.2–0.6). Every phantom
cluster is made of fragments — mean turn 0.7–1.4 s against 7–15 s for the real
speakers — and two of the four are mixtures of all speakers, so the
cosine ≥ 0.8 gated merge fixes the duo (2 speakers, 6.0 %) but not the loop.
Dissolving clusters with a mean turn under 2 s into the time-nearest surviving
turn simulated to 6.4 % / 4.8 % with both counts right — one meeting, and the
2 s was picked after seeing these clusters, so it is a lead, not a result;
its failure mode (a participant who only backchannels) is untested.

**Estimate-path speed.** stenodiar `--mode cpu` takes 316 s for the 37.6-min
ES2003c.mic (1837 s CPU). It never batches embeddings:
`native/stenodiar/vendor/speakrs/src/inference/embedding/load/sessions.rs:162`
asks for `wespeaker-multimask-tail-b64.onnx`, the HF repo ships only `-b32`,
and the batch-1 model runs one chunk per call. A Python port of the same
`--mode cpu` pipeline on onnxruntime 1.31 with batched segmentation, fbank and
embeddings reproduces it (clip300 identical; full file 1 turn 16 ms early — one
frame whose two top segmentation scores differ by 6.7e-6, decided the other way
by the newer ORT) in **165 s** at intra-op 12 (2 repeats, 165.5/165.2; 1934 s
CPU) — ORT's default of 24 threads is slower (181 s) at double the CPU.
Embeddings are 81 % of it (59.5 ms/chunk), fbank 7.6, segmentation 6.2;
clustering and reconstruction under 0.5 s.

**speakrs 0.7's native CPU backend is a Linux regression.** The `speakrs-0.7`
branch (stock) takes 676 s on that channel (551 s of it system time:
`matrixmultiply` threads inside 4 workers oversubscribe). Four output-neutral
patches — the `gemm` crate off macOS, a worker budget of 12, a CPU multi-mask
batch of 24, a conv im2col tile of 256 columns — bring it to 266 s, ~1.2×
faster than best-pinned ORT on this 12-core chip but ~1.3× slower than ORT on a
4-core/8-thread or 4-slow-core emulation (`taskset`), because the win is
parallelism, not efficiency (the matmuls run at ~105–132 GFLOP/s per thread
single-threaded, 46–59 at 12 workers). The chip's sustained fp32 FMA peak is
1.15 TFLOP/s all-core (AVX2 ≈ AVX-512: 256-bit datapaths); the workload is
2.49 TFLOP per audio-minute.

**GPU routes for the ResNet34 multimask embedder** (ms per 10-s chunk; ORT 1.31
CPU at batch 32 = 61.9, all fp32 outputs checked against it): the
`onnxruntime-ep-webgpu` plugin (Microsoft, MIT, 0.5.0) 42.1 fp32 / 30.4 fp16,
numerically exact in fp32, 5 s CPU instead of 392 s; ncnn Vulkan 17.7 fp16 for
the trunk (a second engine, CPU path slower than ORT, no D3D12); IREE 3.13 →
Vulkan 100 at batch 1 with channels-last + im2col (368 default; batch 32 needs a
4.37 GB allocation over the 4 GB limit); Burn 0.22 165 at batch 1, device lost
at batch 32. A hand-written WGSL implicit-GEMM 3×3 conv (the 128-channel stage)
ran 3.9× ORT's single-conv time at ~2 TFLOP/s against the iGPU's 6.3 TFLOP/s
FMA peak — feasible, and declined because the plugin needs no kernel code.
The segmentation LSTM is 14× slower on WebGPU than on CPU.

**Nemotron 3 Diarization** (NVIDIA, 2026-09-23, OpenMDW-1.1, ~100 M params,
end-to-end, up to 8 speakers, no known-count input). Run through the
onnx-community ONNX export with a numpy port of transformers 5.23's offline
chunk loop and speaker cache, it matches the PyTorch reference to 5.6e-6 on CPU
and WebGPU. Three traps: the ONNX `attention_mask` is per *encoder* frame
(cached + chunk), not per mel frame; the processor marks the last mel frame
invalid, which masks the last encoder frame and moves the final chunk by up to
0.06 if ignored; the fp16 export keeps fp32 inputs/outputs. A second WebGPU
session in the same process segfaulted. Per 37.6-min channel: **25 s CPU, 14 s
iGPU fp32, 8.3 s iGPU fp16** (identical output fp32; fp16 flips one of ~48k
frame decisions on a 60 s clip).

On ES2003c it counts right (3 / 2) with **0.0 % speaker confusion**; its 23–26 %
DER is all missed speech — pauses the AMI segment-level references label as
speech, the mismatch NVIDIA's card warns about (it scores against
forced-alignment labels). `merge_words_turns` gives a word outside every turn
the nearest turn's speaker, so that miss does not reach the transcript. ES2003
is in AMI train, which Nemotron trained on, so the comparison that counts is
AMI's evaluation split — **IS1009, ES2004, TS3003, sessions a–d** (12 meetings,
36 channels), fetched with `ami.py`'s own functions but *not* added to
`AMI_GROUPS` (unvetted), scored on **transcript attribution**: the share of
single-speaker reference time the nearest-turn rule assigns to the right
speaker under the best one-to-one mapping (a phantom speaker's time counts as
wrong):

| 16 multi-speaker channels both arms ran | attribution mean (min) | speaker confusion | speed |
|---|---|---|---|
| owned loop, true k (production) | 95.1 % (76.0) | 1.9 % | 136 s / audio-hour (CPU) |
| **Nemotron 3, no count** | **97.1 % (94.0)** | **0.4 %** | 23 s / audio-hour (iGPU fp32) |

Per group, all 24 multi-speaker channels for Nemotron: IS1009 96.8 %, ES2004
97.4 %, TS3003 97.9 %. It over-counts by exactly one on 10 of them; on each, the
extra speaker's speech sits ≥ 60 % inside a meeting participant who is *not* on
that channel — headset crosstalk the channel reference does not list (IS1009's
loud Idiap headsets account for 8). Counting those as explained, its count is
right on 24/24, and on 12/12 single-speaker channels (corrected 2026-11-15:
the extras on ES2004c and IS1009a.mic hold under 1 % of the speech and had been
subtracted twice, so they scored as missing speakers). Applying the true count afterwards (keep the k most
active slots, give orphaned frames to the most probable kept slot) moves
attribution only on 7 channels, 6 of them IS1009: duo 97.6 → 98.2 % (worst
94.0 → 97.2), loop 97.1 → 97.3, mic 99.9 → 100. A count can only remove
speakers; Nemotron never under-counted here.

What this does not show: it is all AMI headset mixes, and attribution is a proxy on
reference speech, not an ASR word run. (The x86 known-count arm stopped short of
11 TS3003 channels and the estimate arm never ran here; both are complete in the
macOS run below.) The literature cuts the other way at k=2 — the strongest hybrid (DiariZen
large, CC-BY-NC) beats Nemotron there on a provisional third-party benchmark
(7.29 vs 10.31 DER; NVIDIA's own card shows Sortformer v2.1 ahead on 2-speaker
CALLHOME, 5.68 vs 5.98) — so non-AMI 2-speaker audio is the open gate
(`PLAN.md`). The architecture for cross-meeting names is settled practice
either way: diarize, embed each speaker with a separate voiceprint model, match
a profile store (pyannoteAI identification, NVIDIA's cascaded pipeline,
AssemblyAI's TitaNet recipe); no released model does both jobs, and
Nemotron's speaker-cache states are not voiceprints.

#### The held-out set on macOS, every arm (2026-11-15)

M4 Max (12 P + 4 E cores), 48 GB, macOS 27.2, mains; single runs, one arm at a
time. Same scripts and channels; production's arms on `main` with sherpa-onnx
1.13.8. The known-count arm's turns are byte-identical to 1.13.4's on the 28
channels both ran — the floor change in `pyproject.toml` is a memory fix only.
Nemotron's port matches the PyTorch reference here too: 7.6e-6 on CPU, 3.1e-6 on
WebGPU (Metal), no flipped decisions.

| all 24 multi-speaker channels | attribution mean (min) | speaker confusion | count right | s / audio-hour |
|---|---|---|---|---|
| stenodiar estimate (CoreML) + collapse | 86.5 % (66.6) | 7.9 % | 0 / 24 | 10.6 |
| owned loop, true k (production) | 95.7 % (76.3) | 1.5 % | 24 / 24 (given) | 55 |
| Nemotron 3, no count, CPU | 97.4 % (94.0) | 0.3 % | 14 / 24 (24 explained) | 15 |
| Nemotron 3, true count applied after | 97.8 % (95.0) | 0.2 % | 24 / 24 | 15 |

Nemotron on WebGPU scores identically to CPU at 5.5 s / audio-hour. Its numbers
reproduce x86's to the rounding (groups IS1009 96.7, ES2004 97.4, TS3003 97.9 %);
TS3003 does not change the owned loop's picture (96.9 % on its 8 channels; the
76.3 % worst is IS1009a.loop, as on x86). **The estimate arm over-counts every
multi-speaker channel** — by 1 on 9, by 2–5 on 15, 3–7 speakers found where
there are 2 or 3 — and only 3 duo extras are crosstalk-explained; single-speaker
channels it gets right (12/12). Under WebGPU macOS warns that two onnxruntime
copies are loaded (the wheel's and the plugin's) — no crash in these runs.

**Re-ID** (`eval/heldout_reid.py`, same day): galleries enroll from session a
(alphabetically-last participant per meeting unenrolled), every b–d cluster is a
trial against its own meeting's gallery, each arm's embeddings recomputed from
its final turns with production's `cluster_embeddings`. Cluster enrollment
(what `steno profiles assign` stores), full duration:

| arm | known / stranger trials | DIR / FAR / FRR @ 0.56 | FAR-0 threshold |
|---|---|---|---|
| owned loop, true k | 41 / 13 | 87.8 / 0 / 0 % | 0.593 |
| stenodiar estimate | 67 / 27 | 97.0 / 3.7 / 1.5 % | 0.595 |
| Nemotron, no count | 49 / 15 | 73.5 / 0 / 26.5 % | 0.377 |
| Nemotron, count applied after | 41 / 13 | **100 / 0 / 0 %** | 0.720 |

With the count applied, Nemotron is perfect at every threshold from 0.45 to
0.70, so 0.56 needs no move. The owned loop's five errors are all IS1009's quiet
FIO084 *named* FIO087 at 0.67–0.82 (the impure-enrollment pathology
`rename_once.py` found on the harness); Nemotron's session-a clusters are clean
enough that it does not occur. Without a count, Nemotron names nobody wrongly
but leaves a quarter of known trials unnamed, for two reasons, both checked: its
extra clusters are crosstalk carrying an on-channel reference label (all 8
headset-enrollment failures are those, 1–67 s each), and one such cluster in
session a (`IS1009a.loop/S3`, 42 s, cosine 0.046 to FIO084's real cluster)
joins FIO084's profile and drags the mean — removing it lifts FIO084's later
clusters from 0.41 to 0.84–0.85. Short clusters still collapse for every arm
(DIR at 0.56 of 34–55 % at 3 s / 2 s clean; FAR-0 points 0.25–0.45), the regime
`store_v2.py` already measured. Small: 3 meetings, 3 strangers, so one stranger
trial is 7.7 points of FAR; all AMI headsets. Count-1 channels in
`turns_known.json` are one channel-long turn, and the embedder's memory is
linear in slice length (~14 MB per second — 26 GB before the guard killed the
first attempt), so the script embeds their reference speech segments instead,
standing in for the transcript entries production embeds.

**Two speakers off AMI** (`eval/owncalls.py`, 2026-11-15): the user's own
recorded calls, labelled by channel instead of by hand — the stored recording
keeps mic and system tap apart (after echo cancellation), so each side's VAD
is the reference. 17 one-to-one online calls (19 h, German) were selected by a
calendar event with exactly one other participant; residual echo, measured as
mic VAD firing during remote speech, was 0–4 % on every call, so none was
dropped. Two sets, owned loop at k=2 (+1 fold) against Nemotron with no count:

| set | owned loop: attribution mean (min) / confusion | Nemotron | Nemotron better on |
|---|---|---|---|
| user + remote, mixed (17 calls) | 98.1 % (96.1) / 0.9 % | 99.3 % (98.3) / 0.1 % | 17 / 17 |
| two remote people taking turns (22 × 10.8 min) | 89.7 % (74.1) / 7.4 % | **98.6 % (81.7) / 1.0 %** | 22 / 22 |

The first set saturates: a laptop mic against a call codec splits by channel
colour, so it shows only that neither diarizer breaks on German call audio.
The second removes that cue — each dialogue splices two different calendar
partners' system-tap speech runs (11 partners, each partner's longest call,
each in four dialogues) end to end with 0.25 s gaps — and separates the
diarizers: the owned loop drops below 95 % on 15 of 22, Nemotron on 2 (one
pair at 81.7 % for both). Nemotron counted 2 on every dialogue and on 16/17
calls, at 5.4 s per audio-hour on WebGPU against 75–93 s for the owned loop.
What it does not show: the dialogues are spliced (no overlap, no backchannels,
uniform gaps), the two voices still come from different calls, and a shared
room mic — the hybrid-meeting case — has no free reference here.

**The library port** (`diarization/nemotron.py`, 2026-11-15, M4 Max,
onnxruntime 1.27.0, fp16, `heldout.py run --arm library`). Its chunk loop and
speaker cache are exact: fed the reference processor's features
(`--reference-features`, written by `nemotron_arm.py features`), its turns are
identical to `nemotron_arm.py`'s on 36/36 channels. Its own numpy log-mel front
end (no torch) is not bit-equal to the processor — 3e-4 worst, 1e-7 mean, 90 %
of values bit-equal on IS1009a.duo — and the speaker cache's near-tied
keep/drop choices amplify that: end-to-end turns are identical on 1/36
channels, first diverging minutes in (IS1009a.duo: at 308 s). The scores do
not move beyond the spread the reference runs already show among themselves:

| held-out set, 36 channels | mic count ok(expl) / attr | duo | loop | s / audio-hour |
|---|---|---|---|---|
| reference fp16 CPU | 6(12) / 99.9 % | 7(12) / 97.6 % | 7(12) / 97.0 % | 17 |
| reference fp32 CPU (and WebGPU) | 7(12) / 99.9 % | 7(12) / 97.6 % | 7(12) / 97.1 % | 15 |
| library fp16 CPU | 7(12) / 99.9 % | 7(12) / 97.6 % | 7(12) / 97.1 % | 17 |
| library fp16 CPU, true count applied after | 12(12) / 100.0 % | 12(12) / 98.2 % | 12(12) / 97.3 % | 17 |

Every extra is crosstalk-explained in every row (rescored 2026-11-15, after
`heldout.py score` stopped subtracting a sub-1 % extra twice). The
count-capped row equals the fp16 reference's to the decimal; confusion is
0.0 / 0.3 / 0.4 % without the cap and 0.0 / 0.2 / 0.2 % with it in every row.
A bit-exact numpy front end was not pursued: torch's own float32 STFT and GEMM
differ by platform, and even the range a frame is computed in changes its last
bits (a one-frame range is a matrix-vector product).

**No single-voice collapse behind Nemotron** (`eval/nemotron_collapse.py`,
2026-11-15). `pipeline.collapse_single_voice` folds an uncounted channel to one
speaker when all its cluster voices are ≥ 0.6 alike — measured against the owned
estimators, which split every monologue 2–3 ways. Run on Nemotron's uncapped
turns (library probabilities for the held-out set, the evaluation's for the
user's calls and dialogues), ERes2Net voices as in production:

| set | channels | over-counted | collapse fired | attribution before → after |
|---|---|---|---|---|
| held-out, one speaker (mic) | 12 | 5 | 1 (count fixed) | 99.9 → 99.9 % |
| held-out, 2–3 speakers (duo, loop) | 24 | 10 | 0 | 97.3 → 97.3 % |
| own calls, 2 speakers | 17 | 1 | 0 | 99.3 → 99.3 % |
| two remote voices, spliced | 22 | 0 | 1 (**real pair merged**) | 98.6 → 97.2 % |

It fixed one count and no attribution (IS1009a.mic: the extra slot holds 0.5 %
of the speech), and merged pair-5d729adf-c637582b's two speakers at cosine 0.646
(81.7 → 50.2 %). No threshold separates the two cases: the real pair sits above
the one collapse that helped (0.645), and the user's own calls reach 0.579. All
five single-speaker extras are another participant's crosstalk (≥ 60 % inside
an off-channel speaker's speech), not a split of the wearer — so the rule had
nothing to fix, and the pipeline no longer runs it.

**The assign guard** (`eval/assign_guard.py`, 2026-11-15). `steno profiles
assign` adds a meeting's voice to a named profile; a guard refuses an entry
whose mean cosine to the profile is below a bar. Held-out meetings, the
library's turns, profiles from the raw session-a headset (clean) or from
session-a clusters as `assign` would build them; entries are sessions b–d. A
phantom is a cluster ≥ 60 % inside off-channel speech (the crosstalk rule —
overlap with the labelled speaker alone calls crosstalk genuine, since it
co-occurs with the wearer's own speech); genuine entries are also re-embedded
from their first 3 s and 2 s of clean speech. No count, headset profiles,
entries refused:

| bar | genuine, full | genuine, 3 s | genuine, 2 s | phantom | wrong person |
|---|---|---|---|---|---|
| 0.30 | 0/54 | 7/54 | 13/54 | 10/11 | 123/162 |
| 0.45 | 0/54 | 15/54 | 21/54 | 10/11 | 159/162 |
| 0.56 (naming) | 0/54 | 24/54 | 28/54 | 11/11 | 162/162 |

Full-length genuine entries score 0.72–0.97, phantoms ≤ 0.50, wrong people
≤ 0.53, so 0.56 separates every full-length case; the session-a phantom
(`IS1009a.loop/S3`) scores 0.053. With the count applied there are no phantoms
and the same separation. The bar is set by what an admitted entry *does*:
each candidate added alone to its person's profile, that person's clusters
from other sessions re-scored at 0.56 (headset profiles, no count; cluster
profiles and the capped arm agree):

| entry admitted | similarity | later full-length named | later 3 s named |
|---|---|---|---|
| genuine 2 s | < 0.30 | 38 → 4 | 11 → 1 |
| genuine 2 s | 0.30–0.45 | 26 → 26 | 10 → 1 |
| genuine 2 s | 0.45–0.56 | 23 → 23 | 12 → 7 |
| genuine 2 s | ≥ 0.56 | 83 → 83 | 64 → 59 |
| genuine 3 s | ≥ 0.56 | 97 → 97 | 72 → 70 |
| phantom | < 0.30 | 29 → 3 | 13 → 0 |

An entry below the naming threshold costs its person their later short turns
whoever's voice it is, so the guard sits at the naming threshold and refuses
the right person's weak samples too (`--force` admits them). Small: 3
meetings, profiles of one to three entries — a larger profile dilutes one bad
entry more; all AMI headsets.

**A robust profile score** (`eval/profile_median.py`, 2026-11-15). A profile is
matched by the mean of its entries' cosines, which one bad entry moves by
(bad − good) / n. Leave one session out on the held-out meetings: a person's
profile is their genuine mic + loop clusters from the other three sessions
(45 of 48 profiles hold 3 entries, 3 hold 2 and are skipped — mean and median
agree below 3), trials are the held-out session's genuine clusters, count
applied; one bad entry joins one profile at a time, paired against the same
profile clean. Own trials named right at 0.56, trials cut to their first 3 s
(at full length every rule names 100 % in every condition; max alone falsely
names 6.4 % of other people's trials once a wrong person's entry is in):

| bad entry | mean | median | mean without the lowest | max |
|---|---|---|---|---|
| none (68 trials) | 61.8 % | 60.3 % | 61.8 % | 61.8 % |
| wrong person | 61.8 → 24.0 | 60.3 → 58.8 | 61.8 → 61.8 | 61.8 → 61.8 |
| crosstalk phantom | 50.8 → 26.2 | 50.8 → 49.2 | 50.8 → 50.8 | 50.8 → 59.0 |
| own 2 s sample below 0.56 | 68.3 → 45.7 | 67.2 → 66.1 | 68.3 → 68.3 | 68.3 → 68.3 |

Other people named as the owner stay at 0.0–0.5 % for mean, median and
mean-without-lowest in every condition; max reaches 5.8–7.2 %. Both robust
rules keep a 4-entry profile with one bad entry where it was clean; the mean
loses a third to two thirds of the short turns. Mean-without-lowest is exact
here partly by construction — with one bad entry it drops exactly that one —
and it survives only one bad entry at any size, where a median of 8 survives
three; profiles of 5–8 entries, and two bad entries, are untested (these
meetings give a person at most 3 sessions to enroll from). The median's clean
cost is one trial of 68. Small: 3 meetings, the same 68 trials reused across
every bad entry, all AMI headsets. Shipped: the median, for surviving more than
one bad entry once profiles grow.

**A real finalize, and its peak memory** (the `verify` skill's stored-meeting
recipe, 2026-11-15, M4 Max, 48 GB). One stored meeting of the user's, 1 h 45 min,
mic + system tap, replayed unpaced (`--no-live`: ASR decodes everything at stop)
and measured with `/usr/bin/time -l` on the venv's `steno` itself:

| run | wall | peak memory footprint | max resident |
|---|---|---|---|
| `--no-diarization` | 108 s | 6.9 GB | 2.9 GB |
| `--diarization`, no count (both channels) | 180 s | 11.2 GB | 6.7 GB |
| Nemotron alone, system channel (`diarize`) | 29 s | 1.8 GB | 1.8 GB |
| … plus the cluster voiceprints (`diarize_with_embeddings`) | 31 s | 2.7 GB | 2.8 GB |

Diarizing costs this meeting 72 s and 4.3 GB of peak. Alone, the diarizer
needs 2.7 GB on the 105-minute channel; in the pipeline it stacks on top of the
resident ASR weights and the meeting's buffers rather than replacing them,
which is most of the 4.3 — the remaining ~1.6 GB is not attributed. Resident
size undercounts the pipeline runs by 4–5 GB (MLX's unified-memory buffers):
read the footprint. It found 1 local and 6 remote speakers on a meeting with no
labels, so the count is unscored; three of the six hold 1–2 transcript entries
each, the shape of crosstalk phantoms, unverified. One meeting — a lead on the
memory cost, not a law; a 48 GB machine has room, an 8–16 GB one with the
desktop open may not, and nothing has measured one.

Where the stop's peak comes from (same meeting and machine, same day; a probe
that decodes both channels with the finalize windows — the live pass decodes
the same ones — then calls `MeetingRecorder.finalize` with those words as
`live_words`, so the stop never touches the ASR; footprint sampled every 50 ms,
its maximum within 1 % of `/usr/bin/time -l` on every run; notes and glossary
off):

| before the stop | footprint when the stop starts | stop's peak |
|---|---|---|
| ASR kept, as shipped | 5.7 GB | 10.2 GB |
| `mx.clear_cache()`, weights kept | 3.5 GB | 8.2 GB |
| ASR unloaded (`unload()` = drop weights + clear cache) | 2.2 GB | 6.2 GB |

After decoding, the ASR holds ~3.4 GB that the stop never uses: ~2.1 GB of
MLX's buffer cache and ~1.3 GB of weights. The stop itself then adds 4.0–4.6 GB
whatever was released first, and keeps it to the end of the process. Unloaded
first, the whole run's peak moves into the decode (6.8 GB) — in a real meeting
that decode is spread over the recording, and whether the live pass reaches
the same 6.8 GB is not measured. Speakers came out identical in all three.

Shipped the same day: `MeetingRecorder.finalize` decodes every channel that
needs it, unloads the ASR, then diarizes. Re-measured through the real
`steno start --no-live --diarization` on the same meeting: peak footprint
11.2 → 7.9 GB, wall unchanged (180 s), and the transcript identical entry for
entry (speaker, start, text) to the run before the change.

#### Speaker-embedding models in the naming seat: ReDimNet2 wins (2026-11-18)

`eval/reid_models.py` compares the shipped ERes2Net-base against six
candidates where the embedder now works: turning a finished cluster into a
voiceprint and matching it against saved profiles. Every model embeds the same
frozen items in its own reference implementation (`eval/embedders/`, parity
with each upstream inference path at cosine 1.0000000), so the comparison needs
no diarization rerun and a conversion bug cannot decide it. There are five trial
sets, weighted equally:

- AMI headset, AMI room mic (Array1-01), and AMI headset-enrolled with room-mic
  trials, each 24 people in six series;
- production Nemotron clusters, 12 people;
- the user's German one-to-one calls (remote side), 4 partners across calls
  and 12 within a call.

Every trial is scored twice: once as a known trial, and once as a stranger
with its own profile removed. Each (person, session) contributes its full
cluster, capped at 120 s, plus ten random 2/3/5/10 s excerpts. Spans are
combined by production's rule.

The decision rule was written before any candidate was scored
(`out/reid_models/decision_rule.md`). Its metric is macro DIR (named, and named
right) on 2 s and 3 s clusters, at the one threshold per model where the macro
stranger-accept rate is 1 %. The headline row averages the two durations, with
95 % paired intervals that resample people within each set.

| model | 2 s | 3 s | short (2+3) vs base | full | CPU s / 10 s speech, 1 thread |
|---|---|---|---|---|---|
| ERes2Net-base (shipped) | 64.5 | 77.6 | 71.1 | 99.4 | 0.15 |
| **ReDimNet2-B6, VoxBlink2+VoxCeleb2, LM** | **81.9** | **89.6** | **85.7 [+7.4, +18.6]** | 100 | 0.51 |
| ReDimNet2-B6, VoxBlink2+VoxCeleb2+CN-Celeb, LM | 80.9 | 89.4 | 85.2 [+7.3, +18.5] | 99.2 | 0.51 |
| ReDimNet2-B6, VoxBlink2+VoxCeleb2, before LM | 79.9 | 87.2 | 83.5 [+4.8, +16.2] | 100 | 0.51 |
| ReDimNet2-B6, VoxCeleb2 only, LM | 81.7 | 84.9 | 83.3 [+5.7, +16.2] | 100 | 0.51 |
| w2v-BERT 2.0 SV (580 M) | 76.4 | 86.8 | 81.6 [+4.9, +14.2] | 100 | 0.76 |
| ERes2Net-large (VoxCeleb) | 61.9 | 75.7 | 68.8 [−7.9, +1.1] | 99.4 | 0.24 |
| TitaNet-L | 58.8 | 69.4 | 64.1 [−12.5, −3.4] | 93.6 | 0.07 |

**The verdict is ReDimNet2-B6 trained on VoxBlink2+VoxCeleb2 with large-margin
fine-tuning (`redimnet2-b6-vb2-lm`).** Head to head, it beats every
non-ReDimNet model, and the "before LM" checkpoint, with intervals clear of
zero:

- w2v-BERT by +4.1 [+1.1, +6.5];
- the before-LM checkpoint by +2.2 [+1.0, +3.9].

It ties two siblings:

- the CN-Celeb variant, +0.6 [−1.5, +2.0], whose order flips when any one set
  is left out;
- the VoxCeleb2-only checkpoint, +2.4 [−0.3, +5.6].

The pre-registered tie-breaks pick it. On the room-mic sets (room and cross,
2 s and 3 s) it scores 73.0 against 71.8 and 69.6. The CN-Celeb variant also
drops three of 72 full-length room trials, against none for this checkpoint.

The gain sits where the product is weakest: short clusters on a room mic. At
2 s, room-mic DIR goes from 40.3 to 69.7 and headset-to-room from 34.2 to 61.9.
Equal-error rates fall in every set, from 4.7 to 1.9 % on headset and from 9.2
to 7.1 % on room.

In each robustness check, first place goes to this checkpoint or its CN-Celeb
sibling, never more than 1.5 points apart, and base trails by 9 or more:

- at 5 % FAR (92.3 vs base 83.3);
- with one threshold shared across all durations, as production has: 2 s 74.2
  vs 51.6, full 100 vs 99.7;
- leaving out any one set.

At production's literal 0.56 without recalibration, it scores 83.4 vs 64.8 at
2 s with FAR 1.2 vs 1.4 %. Its own 1 %-FAR thresholds sit at 0.57–0.62, so a
switch would re-pick the threshold near where it is now (`threshold_pick.py`).
Its calibration spread across sets at 2 s is 0.145, the same as base's 0.146.

What was ruled out:

- **Large-margin fine-tuning does not hurt short clips here.** WeSpeaker and
  DAME report it does below 3 s; on this data the LM checkpoint beats its own
  pre-LM weights at 2 s and 3 s.
- **CN-Celeb data adds nothing measurable** on German and English meetings.
- **ERes2Net-large is not worse at ranking.** Its per-set 2 s DIR@1 % is
  higher than base's (headset 92.5 vs 87.9), but its scores drift by domain
  (threshold spread 0.206). One fixed threshold therefore over-names on the
  calls and under-names on AMI.
- **TitaNet-L collapses headset-to-room.** At full duration it scores 68.1 %
  where every other model scores 100 %.
- **ReDimNet2+ does not beat it.** lab260's retraining of B6 for channel and
  codec shift (`redimnet2-plus-lmft`, `eval/embedders/redimnet_plus.py`, run
  in the authors' model code) scores 76.8 / 85.7 / 81.2 on the decision
  metric, 4.5 points [+0.6, +7.6] behind this checkpoint, and trails most
  where it was meant to help: room mic 62.9 against 69.7 and headset-to-room
  48.1 against 61.9 at 2 s. It is ahead only on production clusters at 2 s
  (88.5 against 87.4).
- **Refused spans do not explain the gap.** Every model refused some spans,
  but all were fragments under 0.1 s, which production's span rule skips
  anyway.
- **The 120 s cap does not change the result.** On base, capped and uncapped
  clusters gave identical naming on all 72 full trials. Cosine between capped
  and uncapped vectors was median 0.995, minimum 0.981.

Limits:

- The German calls hold only four cross-call partners, so on that set one
  person moves DIR by up to 25 points.
- Every model accepts about 4 % of call-side strangers at 2 s; they all share
  that weakness, so it does not separate them.
- ReDimNet2 costs about 3.4× base's CPU per second of speech.
- MPS is no shortcut for variable-length spans: per-shape compilation made
  ERes2Net-large about 10× slower than its benchmark. The run used CPU
  processes, one thread each (`eval/embedders/shard.py`); 37 jobs took
  1 h 50 min on 12 cores.

#### How much of a cluster ReDimNet2 needs, and how small it can be (2026-11-18)

Both measured on the frozen items of the section above; the new embeddings
were computed on a Ryzen HX 370 under Linux, and its vectors match the M4
Max's (310 clusters embedded on both, cosine 1.00000).

**Cluster cuts.** Every full cluster and every enrollment was cut five ways
(`reid_models.py build-caps`, 63.8 h in all) and scored on the 287 full-cluster
trials of the five sets (72/72/72/54/17; `score-caps`), at each model's
all-duration threshold (base 0.616, B6 0.621). DIR is the macro over the sets;
margin is the worst trial's true-profile score minus its best stranger's.

| profile cut → cluster cut | base DIR | base margin | B6 DIR | B6 margin |
|---|---|---|---|---|
| first 30 s → first 30 s | 99.2 | +0.090 | 99.7 | +0.195 |
| first 60 s → first 60 s | 99.7 | +0.054 | 100 | +0.162 |
| first 120 s → first 120 s | 99.7 | +0.108 | 100 | +0.215 |
| 120 s evenly → 120 s evenly | 99.7 | +0.108 | 100 | +0.218 |
| whole → whole | 99.7 | +0.088 | 100 | +0.203 |
| whole → first 30 s | 99.2 | −0.001 | 100 | +0.187 |
| whole → first 60 s | 99.7 | +0.035 | 100 | +0.173 |
| first 30 s → whole | 99.7 | +0.188 | 100 | +0.153 |

- **B6 names from the first 60 s of a speaker exactly as from everything.**
  No trial is lost in any combination at 60 s or more, and its worst margin
  never falls below +0.13 (first 30 s against first 120 s) — base's goes
  negative. A whole cluster holds a median 374 s of speech, so the first 60 s
  is about a sixth of the work.
- **30 s is the edge**: one room-mic trial of 72 goes unnamed when profile and
  cluster are both 30 s; none when the profile is whole.
- The first-60 s vector sits at cosine 0.979 (median; minimum 0.911) to the
  whole-cluster vector, first-120 s at 0.991 (0.940).
- Limit: full clusters are at the ceiling, so this shows no loss among 287
  trials, not that none exists; AMI meetings run about half an hour.

**Smaller sizes.** Decision metric as above (short = mean of 2 s and 3 s macro
DIR at 1 % stranger accepts; interval against base).

| model | training data | 2 s | 3 s | short | full |
|---|---|---|---|---|---|
| base | | 64.5 | 77.6 | 71.1 | 99.4 |
| `redimnet2-b0-vox2-lm` | VoxCeleb2 | 66.2 | 77.0 | 71.6 [−8.3, +4.4] | 98.9 |
| `redimnet2-b2-vox2-lm` | VoxCeleb2 | 66.1 | 79.2 | 72.6 [−3.7, +4.3] | 98.9 |
| `redimnet2-b3-vox2-lm` | VoxCeleb2 | 70.3 | 78.7 | 74.5 [+0.7, +7.3] | 100 |
| `redimnet2-b3-vb2cnc-lm` | + VoxBlink2, CN-Celeb | 77.8 | 87.0 | 82.4 [+5.9, +15.4] | 100 |
| `redimnet2-b6-vb2-lm` | + VoxBlink2 | 81.9 | 89.6 | 85.7 [+7.4, +18.6] | 100 |

- **B3 on the large training set is the only smaller size worth having**: it
  trails B6 by 3.3 points [−0.7, +5.5] — not separable — and is behind on
  every set at 2 s (room mic 64.2 against 69.7, headset-to-room 53.5 against
  61.9).
- **B0 and B2 are no better than base.** They are published on VoxCeleb2
  only, and at B3 that training set alone costs 8 points, so size and data
  are confounded below B3.
- Cost is not settled: the sizes ran on the Ryzen while its power profile
  changed, so their seconds do not compare. Energy per minute of speech on a
  quiet machine (`eval/embedders/energy.sh`) is unmeasured.

#### ReDimNet2-B6 as shipped: export, rule, threshold (2026-11-18)

M4 Max, onnxruntime 1.27.0, one intra-op thread unless stated.

**The export** (`eval/embedders/export_redimnet.py`) is the whole reference
forward pass, front end included, at any input length — the publishers' own
ONNX file is fixed at 6 s. Against the PyTorch model on noise of 0.5 to 10 s
the cosine is 1.0000000 (±1e-7) at every length. Through the production
`VoiceEmbedder` (`eval/embedders/shipped.py`) on 240 of the frozen items,
1,061 spans, the worst cosine to the reference runner's vectors is 1.000000,
and both refuse the same 21 fragments. 800 samples embed; 400 raise.

**Cost grows with the span**, in time and in memory the session's arena then
keeps:

| span | CPU-s per 10 s of speech | process RSS after |
|---|---|---|
| 1 s | 0.54 | 242 MB |
| 2 s | 0.50–0.56 | 320 MB |
| 5 s | 0.60 | 568 MB |
| 10 s | 0.68–0.70 | 980 MB |
| 30 s | 0.86 | 1.7 GB |
| 120 s | 1.66 | 8.1 GB |

(168 MB loaded and idle.) Over 1,500 calls at 451 distinct lengths up to 5 s
the process settles at 860 MB and stays there. Switching the arena off, or
asking it to shrink after each call, returns nothing on macOS. Concurrent
calls on one session at 5 s: one costs 0.59 CPU-s per 10 s, four 0.82 (1.4 GB),
eight 1.20 (2.5 GB) — the model is memory-bound, so the pool is four workers.

**The rule** — the first 120 s of a speaker's clean speech, spans over 5 s
embedded as equal parts, overlap judged on the diarizer's activity before a
stated speaker count is applied — against the plain first-120 s cut of the
section above, on the same 287 full-cluster trials (`build-caps` modes
`first120-split`, `-rule`, `-rule5`; 709 of 9,430 first-120 s spans are longer
than 10 s):

| cut | named right | worst margin | cosine to whole, median (min) |
|---|---|---|---|
| first 120 s, spans as they are | 100 % | +0.215 | 0.9910 (0.9397) |
| parts of at most 10 s | 100 % | +0.219 | 0.9905 (0.9420) |
| that, spans chosen by `VoiceCollector` | 100 % | +0.219 | 0.9903 (0.9420) |
| the same with parts of at most 5 s (shipped) | 100 % | +0.224 | 0.9889 (0.9417) |

Neither change costs anything measurable. On the production-cluster set
(54 trials) the collector's spans differ from the turn-based ones on 25 of 72
clusters and move no score by more than 0.001 at the extremes (worst known
0.796, best stranger 0.481 either way).

**The threshold** is 0.62 (`voiceprints.DEFAULT_THRESHOLD`), read off the
main run's trials: macro over the five sets, named right / strangers named.

| threshold | 2 s | 3 s | 5 s | 10 s | full |
|---|---|---|---|---|---|
| 0.56 | 83.4 / 1.2 | 91.9 / 2.1 | 97.0 / 2.8 | 99.5 / 3.6 | 100 / 6.4 |
| 0.58 | 80.6 / 0.8 | 90.3 / 1.2 | 96.1 / 2.3 | 99.2 / 2.8 | 100 / 3.5 |
| 0.60 | 76.9 / 0.4 | 88.4 / 0.9 | 95.3 / 1.4 | 99.2 / 1.9 | 100 / 2.4 |
| **0.62** | 74.2 / 0.1 | 86.6 / 0.6 | 94.3 / 1.3 | 99.0 / 1.2 | 100 / 2.4 |
| 0.64 | 70.8 / 0.0 | 84.1 / 0.2 | 93.1 / 1.0 | 98.7 / 1.0 | 100 / 1.2 |
| 0.68 | 63.0 / 0.0 | 77.2 / 0.1 | 89.5 / 0.1 | 97.6 / 0.3 | 100 / 0.0 |

0.621 is where strangers named average 1 % over all five durations. At full
length every stranger named comes from the German call set (17 trials, four
partners: 0.660, 0.621, 0.592); on the four AMI sets no full-length stranger
scores above 0.579 and no known speaker below 0.697. `threshold_pick.py` was
not re-run: it reads a diarize-and-transcribe pass over the whole corpus made
with the previous embedder, and these trials cover its arms (headset and
cluster enrollment, 2 s, 3 s and full) on more sets.

#### Voiceprints collected during the meeting (2026-11-18)

Each diarizer step's speech goes to the `VoiceCollector` in the same wake-up
(`NemotronStream`), so the stop embeds only what the last step or two
settled. The production path (`NemotronDiarizer().stream()` fed in 0.5 s
blocks, then `resumed().diarize_with_embeddings`) against one pass over the
finished recording, from stored meetings' `audio.opus`, with no count and
with a count of 2:

| meeting, channel | speakers (no count → count 2) | embedded while recording | left for the stop | turns and voiceprints |
|---|---|---|---|---|
| 6.1 min, mic | 2 → 2 | 65 spans, 176 s | 1 span, 1 s | bit-identical |
| 8.0 min, mic | 2 → 2 | 87 spans, 198 s | 3 spans, 4 s | bit-identical |
| 25.8 min, mic | 1 → 1 | 44 spans, 81 s | none | bit-identical |
| 25.8 min, system | 3 → 2 | 157 spans, 360 s | none | bit-identical |
| 49.4 min, mic | 4 → 2 | 100 spans, 248 s | none | bit-identical |

The stop's voice work fell to 0.2–0.4 s per channel on a performance core,
the diarizer's last steps included. A slot the count removes costs its
embeddings and nothing else. The 8-minute meeting replayed at meeting pace
through `steno start` (live captions, diarization, no count) and again
unpaced without the live pass gave the same two speakers; their voiceprints
in `voiceprints.json` differ by 1.5e-8 at most, one float32 step, in the
sidecar's re-normalisation. Not measured: the collection's energy on the
efficiency cores, and its memory beside a live meeting — the session alone
settles at 860 MB (section above).

#### Profiles carried across an embedder change by translation (2026-11-19)

A profile entry is readable only by the embedder that made it, so a model
change leaves every saved profile inactive. `eval/reid_bridge.py` asks whether
a map fitted on corpus audio both models embedded can translate stored
ERes2Net-base vectors into ReDimNet2-B6 space well enough to keep naming
people. It reuses `reid_models`' frozen items and trials (no re-embedding, 15 s
a run): profile entries are translated old vectors, trials are native
new-model vectors — the state right after an upgrade. The map never sees its
test people: AMI and Nemotron items fold by meeting series (fit on the other
five, about 20 voices, all English meetings), and the user's German calls are
tested with a map fitted on AMI alone. Fitted on deliberately mispaired spans,
the map names 0.2 % — the harness measures the map.

Macro DIR at each arm's own 1 %-stranger threshold, as in the section above;
brackets are the paired 95 % interval against ERes2Net-base:

| arm | 2–3 s | full | its threshold (2 s / full) |
|---|---|---|---|
| ERes2Net-base end to end | 71.1 | 99.4 | 0.562 / 0.663 |
| ReDimNet2-B6 end to end | 85.7 | 100.0 | 0.572 / 0.660 |
| translated by rotation (orthogonal Procrustes) | 85.7 [+6.5, +19.2] | 96.6 [−7.1, +0.8] | 0.462 / 0.532 |
| translated by ridge regression | 74.5 [−3.5, +7.7] | 92.0 [−17.2, −2.5] | 0.530 / 0.607 |

- The rotation names short clusters as well as fresh profiles do and better
  than the old model did; on full clusters it misses about three points
  against the old model, not resolved at this size (some 35 people, one full
  trial is 1.4 points of a set). Ridge, with more freedom, overfits 20 voices.
- On the user's calls, a map that never heard German or a phone codec: 97.6 %
  named at 2 s with 1.2 % strangers named, against 93.5 % / 4.7 % for the old
  model (about eleven partners).
- Translated entries need their own threshold. At the shipped 0.621 they name
  nobody wrongly but far fewer people (calls at 2 s: 42 % against 83 %).
  Their 1 %-stranger point barely moves between headset, room mic and phone
  (spread 0.021, against 0.145 for either model), so one fixed value should
  carry — inferred; every threshold here is picked on the test trials, and one
  chosen in advance has not been checked on unseen people.
- More voices help up to about twelve, then the gain stops showing at this
  size: fitted on 4 / 8 / 12 / 16 / 20 people (ten random draws below 20)
  the rotation reaches cosine 0.71 / 0.77 / 0.80 / 0.80 / 0.82 to the native
  vector, 70 / 78 / 84 / 84 / 86 % DIR on 2–3 s clusters and
  84 / 93 / 97 / 95 / 97 % on full ones. Fitted on everything, test people
  included, it reaches 0.94 and 99.4 % — partly memorisation, so a bound,
  not a target. Whether a map fitted on thousands of public voices closes
  part of that gap is open.

Not built: the store has no notion of a translated entry, and how a profile
mixing translated and native entries should be scored is open. For the
ERes2Net-to-ReDimNet2 change it was not needed: those profiles were never used
in practice, so the store deletes them on load (`voiceprints.RETIRED_MODELS`,
Daniel's call, 2026-11-20). The next embedder change decides afresh.

#### The chunk loop runs during the meeting (2026-11-17)

M4 Max (12 P + 4 E cores), 48 GB, macOS 27.2, onnxruntime 1.27.0, mains. A
live meeting now runs every Nemotron step whose features and attention mask
can no longer change (`ChunkLoop` in `diarization/nemotron.py`) on a
background thread while it records; the stop runs only the last one or two
steps. Every step reads only audio already in the append-only store, and the
offline `speaker_probabilities` is the same loop run in one go.

**Bit-identity**, the production path (`NemotronDiarizer().stream()`, fed in
0.5 s blocks, finished by `resumed()`), against the offline loop as shipped
before the change (ORT's default threads), from each meeting's stored
`audio.opus`:

| meeting | channel | steps streamed / left at the stop | probabilities | turns |
|---|---|---|---|---|
| 2.9 min, mic nearly silent (1 % of seconds active) | mic, system | 6 / 1 each | bit-identical | identical |
| 22.9 min, system nearly silent (4 %) | mic, system | 50 / 1 each | bit-identical | identical |
| 25.4 min, both active | mic, system | 55 / 1 each | bit-identical | identical |
| 104.3 min | mic, system | 229 / 2 each | bit-identical | identical |

Through `MeetingRecorder.finalize` on the 104-min meeting (live words reused,
re-ID on the real profile store), the transcript is identical entry for entry
(465 × speaker, start, end, text), and so are the detected counts, the
profile names and every voice embedding, bit for bit. Every thread count
from 1 to 16 gave bit-identical probabilities (2.9-min meeting; the 1-thread
session on all eight channels above), which is what lets the stream use a
one-thread session and the stop switch to the offline one between steps.
Weight prepacking stays on in both sessions. Off, the streaming session
would hold 0.4 GB less, but the probabilities move: on the M4 Max only the
final, shorter chunk (up to 2.4e-8 on the 104-min meeting), on the Ryzen AI 9
HX 370 every frame (up to 6.8e-6; one 10-ms frame of 41,553 speech frames
changed its speaker, 25.4-min meeting) — a platform-dependent result for
memory that a 48 GB or 24 GB machine does not need. The memory arena off
alone is bit-identical and saves 0.02 GB, not worth a switch.

**The stop** of the 104-min meeting (`MeetingRecorder.finalize`, live words
reused, embedder warm, footprint from `proc_pid_rusage`):

| | the stop | footprint when it starts | process peak |
|---|---|---|---|
| offline, as before (session loaded at the stop) | 76.2, 76.1, 74.2 s | 5.5–5.6 GB | 6.7–7.4 GB |
| streamed | 11.1, 10.9, 10.3 s | 4.4 GB | 6.4 GB |

About 9 s of the streamed stop is the voice embeddings; the two remaining
steps take 0.4 s per channel. One more streamed run took 118 s, on a machine
that was busy at the time (its streamed phase ran 171 s at 85 % P-core where
the quiet runs took 128 s at 100 %); not reproduced in three reruns, cause
unknown.

**Energy per step**, by session configuration: one process per arm, 8
warm-up steps then 30 timed steps at one step every 2 s (or twice the step
time, if longer), so every step starts from idle as in a meeting; CPU energy
of the process from `proc_pid_rusage` (`ri_energy_nj`), time on P-cores from
`ri_*_ptime`; two interleaved repetitions. A channel takes one step per
27.2 s of audio, 132 per audio-hour.

| threads | spin-wait | thread QoS | step | on P-cores | J / step | J / audio-hour / channel |
|---|---|---|---|---|---|---|
| default (16) | on | default | 0.15 s | 87 % | 3.68, 3.56 | 479 |
| default (16) | off | default | 0.14 s | 99 % | 1.70, 1.66 | 222 |
| 12 | on | default | 0.14 s | 90 % | 5.03, 4.40 | 625 |
| 12 | off | default | 0.13 s | 98 % | 1.69, 1.69 | 223 |
| 4 | on | default | 0.23 s | 78 % | 3.53, 3.50 | 466 |
| 4 | off | default | 0.16 s | 99 % | 1.52, 1.52 | 201 |
| 2 | on | default | 0.17 s | 98 % | 1.46, 1.43 | 192 |
| 2 | off | default | 0.20 s | 99 % | 1.41, 1.41 | 187 |
| 1 | – | default | 0.29 s | 99 % | 1.40, 1.40 | 186 |
| 1 | – | utility | 0.28 s | 98 % | 1.41, 1.40 | 186 |
| default (16) | off | background | 0.33 s | 72 % | 1.03, 0.97 | 132 |
| 4 | on | background | 0.51 s | 72 % | 6.36, 7.56 | 921 |
| 4 | off | background | 0.57 s | 53 % | 0.72, 0.71 | 94 |
| 2 | off | background | 1.04 s | 22 % | 0.53, 0.44 | 64 |
| **1** | – | **background** | **2.6 s** | **0 %** | **0.20, 0.18** | **25** |
| 1, four steps per wake-up | – | background | 2.3 s | 0 % | 0.22, 0.23 | 30 |

Shipped: one thread at background QoS. Only background QoS reaches the
efficiency cores (utility stays on the P-cores), and ORT's own pool threads
do not inherit it, so every second thread drags part of each step back to
the P-cores; with spin-waiting on that thread burns its wait there too.
Batching four steps per wake-up saves nothing. Both channels together then
cost 0.014 W during a meeting against 0.27 W for the session the stop used
(default threads, spin on), next to the live pipeline's ~0.6 W; each
channel keeps one efficiency core ~10 % busy.

Cross-checked at system level, because `ri_energy_nj` is the kernel's model:
steps back to back for 30 s, alternating with 30 s idle, four pairs,
`powermetrics --samplers cpu_power` over every window. One thread on a
P-core: 3.3 W net by powermetrics against 5.9 W by the kernel model, so the
model overstates P-core energy ~1.8×. One thread at background QoS: 0.08 W by
the kernel model, and 0.09 W net by powermetrics in the one pair whose idle
windows held still (idle drifted 0.3–1.5 W between windows, so the median of
all four, 0.02 W, is noise). The ranking holds at either scale: the shipped
arm is at least 4× cheaper per step than any P-core arm, 7× by the model.
The idle machine's drift is also why the duty-cycle arms above are not
measured by powermetrics: a 0.2 J step every 2 s is below it.

**The live meeting** (`steno start --replay` paced, the first 10 min of the
104-min meeting, `--diarization`, no counts, so both channels stream; once
with streaming and once with `_open_streams` stubbed out, everything else
identical; footprint sampled 4×/s plus the kernel's lifetime maximum):

| | live-phase peak (sampled / kernel max) | steady, minutes 4–9 | the stop |
|---|---|---|---|
| without streaming | 5.32 / 5.51 GB | 4.24 GB | 6.8 s |
| streaming, as shipped | 5.58 / 5.99 GB | 4.66 GB | 1.2 s |

Both runs: no load-shedding (0.0 s on either channel), the same 14 caption
lines arriving at the same replay time (a constant +0.34 s, the second run's
slower model load), 42 streamed steps (21 per channel, as 600 s allows), the
transcript identical entry for entry with the same four speakers and
voiceprints. The first streaming run loaded its session lazily on the
background thread at the first step: the sampler thread then stalled 1.75 s,
and a 5-ms ticker measured the session creation blocking every other Python
thread for up to 0.68 s on an efficiency core (0.12 s on the meeting's
thread, the GIL held throughout); the steps after it block other threads
under 10 ms (p99 1.3 ms). Shipped: the stream opens, and loads, at the start
of the meeting on the meeting's thread — the run in the table, whose only
sampler stall (0.76 s) falls inside the model loading at meeting start, where
the ASR load already holds the lock. Streaming costs the meeting ~0.4 GB
resident throughout and 0.3–0.5 GB of peak; on the 104-min stop it saves
~1 GB of peak instead, since the stop no longer loads the 1.8 GB offline
session. One 10-min excerpt — a lead for longer meetings, not a law. On the
HX 370 (5-min paced replay, peak RSS sampled 2×/s) streaming raised the peak
3.0 → 3.4 GB.

#### Background threads, by platform (2026-11-17)

`stenograf.cpu` states one rule — as many threads as the CPU has efficiency
cores, all confined to them, every physical core unconfined on a CPU with one
kind of core, never spin-waiting — and reaches it per OS. Measured on the M4
Max (above) and on a GPD with a Ryzen AI 9 HX 370 (4 Zen5 at 5.16 GHz + 8
Zen5c at 3.29 GHz, CachyOS, kernel 7.3, mains; package energy from amdgpu's
PPT sensor against an 8-s idle baseline before each arm, 21–23 steps per arm;
the arms on synthetic or stored audio cost the same per step).

**macOS runs one thread.** It has no CPU affinity; background QoS on the
calling thread is what puts a step on the E-cores, and ORT's pool threads do
not take it on even when the session is created on a background-QoS thread:
four threads ran 51 % on the P-cores created there, 53 % created on a normal
thread (0.82 and 0.64 J per step, 108 and 85 J per audio-hour) against 0 %
and 0.16 J (21 J) for one thread — so on macOS the rule's count is one.

**Linux pins.** Linux places a lone thread on a fast core even at
`SCHED_IDLE` (99 % of samples on Zen5): the one-thread stream cost ~15 J per
step there, ~2,000 J per audio-hour per channel, three times the offline
loop it replaced. Per step, by arm:

| arm | step | J / audio-hour / channel |
|---|---|---|
| 1 thread, unconfined (`SCHED_IDLE`) | 1.30 s | 1,980–2,000 |
| 1 thread, on the Zen5c | 1.88 s | 775–793 |
| offline loop: ORT default threads, spin on | 0.36 s | 643–654 |
| every core (24 threads), spin off | 0.35 s | 497 |
| 16 threads on the Zen5c, spin off | 0.39 s | 534 |
| **8 threads on the Zen5c (one per core), spin off** | **0.37–0.39 s** | **295–489** |
| 4 / 2 threads on the Zen5c, spin off | 0.54 / 0.96 s | 747 / 774 |

Race-to-idle wins here, the opposite of the M4 Max: the package draws ~5 W
above idle the moment any core runs, so finishing on eight compact cores
beats crawling on one. Pinning through ORT's `session.intra_op_thread_affinities`
(which Windows honours too) and through inheritance from a pinned creating
thread gave the same placement (88 % of running samples on the Zen5c; the
rest the sampler itself) and the same energy within the idle baseline's
noise. Through the shipped path (`background_plan` + `DiarizationStreamer`,
10 min of the 25.4-min meeting): 8 threads, 100 % of running samples on the
Zen5c, 0.39 s and 466–489 J per audio-hour per channel, probabilities and
turns bit-identical to the offline loop as released. Thread count and
pinning never changed a probability (1 vs 24 threads; every arm above).

What the detection reads: x86 kernels write a flat `cpu_capacity` of 1024
for every core even on this hybrid CPU, so a flat ranking defers to
`cpuinfo_max_freq`; an efficiency core is one at ≤ 85 % of the fastest core's
peak (`EFFICIENT_SHARE`). Two traps of this box that are not stenograf's:
numpy's OpenBLAS spread the log-mel matmul over all 24 threads and left them
spinning (4.5 CPU-s per step instead of 1.4, energy ~10 % higher) — set to
one thread at `import stenograf`; and CachyOS's `ananicy-cpp` moves every
Python process into `SCHED_IDLE`, capture and live captions included.

**The live captions follow the same rule at normal priority** (Linux and
Windows; macOS decodes on MLX, and its ONNX fallback only stops spinning).
onnx-asr had run Parakeet on ONNX Runtime's defaults — spinning, 12 threads on
any core. Measured on the same GPD: 20 min of the 2026-11-16 15:30 meeting,
both channels, its 55 live windows decoded back to back with a 1-s pause after
each (so pool threads go quiet as between real windows), net of idle, two
runs per arm within 2 %:

| arm | J / audio-minute | decode speed | CPU-s |
|---|---|---|---|
| ONNX Runtime defaults (spin on, 12 threads) | 62 | 15× realtime | 1,925 |
| spin off | 26 | 15× | 1,090 |
| spin off, 4 threads | 22 | 13× | 352 |
| **spin off, 8 threads on the Zen5c** | **20** | **15×** (slowest window 14×) | 558 |

Every arm produced the same 2,260 words. Through the shipped backend (10 min
of the same meeting): 100 % of running samples on the Zen5c, 18 J per
audio-minute against 68 for the released defaults, the same 898 words.
Fewer compact cores, 10 min (a file copy ran alongside): 4 threads 9×
realtime (slowest window 9×), 2 threads 6× (5×), 1 thread 4× (3×). Eight compact cores decode as fast
as twelve on any core while four fast ones are slower, so the encoder still
scales to eight threads; beyond that the per-token Python loop sets the pace
(inferred). Not measured: with a glossary (the biased loop), with the
diarization stream sharing the Zen5c (it yields while the captions are
behind), and on a CPU with few or weak efficiency cores — nothing guards the
rule there (Daniel's call, 2026-11-17).

**Windows** reads the efficiency classes from `GetSystemCpuSetInformation`
and pins the same way. Measured on the same GPD under Windows 11 (26200,
Balanced, mains), the 10-min excerpt from minute 5 of the same meeting,
energy from the Energy Meter's RAPL package counter (picowatt-hours) net of
idle, placement from per-processor busy time. The detection reads class 0
for the 8 Zen5c (logical 8–23) and 1 for the 4 Zen5, so the plan is the
Linux one, and pinned work stayed 98–100 % on the Zen5c. The live windows'
898 words were identical in every arm; the stream was bit-identical to the
offline loop.

A windowless process — anything started over SSH — runs under Windows'
execution-speed throttling, which slowed every arm *and* cost energy; the
arms were therefore run throttled and with the process opted out
(`SetProcessInformation(ProcessPowerThrottling)`), the state of a foreground
app, on mains and on battery (where Windows switches to its "best power
efficiency" overlay). Runs whose idle baseline had not settled (Defender and
the indexer after a long-unused boot, the minutes after unplugging) are left
out; each cell lists every remaining run:

| | mains, throttled | mains, not throttled | battery, throttled | battery, not throttled |
|---|---|---|---|---|
| live captions, ONNX Runtime defaults (J / audio-min) | 59.0, 59.4 — 20× | 48.0, 48.0 — 28–30× | 37.0 — 12× | — |
| live captions, spin off only | 51.6, 51.7, 54.8 — 18× | — | — | — |
| **live captions, shipped rule** | **53.8, 54.4, 55.7 — 14×** | **40.1, 49.5 — 26×** | **29.1 — 8×** | **33.7 — 7×** |
| diarization stream, released defaults (J / audio-h / ch) | 1,140 — 0.41 s/step | 917, 920 — 0.31 s | — | 862 — 0.42 s |
| **diarization stream, shipped rule** | **1,194, 1,490 — 0.55 s** | **927, 933 — 0.35 s** | — | **848 — 1.21 s** |

So on Windows the rule gains little: about −8 % for the captions on mains
throttled, about −20 % on battery (one run per arm), and nothing
established unthrottled on mains (one run each side of the default); the
stream is equal wherever the baseline was quiet. Windows already places much
sustained work on the Zen5c, and ONNX Runtime's Windows pool spins far less
(255–365 CPU-s by default against Linux's 995 for the same 10 min). The rule
costs speed on battery — the captions' slowest window at 5–6× realtime
against the default's 11–15×, still far ahead of the meeting — and keeps
placement independent of power mode and throttling, so it stays uniform.
Battery runs cost less than mains runs for the same work at lower clocks,
and Windows decoded the Zen5c windows in about half Linux's CPU-seconds:
Windows on mains clocks higher (inferred, clocks not logged). The same work
reads 2–3× the energy of the Linux runs, but the sensors differ (RAPL
package here, amdgpu PPT there), so absolute numbers do not compare across
the two; the battery fuel gauge reports in ~420 mWh steps, too coarse to
arbitrate at this run length. Untested on Windows: another power mode, and
the real app's window states.

## Speech detector: Silero v6 (2026-11-19)

**Why it changed.** A phone on speakerphone next to the Mac reaches only the
mic channel, quietly and band-limited, and Silero v5 at threshold 0.5 called
most of it silence. Reference per recording: Parakeet over the whole mic
channel with no gate, cut at quiet points; score: share of its words a
re-transcription reproduces in order. On four of seven calls v5 kept 29 %,
35 %, 62 % and 69 %. Lowering the threshold or levelling the signal before
the detector helped unevenly; a better detector did not. Over ten
recordings (the seven calls plus three ordinary ones, 18 k reference words),
each detector feeding the shipped decode path:

| detector (threshold 0.5) | worst four calls | the other six |
|---|---|---|
| Silero v5 (before) | 29 / 35 / 62 / 69 % | 86–97 % |
| **Silero v6** | **90 / 83 / 89 / 85 %** | 87–97 %, none lower |
| TEN VAD | 84 / 72 / 81 / 81 % | 85–97 % |
| Nemotron activity (all speakers merged) | 93 / 83 / 92 / 93 % | 90–98 % |

Nemotron was not chosen: it already runs on multi-speaker meetings, but
using its activity as the gate would make every one-speaker recording pay
for it (~9× v5's CPU, +690 MB RSS on 13.9 min) and delay captions by its
27.2 s chunk plus 3.2 s look-ahead (simulated on the same call: median 19 →
34 s), for a few points on two calls.

**End-to-end, both arms through `steno transcribe --no-notes
--no-diarization`** (control = the same commit with v5, sixteen recordings):

- Calls and ordinary recordings, the ten above: **73.7 % → 90.2 %** of the
  reference words, 639 → 921 words beyond them.
- The six window-study meetings (`retranscribe_compare.py --old-dir`,
  7.4 h): **61 153 → 61 706 words (+553)**, 425 added regions (383 of at
  most three words), 196 removed, 839 changed. Whisper refereed the changed
  regions 113 : 111 (99 ties), so those are churn, not gain. The new
  detector moves every window bound, and the greedy decode's tail instability
  (window-length study) re-rolls with it. Passages of five or more words
  that shrink to at most half: 15 lost (144 words, a few of them removed
  hallucinations such as ten "nein"s), 39 gained (277 words); one German
  passage on 0714-160212 decodes as English. The threshold-0.4 failure mode
  does not reappear: windows ending inside speech 15 → 18, runs
  force-closed at `max_speech` 47 → 58, over both channels of the six
  meetings.
- Nemotron as the gate, the same arms (merged turns, then Silero's 0.5 s /
  0.25 s / 28 s rules; batch only, the live path has no adapter): the ten
  recordings reach **92.6 %** (1144 extra words), the six meetings **+834
  words**, 16 passages lost (123 words) and 49 gained (354), referee 147 :
  132 (131 ties), the 0714-160212 flip absent. Batch wall time with
  diarization off: +21 % over the v6 arm (sixteen recordings, 728 s vs
  600 s).
- `live.py`: 0.0 % WER against finalize on de-1 and en-1, no monotonicity
  violations; the reuse guarantee holds. The stream-vs-batch and
  byte-identical-slice tests pass on v6.

Cost: 2.43 s CPU vs 2.35 s on 13.9 min of audio, +92 MB vs +87 MB RSS.

## Echo cancellation

Layer-0 signal scoring of the AEC path. A meeting run with `--aec-dump DIR`
writes the clock-aligned `mic.wav`/`lpb.wav`/`enh.wav` triple (near end as
captured, far-end reference, near end as the ASR receives it); score it with:

```sh
uv run --group eval eval/aec_score.py DIR --scenario st   # far-end single-talk
uv run --group eval eval/aec_score.py DIR --scenario dt   # double-talk
```

Reports ERLE + residual level (energy over 10 ms frames during far-end
activity) and AECMOS (`speechmos`, the AEC-Challenge metric) — `echo_mos` for
"is the echo gone", `deg_mos` for "did we damage the local speaker". The
`--no-aec --aec-dump DIR` combination records the uncancelled baseline.

`aec_rig.py` runs a whole scenario on real hardware — plays a speech WAV out
the speakers while the real pipeline captures — and scores both layers (signal
metrics + leaked `Local-N` lines in the transcript):

```sh
uv run --group eval eval/aec_rig.py far-only --seconds 60   # pass = 0 leaked lines
uv run --group eval eval/aec_rig.py far-only --no-aec       # uncancelled baseline
uv run --group eval eval/aec_rig.py double-talk             # talk over it yourself
```

Runs land in `eval/out/aec/<scenario>-<stamp>/` with the meeting output, the
dump triple, and `rig.json`. Keep volume, lid angle, and source clip fixed
across runs you compare. Measured 2026-07-10 (MacBook speakers, volume 63):
AEC on → 37.6 dB ERLE, −65 dBFS residual, AECMOS echo 4.73, **0 leaked lines
before any text backstop**; AEC off → −27 dBFS raw echo, AECMOS echo 1.49.
(`aec_rig.py` itself is macOS-only — it drives `afplay`.)

### Before any of it: is there an echo path?

```sh
steno start --local 1 --remote 1 --max-seconds 60 --aec-dump probe \
    --out probe-meeting              # …with speech over the speakers…
uv run python eval/aec_echo_present.py probe     # PASS = the mic hears them
```

ERLE is undefined when no echo reaches the microphone, and `aec_score.py` cannot
tell that apart from a canceller that failed: both read as ~0 dB. A 33-minute
Windows run was spent learning this on 2026-07-26. The mic must be ≥6 dB louder
while the speakers play than while they rest — **output volume is the first
knob** (40 % on a laptop chassis gave no echo path at all; 90 % gave 22 dB) and a
capture endpoint's driver audio-enhancements the second.

### Windows, measured 2026-07-26 (GPD notebook, Realtek, volume 90)

80 s far-only, from the notebook's built-in speakers into its own mic:
**2.6 dB ERLE, −29.6 dBFS residual, 2 leaked `Local-1` lines** — against
macOS's 37.6 dB above. The cause was not the canceller but the timestamps it
pairs channels by: WASAPI's loopback tap is a longer transport than the mic, both
are stamped on arrival, so the reference was labelled ~60 ms *after* its own
echo, and AEC3 searches its far-end history backwards only. With
`capture.windows.FAR_END_LAG_S` compensating it: **13.7 dB ERLE, −42.1 dBFS,
0 leaked lines** on a re-run of the same script.

**Both numbers are now historical.** The constant they measured was deleted in
2026-08 along with the in-process WASAPI capture that needed it: the Rust
capture helper stamps both taps from `pu64QPCPosition`, one device clock, so
there is nothing left to declare. Re-measuring on the helper is the open gate —
these two rows are what it has to beat.

`eval/aec_alignment.py` is that whole diagnosis in one command — it measures the
lag, replays the dump through the **real** `EchoCanceller` at a sweep of
corrections, and prints the `far_end_lag_s` to ship:

```sh
uv run python eval/aec_alignment.py probe            # measure + sweep
uv run python eval/aec_alignment.py probe --no-sweep # measure only
```

It compares what it measures against what the capturing provider already
declares, so a fixed machine reads PASS. No transport is arrival-stamped any
more: the Linux backend joined the Rust helper in 2026-08 (`parec` left with
it), so all three platforms anchor both channels to one OS clock, and this
instrument's job is to *verify* that a dump needs no correction rather than to
recommend one. Every AEC ERLE number on record is still from macOS.

Two notes for whoever measures this next:

- The dump records frames as the provider stamped them, so `lpb.wav` still
  trails `mic.wav` *after* the fix — the correction lives in the canceller, not in
  the timeline the dump is written on. Judge the fix by ERLE and by leaked lines,
  not by re-measuring the dump's own alignment.
- **The lag is re-rolled at every meeting start**, since each channel anchors on
  its own first frame and the two pump threads open independently: the two runs
  above measured 60 ms and 10–25 ms, minutes apart on one machine. So one dump is
  one sample — fit a constant to it and the next meeting can be dead. It is also
  why per-quarter lag figures come with their correlation and no drift verdict:
  over 20 s the estimate is noisy enough to invent a 250 ms shift, and a quarter
  with little far-end activity correlates silence against silence.
- 13.7 dB is a working canceller on a small chassis, not parity with the Mac's
  37.6 dB. Whether the remaining gap is the speaker's nonlinearity, the driver's
  own processing, or a lag constant that could be tighter is unmeasured.

## Contextual-biasing evaluation (Phase 5)

Decode-time biasing (`stenograf.asr.biasing`) ships with a tree verified against
NeMo's golden vectors — but its *effect* rested on one TTS clip and three meeting
WAVs, which is enough to prove the mechanism fires and not enough to set `[asr]
boost` or to defend our two deliberate divergences from NeMo (`unk_score=1.0`, and
the German compound-tail tokenization). This harness replaces the anecdotes with
numbers, and needs **zero hand labeling**: every reference and every word list is
derived from corpora that already ship them.

```sh
# 0. Fetch/derive the benchmarks (is21 English lists; German built from MLS)
uv run --group eval eval/bias_data.py --fetch all --sizes 100 500 1000 2000

# 1. Correctness gate — the only language with published numbers to check against
uv run --group eval eval/bias.py --tier english --n 100

# 2. The benchmark that sets the shipped defaults (ablates boost/unk/compound-tail)
uv run --group eval eval/bias.py --tier german --sweep

# 3. False insertions with ground truth: bias with words known to be ABSENT,
#    so any change at all is a false insertion. Runs on real meeting audio.
uv run --group eval eval/bias.py --tier distractor --wav eval/audio/*.wav

# 3b. BOTH glossary layers — decode-time biasing vs post-hoc fuzzy correction vs
#     the stack. Works on any tier and re-decodes nothing (post-correction is a
#     pure text transform over the arms' cached hypotheses).
uv run --group eval eval/bias.py --tier german --post 0.88 0.92 0.95

# 4. Reachability probes (synthetic; a diagnostic, never a quality metric)
uv run --group eval eval/bias_tts.py && uv run --group eval eval/bias.py --tier tts

# 5. Head-to-head vs TypeWhisper's engine (FluidAudio), English only — see below
uv run --group eval eval/bias_fluid.py --cli /path/to/.build/release/fluidaudiocli
```

## The head-to-head: FluidAudio (TypeWhisper's Parakeet engine)

`bias_fluid.py` runs the *same acoustic model we do* (Parakeet TDT 0.6b v3) through
FluidAudio's context-biasing path — the engine behind TypeWhisper's Parakeet plugin —
on the same pinned 500 utterances, the same per-utterance 100-term lists, scored by
the same `bias_score`. Held constant: model, audio, lists, scorer. The only variable is
the **mechanism**: our boosting tree over token logits *inside* the greedy TDT loop, vs
their second CTC model rescoring the *finished* transcript. Each system is reported
against **its own** unbiased baseline (their encoder is CoreML int8, ours MLX fp32 — the
absolute WERs are not comparable; the deltas are).

**English only, and that is their limit, not our choice.** Their spotter
(`parakeet-ctc-110m`) has a 1024-token vocabulary with **zero** non-ASCII tokens, so
German terms tokenize into `<unk>` holes and are silently kept. TypeWhisper pairs it
with the multilingual transcriber with no language check. Our German tier has no
opponent — that is a capability they lack, not a number they lose.

**Result (2026-07-13).** As TypeWhisper ships it, the vocabulary posts a spectacular
B-WER −75.3 % and **destroys the transcript doing it**: U-WER **+305.6 %**, **375 false
insertions**, 287 of 500 utterances altered, with rewrites like `glowing`→`unloving`
and `pound`→`compound` — correct common words snapped onto *distractors*, terms the
benchmark included precisely because they are absent. Not a tuning accident: their own
defaults give 374, their documented `cbw 3.0` gives 362.

**But the mechanism is sound and the configuration is the bug.** Forced to
`minSimilarity 0.85`, the same engine reaches **B-WER −32.2 %, U-WER +2.4 %, 8 false
insertions** — parity with our −34.9 % / +0.0 % / 2. On accuracy alone this benchmark
does not separate in-loop boosting from post-decode rescoring. What separates them is
the rest of the bill: one pass vs two models, streaming vs batch-only, and German.

Two defaults do the damage, both worst where real users live:

| | |
|---|---|
| `rescorerConfig(forVocabSize:)` | minSimilarity **0.60** above 100 terms, **0.55** at 11–100, **0.50** at ≤10 — *smaller glossary, looser matching*. A real meeting glossary is 10–30 terms. |
| **spotter rescue** (on by default) | wrecks small lists: an oracle list of only genuinely-spoken words yields **762** false insertions and U-WER +1407 %; `--vocab-disable-spotter-rescue` alone drops it to 104. minSimilarity does not gate it — 0.85 leaves the collapse intact. |

**Read it fairly.** is21's "rare words" are ordinary English words (`frail`, `idly`,
`holiness`), not the entities and jargon FluidAudio is designed for; their published
99.3 % precision on earnings calls is plausibly true in that domain. The bounded claim:
*on the standard biasing benchmark, at its standard list size, the configuration
TypeWhisper ships is destructive — and the mechanism underneath it is not.*

## The confidence gate we did NOT build (`bias_confidence.py`)

The idea worth stealing from FluidAudio is the principle, not the code: **a post-hoc
correction must answer to the audio**, which is exactly what our fuzzy layer cannot do
and why over-correction forced the threshold to 0.95. The cheap version of that idea:
our TDT already emits per-token confidence from the *unbiased* distribution, so gate
corrections on it — refuse to rewrite a word the model was sure about. No second model,
no extra compute, works in every language.

`bias_confidence.py` tested the claim before anyone implemented it, and **killed it**:

| | n | median conf | p10 |
|---|---|---|---|
| false insertions | 53 | **0.999** | 0.938 |
| true fixes | 165 | **0.951** | 0.858 |

Directionally right, practically useless. The best gate (c = 0.95) blocks 45 of 53 false
insertions and destroys **84 of 165 true fixes**; looser points are worse. `der` → `deri`
(false insertion) scored 1.000 and `finde` → `find` (real fix) scored 0.997 — a greedy
RNN-T's entropy-normalized confidence saturates at ~1.0 for nearly everything, so it
cannot carry a gate. One decode pass, no refactor, question closed.

What survives is the *other* question. The gate asked "how sure was the model about the
word it wrote?" (always: certain). The discriminative one is "how much does the model
dislike the word we want to put there?" — scoring the **candidate** over the span,
against the tokens actually emitted. That is FluidAudio's idea done with the right model
(our 600M TDT, not a weak English-only 110M CTC), and it needs forced alignment through
the joint. Unproven, unbuilt; `bias_confidence.py` is what would price it.

**Metrics** (`bias_score.py`, pure — pinned by `tests/test_eval_bias.py`): B-WER
(WER over reference words in the biasing list — must fall), U-WER (every other word
— must **not** rise; over-boosting is visible here and nowhere else), entity
recall/precision/F, false insertions, and surface damage (`Ada` → `ADA`, which every
WER-shaped metric normalizes away before it can see it). Entity numbers are reported
strict *and* prefix-tolerant, because in German a term survives inside an inflected
or compounded word (`Europa` in `Europas`).

The scorer is a faithful port of is21's own alignment, so their **44 published
hypothesis/result file pairs are a free correctness oracle** — `tests/test_eval_bias.py`
reproduces every one of them to the digit (it skips until `--fetch is21` has run).

**What `--post` found (2026-07-13), and why the threshold moved to 0.95.** We ship
two glossary layers and had only ever measured one. Scored on the same terms, the
post-correction layer at its old 0.82 default looked spectacular on B-WER (−52.8 %
German, −55.5 % English — *twice* what decode-time biasing gets) and was in fact
damaging the transcript: U-WER **+9.9 % German / +86.3 % English**, 85–86 false
insertions against biasing's 3. The shipped stack (both layers, 0.82) came in at
U-WER +6.5 % and 84 false insertions — **worse than the `boost = 2.0` config we had
already rejected** (+6.7 %, 27 insertions). We had a bar; we had just never pointed
it at the second layer.

The cause is structural and worth remembering when adding any correction layer:
fuzzy text matching answers to no acoustics, so nothing prevents it from snapping a
correct common word onto a glossary term that merely looks like it, and **B-WER
alone cannot see the difference** — only U-WER and false insertions can. At 0.95 the
layer becomes a free win (B-WER −30.3 % German / −36.3 % English, i.e. *better* than
biasing alone, with U-WER flat and false insertions at biasing's own level), and on
real meeting audio with a realistic 30-term glossary it inserts nothing that biasing
did not already insert. One caveat on the tables: damage scales with list length, so
the N=100 lists overstate the risk for the 10–30-term glossaries real meetings use.

**The surface-damage column paid for itself here.** It was the only metric that could
see a second bug, and it sat at ~1190 for the post arms *at every threshold* — a
quantity that ignores the knob gating it is not being produced by that knob. Cause:
these word lists are 100 % lowercase, and `glossary.py` imposed a term's spelling
verbatim, so it dutifully de-capitalized German nouns. Fixed by teaching it what
`asr.biasing.surface_forms` already knew — capitals in a term are a deliberate
spelling and win; an all-lowercase term asserts nothing about case and must not
overwrite the model's. Re-running the tier drove the column **1190 → 2 with B-WER,
U-WER and false insertions bit-identical** (they normalize case, so they were blind
to the whole thing). Every WER-shaped metric would have shipped this.

Landmines, each verified and each worth a day: Parakeet emits punctuation and case
while LibriSpeech/MLS references do not (normalize both sides, and case-match the
boost phrases); every German noun is capitalized, so rare-by-frequency is the only
usable definition of a rare word; Common Voice is a dead stub on HuggingFace since
Mozilla moved it behind their Data Collective; AMI is uppercase, unpunctuated and
proper-noun-*sparse*, i.e. near-worthless for biasing despite being the closest
thing to meeting audio.

Sweeps run on a **pinned 500-utterance subsample** (the full grid is 5–10 h of
decoding); only the winning config is re-run over the full test set, and every table
states which it was. Hypotheses are cached per config, so an interrupted sweep only
costs the configs it had not reached.

## Stored-audio codec (2026-08-06)

`--record-audio` writes **Ogg Opus at 32 kbps per channel** (nominal
≈14 MB/h/channel vs 115 MB/h WAV, 59 MB/h FLAC; the real default — joint
stereo at 64k nominal — measured 34 MB/h on a 10-min mic+system pair, libopus
VBR overshooting to 76 kbps, where two mono encodes of the same pair total
19 MB/h — 2026-08-06 adversarial review). Chosen from a literature survey plus a
listening ladder Daniel judged by ear (`eval/out/opus-ladder/`, gitignored;
rebuild = ffmpeg `-c:a libopus` over any manifest WAV). The load-bearing
measurements, all on clean/curated corpora — none on post-AEC meeting audio:

- **ASR**: Opus WER penalty on LibriSpeech (SpeechBrain conformer) is
  +0.06 pt at 12 kbps, +1.07 at 6 (NoLACE, arXiv:2309.14521 Table 1).
  GigaSpeech ships this exact config (Opus 32 kbps, 16 kHz mono) and measured
  train-on-wav/eval-on-opus at +0.1–0.2 pt over 10 k hours (arXiv:2106.06909
  Table 3). Codec artifacts are also in-distribution for Parakeet v3: ~660 k
  of its ~670 k training hours are Granary = YouTube-derived Opus/AAC, and
  Common Voice is 48 kbps MP3 — an argument from provenance, not a vendor
  claim.
- **Speaker embeddings bind tighter than ASR**: ECAPA/CAM++/ERes2Net EER on
  VoxCeleb1-O is ~clean at 24 kbps (+0.24 pt), cliffs between 12 and 6 kbps
  (1.44 → 6.38 %), worse on hard pairs (arXiv:2509.02771). So the floor is
  the embedding one, not the WER one. Never below 12 kbps; never resample to
  8 kHz (costs more than any codec — Ferro Filho, Interspeech 2025).
- **libopus band-limits to 6 kHz below its ~10 kbps mode switch** (measured
  2026-08-06 on our bundled ffmpeg by band-energy comparison; the "narrowband
  at 4 kHz" claim in arXiv:2509.02771 did not reproduce).
- **Crash safety** (measured 2026-08-06, corrected same day by adversarial
  review): ffmpeg buffers output in 256 KB blocks, so WITHOUT
  `-flush_packets 1` a SIGKILL loses everything since the last block boundary
  — 5/10/20/30 s recordings all died at 0 bytes, undecodable. WITH the flag
  (shipped) the same kill leaves a readable file missing 2–5 s, matching the
  WAV sink's patched-header contract. Encode runs ~150× realtime (49-min
  channel in 20 s), so live encoding costs <1 % CPU.
- Above ~96 kbps lossy is pointless: Opus 128 kbps measured *larger* than
  FLAC on the ladder clips.

**Unmeasured, deliberately left open**: (a) transcript divergence on our own
corpus — `retranscribe_compare.py --old-dir` with an uncompressed-decode arm
vs an Opus-ladder arm would resolve it far below 0.1 WER pt (noise floor
−27 words/10.5 k); (b) DER vs bitrate for our diarizer via `der.py` — no
published curve exists for any off-the-shelf pipeline; (c) codec × noise
interaction on far-field audio (the one crossed study, SVeritas
arXiv:2509.17091, suggests multiplicative, but its codec tables are defective).

## The ONNX ASR loads from its own model folder (2026-11-15)

The release dry run's Windows smoke failed at `steno setup --models-only`:
onnxruntime refused `encoder-model.onnx.data` with "External data path
validation failed". Two upstream changes meet there. huggingface_hub 1.32
(2026-09) moved large Xet-stored files into a cache-wide store sharded by hash
(`hub/blobs/<2 hex>/<hash>`, a second symlink hop behind the repo's own
`blobs/`), so the graph and its weights resolve into different directories —
measured on macOS: `blobs/9a/…` and `blobs/79/…`, the same `79` the Windows
error names. Every onnxruntime from 1.24.2 through 1.31.1 rejects external
data outside the graph's directory; 1.32.0 (2026-11-13) accepts it. DirectML,
the Windows flavor, ends at 1.24.4. Measured on macOS, CPU provider, onnx-asr
0.12.0:

| model source | onnxruntime 1.24.4 |
|---|---|
| hub cache written by huggingface_hub 2.4.0 | fails, the Windows error |
| hub cache written by 0.36.2 | loads |
| a directory of plain files | loads |
| the old layout with only the `.data` blob moved into a `blobs/79/` | fails |

1.25.1 through 1.31.1 fail the shared layout too; 1.32.0 loads it. Fix:
`parakeet_onnx.model_dir()` downloads the five files the fp32 export needs
(`snapshot_download(local_dir=…)`, pinned revision) into stenograf's cache as
plain files, marks the folder complete only afterwards, and hands onnx-asr the
folder. With huggingface_hub 2.4.0 and onnxruntime 1.24.4 the backend then
loads from a folder with no symlinks and transcribes the TTS sentinel clip
correctly. The Nemotron assets were never exposed: `assets.py` downloads them
into one directory of plain files.

**Nemotron moved to the fp32 export the same day.** On onnxruntime 1.32 the
fp16 export ran twice as slow: 34–35 vs 18 s per audio-hour on the 105-minute
system channel of a stored meeting (M4 Max, interleaved runs, nothing else
running), at unchanged accuracy on AMI's held-out set (attribution identical
on mic and loop, 97.4 vs 97.3 % on duo with the count cap). The step is between
1.28 (16–17 s) and 1.29 (33 s); 1.31.1 is 34. Not threads — both use all 8
cores, 1.29 spends 43.6 vs 20.7 s of CPU on 10 minutes of audio. An ORT
profile puts it in MatMul: 1.28 runs all 4 560 MatMul calls in fp32 (the fp16
graph converted around them), 1.29 runs 4 512 of them in fp16 natively, 3.79
vs 1.45 s. The fp32 export runs 15 s per audio-hour on 1.32 (18 on 1.27), its
peak resident memory 1.78 vs 1.73 GB, and its held-out scores match fp16's
within 0.1 pt (the reference fp32 row above). Cost: a 398 MB download instead
of 199. Whether x86 CPUs show the same fp16 step is not measured.

## Side quests

- ~~Canary-1B-v2 runtime~~ — resolved, see above: no accelerated runtime with
  word timestamps exists; NeMo-on-MPS reference backend wired up instead.
- **speakrs diarization sanity check** on the same audio (not wired up yet).

## Metrics

`transcribe.py` records per segment: wall time, speed (×RT), model load time,
peak RSS, MLX peak memory, detected language. `score.py` adds WER/CER (jiwer)
after normalization (lowercase, punctuation stripped, umlauts kept).
