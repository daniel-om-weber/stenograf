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
(`sherpa.py::_pool_workers`), every ORT session at ONE intra-op thread, one
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

## Side quests

- ~~Canary-1B-v2 runtime~~ — resolved, see above: no accelerated runtime with
  word timestamps exists; NeMo-on-MPS reference backend wired up instead.
- **speakrs diarization sanity check** on the same audio (not wired up yet).

## Metrics

`transcribe.py` records per segment: wall time, speed (×RT), model load time,
peak RSS, MLX peak memory, detected language. `score.py` adds WER/CER (jiwer)
after normalization (lowercase, punctuation stripped, umlauts kept).
