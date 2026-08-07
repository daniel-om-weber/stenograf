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
calibration caveats in `PLAN-DIARIZATION.md` step 0.

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
`PLAN-DIARIZATION.md` step 1.2 on these numbers.

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
