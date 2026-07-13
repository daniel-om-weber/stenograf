# Phase 0 — model evaluation harness

**Result (2026-07-06): Parakeet-TDT-0.6B-v3 is the default for finalize + live.**
Blind adjudication of 161 disagreement sites: Parakeet tied Whisper 42:42,
Voxtral lost to both, Canary lost every pairing ~1:2. Details in PLAN.md §2;
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
English), including one in-room far-field sample, per PLAN.md Phase 0.

## Diarization scoring (Phase 3, Task 0d)

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

## Echo cancellation (PLAN-AEC.md)

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

## Side quests

- ~~Canary-1B-v2 runtime~~ — resolved, see above: no accelerated runtime with
  word timestamps exists; NeMo-on-MPS reference backend wired up instead.
- **speakrs diarization sanity check** on the same audio (not wired up yet).

## Metrics

`transcribe.py` records per segment: wall time, speed (×RT), model load time,
peak RSS, MLX peak memory, detected language. `score.py` adds WER/CER (jiwer)
after normalization (lowercase, punctuation stripped, umlauts kept).
