# A more accurate ASR model — the recurring question, and the gate it must pass

This file exists because the question returns every few months with a new name
on it: *the leaderboard has a new leader — should stenograf run it?* Twice the
answer was measured and negative (Voxtral Small 24B and Canary-1B-v2, Phase 0,
July 2026). The third arrival is **Cohere Transcribe 03-2026**, raised in the
2026-07-26 session and **not measured**.

Unlike `PLAN-WINDOWS.md`, almost nothing here is open work. It is **one
experiment — C1 — that must run before any line of C2 is worth reading**, plus
the record of why the two previous challengers lost, so the next session does
not re-derive it. If C1 loses, close this file the way `PLAN-LINUX.md` closed:
delete it, and let `git log --follow -p` hold the evidence.

---

## The candidate (facts as of 2026-07-26, none verified against our audio)

**`CohereLabs/cohere-transcribe-03-2026`** — 2 B params, Apache-2.0, released
2026-03-26. Fast-Conformer encoder + deliberately lightweight transformer
decoder (>90 % of parameters in the encoder, to hold down autoregressive cost).

- **5.42 % average WER, #1 on the HuggingFace Open ASR Leaderboard** at release,
  ahead of Whisper large-v3 (7.44 %) and of ElevenLabs Scribe v2 and
  Qwen3-ASR-1.7B. Parakeet-TDT-0.6B-v3 sits at ~6.32 % on the same English
  board. **That board is read speech; ours is meetings.** See the graveyard
  below for what that distinction has already cost.
- 14 languages: en, fr, de, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar. (Parakeet
  v3 covers 25.)
- Runtimes: transformers, vLLM, **MLX-Audio** (the one that matters here — the
  Mac is where the parity harness runs), a Rust implementation, transformers.js
  + WebGPU. **No documented ONNX export**, which is why C1 is macOS-only and why
  a positive result still would not reach the cross-platform default.
- Native ~35 s `max_audio_clip_s`; the feature extractor auto-splits longer
  audio and returns `audio_chunk_index` for reassembly.
- A separate `cohere-transcribe-arabic-07-2026` variant exists. Irrelevant
  unless scope changes.

Its own model card's Limitations, all four of which bear on us:

1. **No timestamps** — neither word-level nor segment-level. No diarization.
2. **No automatic language detection**; the language must be specified up front.
3. **Poor code-switching** — and our benchmark audio is German and English.
4. **Hallucinates from background noise**; the card recommends a VAD or noise
   gate in front. We already have one (`vad.py`), so this is the one limitation
   that costs us nothing.

---

## Why it cannot be a backend, only a second pass

`asr/base.py:13` is the contract: *"Word-level timestamps are mandatory: speaker
assignment intersects them with diarization turns."* Limitation 1 fails it
outright. Concretely, three things break:

- **Speaker attribution.** `pipeline.py:94` flattens segments to words and
  `pipeline.py:146` assigns speakers by word midpoint `(w.start + w.end) / 2`.
  The diarized transcript *is* the product.
- **Live captions.** `live.py:31` states it: *"the live pass needs word
  timestamps, so in practice it runs Parakeet."* LocalAgreement-2 commits a word
  once two consecutive window decodes agree on it. The model is offline-batch
  anyway, with 35 s chunks against our short trailing re-decode window.
- **Language.** Parakeet v3 is multilingual and ignores the argument
  (`parakeet_onnx.py:134`); LID runs afterwards over finalized text (`lid.py`).
  Limitation 2 inverts that. `base.py:75` anticipates the case — *"A
  language-requiring backend must handle `None` itself (detect once, then
  lock)"* — but it is work, not a registration.

This is the same wall Canary hit (`base.py:6`), and worse: Canary *has*
timestamps in NeMo and lost on runtime. This model has none by design.

So the only shape worth considering is **a finalize-only second pass**: Parakeet
supplies timed words, the challenger supplies text, and something merges them.
That is C2, and C1 gates it.

---

## Already measured — do not re-derive

From the PLAN.md architecture section pruned 2026-07-25 (`git show 5b4bdbb --
PLAN.md`). Paper German WER, and then what happened on real meeting audio:

| Model | German WER (paper) | Adjudication result on our audio |
|---|---|---|
| Voxtral Small 24B | **3.01 %** | **lost 32:38 to Parakeet**, 22:28 to Whisper |
| Canary-1B-v2 | 4.10 % | **lost every pairing ~1:2** — weakest arm |
| Qwen3-ASR-1.7B | 4.12 % | not run |
| Parakeet-TDT-0.6B-v3 | 4.20 % | **default; tied Whisper large-v3 exactly, 42:42** |
| Whisper large-v3 | 4.26 % | the pivot and fallback |

*Phase 0, July 2026: blind adjudication of 161 model-disagreement sites, de+en,
variants shuffled and unlabeled.* Verdict on record: **"Voxtral's read-speech
advantage did not transfer to meetings — demoted from 'opt-in max accuracy' to
not-worth-shipping pending new evidence."** Parakeet won the default on being
~10× faster and 5× smaller at identical quality.

**Read the table by column, not by row.** Voxtral led Parakeet by 1.19 WER
points on paper — a *wider* margin than Cohere Transcribe's ~0.9 — and lost the
head-to-head anyway. Every model that has beaten Parakeet on a benchmark has
lost to it on our audio. That is the prior C1 has to overturn, and it is the
whole reason this file refuses to let C2 be designed first.

Two more results worth not repeating:

- **`c837c69` — post-correction was rewriting words that were right.** The
  glossary's fuzzy-correction layer posted a B-WER drop twice biasing's and paid
  for it with U-WER +9.9 % German / +86.3 % English and 85 false insertions
  against biasing's 3. The commit's own generalization: *"it generalizes to any
  correction layer we add later … B-WER alone cannot see the difference. Only
  U-WER and false insertions can."* A second ASR model does re-read the audio,
  so it is **not** the identical failure mode (that layer answered to no
  acoustics). The measurement discipline transfers regardless — see C2's bar.
- **`2fdc146` — cut-overlap decoding reverted** (per-site coin flip + language
  flips), and **`a48eb2c` — VAD edge retunes reverted**. Both were plausible
  accuracy wins that measured as noise or worse. The pattern is consistent
  enough to be the house prior.

---

## C1 — the gate: adjudicate before designing anything

**Do not write merge code. Add an eval arm and run the existing harness.** The
harness was built for exactly this and already absorbs everything awkward about
the model:

- `eval/backends.py:111` — `VoxtralMLX` is already a **timestamp-free arm**
  returning `"segments": []`, and it already **demands an explicit language**
  (`:143`, raises when `None`). Both of Cohere Transcribe's blocking limitations
  have a working precedent in that class.
- `eval/adjudicate.py:35` — `PIVOT = "whisper"`, chosen as *"the only backend
  with reliable word-level timestamps in its output"*, so arms without
  timestamps compete on text alone.

The work:

1. A `CohereTranscribe(Backend)` in `eval/backends.py`, modeled line-for-line on
   `VoxtralMLX`: MLX-Audio runtime, raise on `language is None`, feed ~30 s
   windows cut at silence via `common.split_at_silences` (mirrors the production
   VAD → batch finalize path *and* sits under the model's own 35 s clip limit),
   return `{"text": ..., "segments": [], "detected_language": None}`.
2. One name in the `BACKENDS` dict (`eval/backends.py:219`).
3. Set the language per meeting in `eval/manifest.json` — there is no
   auto-detect to fall back on.
4. `uv run --group eval eval/transcribe.py`, then `eval/adjudicate.py`, judge in
   the browser, then `--score` the downloaded JSON.

Arms: **parakeet, whisper, cohere.** Keep Whisper — it is the pivot and the
Phase 0 tie-reference, so a new run stays comparable to the old one.

**The bar: it must beat Parakeet head-to-head on de+en meeting audio. A tie is a
loss.** Parakeet is 3× smaller, already shipped, cross-platform via ONNX, and
carries the timestamps this candidate cannot. Parity buys a second model
download, a merge layer, and a new failure mode, in exchange for nothing.

Watch for two things while judging: whether losses cluster at **code-switched**
sites (limitation 3, and the one our audio is built to expose), and whether the
arm invents text in pauses despite `split_at_silences` (limitation 4).

---

## C2 — only if C1 wins: the second-pass merge

Unwritten on purpose. If C1 loses, none of this needs designing; these are the
questions that would then be open, not a design.

- **Alignment.** Timestamp-free text has to be aligned to Parakeet's timed words
  (`difflib`/Needleman-Wunsch) and the timings transplanted. Where the two agree,
  the merge is a no-op. **The layer only ever acts at disagreement sites — which
  is precisely what C1 scored**, so C1's margin is a direct estimate of this
  layer's ceiling.
- **The timing hazard.** At sites where the word *counts* differ, transplanted
  timings are interpolation, and they feed `pipeline.py:146`'s midpoint speaker
  assignment. Cheapest bound: accept only equal-length substitutions and leave
  insertions/deletions to Parakeet. Decide whether that restriction throws away
  most of the win before building the general case.
- **The metric.** Not headline WER, and not B-WER. `c837c69`'s bar: **U-WER and
  false insertions**, or this layer gets adopted for the same reason the
  post-correction layer did.
- **Cost.** A 2 B model re-reading the whole meeting *after* Parakeet already
  did. At best it roughly doubles finalize time; realistically worse, since
  Parakeet's ONNX CPU path runs 36–44× realtime.
- **Scope: finalize only, opt-in, macOS only** until an ONNX export exists. The
  live pass is out of reach on both timestamps and latency.

---

## Decided — do not re-litigate

- **Not a backend.** It fails `asr/base.py:13`. No registration in
  `asr/registry.py` regardless of how C1 goes.
- **The VAD question is closed.** Silero v5 already runs in front of ASR on both
  passes (`vad.py`, `pipeline.py`, `live.py`) — for window cutting and compute
  gating, not hallucination suppression, since Parakeet is a transducer. It
  happens to be exactly the mitigation this model's card asks for, so a
  challenger inherits it free. Its parameters are tuned against eval audio with
  the negative results written down (`vad.py:51`, the reverted 0.4 threshold);
  do not retune them for a challenger.
- **Handy's answer does not transfer.** Handy runs this model well because its
  output is dictated text into a text field: no timestamps needed, VAD in front,
  single speaker. Our output contract is a diarized transcript. Same model,
  opposite conclusion — and that asymmetry is the point, not a contradiction.
- **`eval/adjudicate.py` is the evaluation path**, not hand-corrected
  references: Phase 0 found those slow and anchor-biased.

---

## What would reopen this

Any one of these changes the answer enough to justify re-reading the file:

- **The model gains word timestamps** upstream (an aligned export, or a NeMo-style
  forced-alignment path that ships). Then it becomes a candidate *backend* and
  skips C2 entirely.
- **An ONNX export appears.** Without one this is macOS-only, and the
  cross-platform default is `parakeet-onnx`.
- **A new leaderboard leader that already emits word timestamps.** That one goes
  straight to `asr/registry.py` as a `BackendSpec` — the seam exists precisely
  for it — and this file is irrelevant to it.
- **Our audio changes.** The graveyard's verdicts are about meeting audio: far-field,
  overlapped, disfluent, code-switched. A different corpus is a different question.
