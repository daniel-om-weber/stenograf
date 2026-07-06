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

## Side quests

- ~~Canary-1B-v2 runtime~~ — resolved, see above: no accelerated runtime with
  word timestamps exists; NeMo-on-MPS reference backend wired up instead.
- **speakrs diarization sanity check** on the same audio (not wired up yet).

## Metrics

`transcribe.py` records per segment: wall time, speed (×RT), model load time,
peak RSS, MLX peak memory, detected language. `score.py` adds WER/CER (jiwer)
after normalization (lowercase, punctuation stripped, umlauts kept).
