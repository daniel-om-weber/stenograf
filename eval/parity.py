"""MLX ↔ ONNX Parakeet parity (Phase 5 verification, label-free).

Runs the two shipped backends — ``parakeet`` (parakeet-mlx) and
``parakeet-onnx`` (onnx-asr fp32) — over the eval WAVs and reports how far
apart they are, without hand-written references (the Phase-2 approach: the
same model in two runtimes should agree, and where it doesn't, disagreement
is symmetric evidence, not error):

- cross-WER between the two transcripts (normalized as in score.py),
- word-timestamp deltas on jiwer-aligned *equal* words (TDT timestamps are
  80 ms-quantized, so deltas well under ~0.2 s mean speaker attribution is
  unaffected),
- per-backend timestamp sanity (monotonic starts, spans inside the audio).

Advisory thresholds, not a test suite: cross-WER ≤ 8 % and median |Δstart|
≤ 0.12 s per file prints PASS. The WER bar is the measured fp32 two-runtime
disagreement on this audio (2.0–6.8 %, 2026-07-11; sherpa's int8-only v3
export sat at 4.1–20.8 %, which is why Decision A went to onnx-asr — see
asr/parakeet_onnx.py). macOS-arm64 only (MLX).

Usage: uv run --group eval python eval/parity.py [wav ...]
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import normalize  # noqa: E402

from stenograf.asr import create_backend  # noqa: E402

DEFAULT_WAVS = sorted((Path(__file__).parent / "audio").glob("*.wav"))

WER_THRESHOLD = 0.08
DELTA_THRESHOLD = 0.12  # seconds; TDT timestamps are 80 ms frames


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def run_backend(backend, samples: np.ndarray) -> tuple[list, float]:
    t0 = time.perf_counter()
    segments = backend.transcribe(samples, None)
    words = [w for seg in segments for w in seg.words]
    return words, time.perf_counter() - t0


def sanity(words: list, duration: float) -> list[str]:
    problems = []
    starts = [w.start for w in words]
    if starts != sorted(starts):
        problems.append("word starts not monotonic")
    if any(w.end < w.start for w in words):
        problems.append("word with end < start")
    if words and not (words[0].start >= 0 and words[-1].end <= duration + 1.0):
        problems.append(f"span [{words[0].start:.2f}, {words[-1].end:.2f}] outside audio")
    return problems


def _tokens_with_sources(words: list) -> tuple[list[str], list]:
    """Normalized tokens plus, per token, the source word carrying its time.

    ``normalize`` can split one word into several tokens ("15.7." → "15 7")
    or drop it entirely, so jiwer's token indices don't map 1:1 to words.
    """
    tokens: list[str] = []
    sources: list = []
    for word in words:
        for token in normalize(word.text).split():
            tokens.append(token)
            sources.append(word)
    return tokens, sources


def aligned_deltas(mlx_words: list, onnx_words: list) -> list[float]:
    """|Δstart| for tokens jiwer aligns as equal (normalized)."""
    mlx_tokens, mlx_sources = _tokens_with_sources(mlx_words)
    onnx_tokens, onnx_sources = _tokens_with_sources(onnx_words)
    out = jiwer.process_words(" ".join(mlx_tokens), " ".join(onnx_tokens))
    deltas = []
    for chunk in out.alignments[0]:
        if chunk.type != "equal":
            continue
        for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
            deltas.append(
                abs(mlx_sources[chunk.ref_start_idx + i].start
                    - onnx_sources[chunk.hyp_start_idx + i].start)
            )
    return deltas


def main() -> int:
    wavs = [Path(a) for a in sys.argv[1:]] or DEFAULT_WAVS
    mlx = create_backend("parakeet")
    onnx = create_backend("parakeet-onnx")
    print("loading backends…")
    mlx.load()
    onnx.load()

    all_ok = True
    print(f"{'file':<14} {'xWER':>6} {'medΔ':>6} {'p95Δ':>6} {'mlx':>7} {'onnx':>7}  verdict")
    for wav in wavs:
        samples = read_wav(wav)
        duration = len(samples) / 16000
        mlx_words, mlx_dt = run_backend(mlx, samples)
        onnx_words, onnx_dt = run_backend(onnx, samples)

        problems = [f"mlx: {p}" for p in sanity(mlx_words, duration)]
        problems += [f"onnx: {p}" for p in sanity(onnx_words, duration)]

        wer = jiwer.wer(
            normalize(" ".join(w.text for w in mlx_words)),
            normalize(" ".join(w.text for w in onnx_words)),
        )
        deltas = sorted(aligned_deltas(mlx_words, onnx_words))
        med = deltas[len(deltas) // 2] if deltas else float("nan")
        p95 = deltas[int(len(deltas) * 0.95)] if deltas else float("nan")

        ok = wer <= WER_THRESHOLD and med <= DELTA_THRESHOLD and not problems
        all_ok &= ok
        print(
            f"{wav.name:<14} {wer:>6.1%} {med:>5.2f}s {p95:>5.2f}s "
            f"{duration/mlx_dt:>6.1f}x {duration/onnx_dt:>6.1f}x  "
            + ("PASS" if ok else "FAIL " + "; ".join(problems))
        )

    mlx.unload()
    onnx.unload()
    print("PARITY:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
