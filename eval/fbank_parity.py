"""Can we compute the embedder's input features ourselves, exactly?

The shared-trunk route (`PLAN-DIARIZATION-SPEED.md`) has to leave sherpa's
`SpeakerEmbeddingExtractor` behind: it runs the trunk once per (chunk,
speaker) pair, and running it once over a long block instead means feeding the
ONNX session ourselves. That is only allowed if our own front end reproduces
sherpa's features — otherwise every downstream number is measured on
different input than production sees.

**The config is not Kaldi's default, and one value is the whole story:**
``high_freq`` is **7600 Hz**, not Nyquist. With Kaldi's default the features
are subtly wrong in every frame, embeddings land at cosine 0.83-0.92 against
sherpa, and the failure reads exactly like a broken front end rather than one
wrong constant (an earlier attempt spent a day at 0.855 and concluded the
route was risky). Everything else is Kaldi's: 80 mel bins, 25 ms / 10 ms,
dither 0, ``snip_edges=false``, povey window, preemphasis 0.97, power
spectrum, log floor at float32 epsilon.

Two things made this cheap once found, and both are worth knowing before
touching features again:

- ``OnlineStream.get_frames(index, n)`` exists precisely to compare FBANK
  across pipelines, so the front end is checkable against ground truth
  instead of inferred from embedding cosine. It aborts the **process** on an
  out-of-range request, so compute the frame count, never probe for it.
- ``kaldi-native-fbank`` on PyPI is the same k2-fsa library sherpa bundles.
  Diffing a candidate option set against it (in a throwaway env — it is NOT a
  dependency) identifies the config in one sweep.

Run::

    uv run --group eval eval/fbank_parity.py             # the gate
    uv run --group eval eval/fbank_parity.py --clips 20  # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from common import read_pcm16

from stenograf.audio import to_float32

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = 7600.0  # see the module docstring — NOT Nyquist
PREEMPH = 0.97
FFT_SIZE = 512  # Kaldi rounds the window up to a power of two
LOG_FLOOR = float(np.finfo(np.float32).eps)

MODEL = Path.home() / ".cache/stenograf/eres2net-voxceleb-16k.onnx"
CHANNEL = Path("eval/audio/ami/ES2003c.loop.wav")
GATE = 1e-4
"""Cosine bar: production embeddings must be identical to sherpa's to well
inside any threshold the loop cares about (the operating point is 0.56)."""


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def mel_filterbank() -> np.ndarray:
    """Kaldi's triangular mel bank, [NUM_BINS, FFT_SIZE // 2 + 1]."""
    num_fft_bins = FFT_SIZE // 2
    mel_low, mel_high = _mel(LOW_FREQ), _mel(HIGH_FREQ)
    delta = (mel_high - mel_low) / (NUM_BINS + 1)
    mels = _mel(SAMPLE_RATE / FFT_SIZE * np.arange(num_fft_bins))

    bank = np.zeros((NUM_BINS, num_fft_bins + 1), dtype=np.float64)
    for b in range(NUM_BINS):
        left = mel_low + b * delta
        center, right = left + delta, left + 2 * delta
        rising = (mels > left) & (mels <= center)
        falling = (mels > center) & (mels < right)
        bank[b, :num_fft_bins][rising] = (mels[rising] - left) / delta
        bank[b, :num_fft_bins][falling] = (right - mels[falling]) / delta
    return bank


def num_frames(num_samples: int) -> int:
    """Kaldi's ``snip_edges=false`` count — the frame is centred on its slot,
    so a 1 s clip yields 100 frames, not 98."""
    return (num_samples + FRAME_SHIFT // 2) // FRAME_SHIFT


_BANK = mel_filterbank()
_WINDOW = np.power(
    0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LENGTH) / (FRAME_LENGTH - 1)), 0.85
)


def fbank(audio: np.ndarray) -> np.ndarray:
    """Log-mel features, [frames, NUM_BINS]."""
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    starts = np.arange(num_frames(n)) * FRAME_SHIFT + FRAME_SHIFT // 2 - FRAME_LENGTH // 2
    index = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
    # Kaldi mirrors out-of-range indices at both edges (feature-window.cc).
    while (index < 0).any() or (index >= n).any():
        index = np.where(index < 0, -index - 1, index)
        index = np.where(index >= n, 2 * n - index - 1, index)

    frames = audio[index]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    frames[:, 1:] -= PREEMPH * frames[:, :-1]  # preemphasis, backwards in place
    frames[:, 0] -= PREEMPH * frames[:, 0]
    frames *= _WINDOW
    spectrum = np.abs(np.fft.rfft(frames, n=FFT_SIZE)) ** 2
    return np.log(np.maximum(spectrum @ _BANK.T, LOG_FLOOR))


def features(audio: np.ndarray) -> np.ndarray:
    """What the ONNX session actually receives: fbank minus its per-call mean
    over time. sherpa applies this outside the graph; dropping it moves cosine
    0.86 -> 0.33, so it is part of the front end, not an optional nicety."""
    f = fbank(audio)
    return f - f.mean(axis=0, keepdims=True)


def _clips(audio: np.ndarray, count: int, seconds: float, rng) -> list[np.ndarray]:
    span = int(seconds * SAMPLE_RATE)
    return [audio[s : s + span] for s in rng.integers(0, len(audio) - span - 1, count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=20, help="clips per length regime")
    args = parser.parse_args()

    import onnxruntime
    import sherpa_onnx

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=1)
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(MODEL), sess_options=options, providers=["CPUExecutionProvider"]
    )

    def theirs_frames(clip):
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(clip, dtype=np.float32))
        stream.input_finished()
        n = num_frames(len(clip))  # NEVER probe: out of range aborts the process
        return np.reshape(np.asarray(stream.get_frames(0, n), dtype=np.float64), (n, -1))

    def theirs_embedding(clip):
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(clip, dtype=np.float32))
        stream.input_finished()
        v = np.asarray(extractor.compute(stream), dtype=np.float64)
        return v / np.linalg.norm(v)

    def ours_embedding(clip):
        (emb,) = session.run(None, {"x": features(clip).astype(np.float32)[None]})
        v = np.asarray(emb, dtype=np.float64).ravel()
        return v / np.linalg.norm(v)

    audio = to_float32(read_pcm16(CHANNEL))
    rng = np.random.default_rng(7)
    regimes: list[tuple[str, list[np.ndarray]]] = [
        (f"contiguous {s:g} s", _clips(audio, args.clips, s, rng))
        for s in (0.5, 1.0, 4.0, 10.0, 30.0)
    ]
    # Production embeds CONCATENATED sole-speaker runs, never plain contiguous
    # audio (diarization/loop.py) — the seam most likely to break parity.
    concatenated = [
        np.concatenate(
            [
                audio[s : s + int(rng.integers(4000, 32000))]
                for s in rng.integers(0, len(audio) - 32001, rng.integers(2, 6))
            ]
        )
        for _ in range(args.clips)
    ]
    regimes.append(("concatenated runs (production)", concatenated))

    print(f"{'regime':34s} {'n':>4s} {'worst 1-cos':>12s} {'worst feat |d|':>15s}")
    ok = True
    for label, clips in regimes:
        worst_cos = max(
            1.0 - float(np.dot(ours_embedding(c), theirs_embedding(c))) for c in clips
        )
        worst_feat = max(float(np.abs(fbank(c) - theirs_frames(c)).max()) for c in clips)
        ok &= worst_cos < GATE
        print(f"{label:34s} {len(clips):4d} {worst_cos:12.3e} {worst_feat:15.3e}")
    print(f"\ngate (1-cos < {GATE:g} everywhere): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
