"""Can we compute the embedder's input features ourselves, exactly?

Running the embedding trunk over a long block instead of once per (chunk,
speaker) pair means feeding the ONNX session directly, leaving sherpa's
``SpeakerEmbeddingExtractor`` — and its front end — behind. That is only
allowed if our own front end reproduces sherpa's features, or every number
measured downstream describes different input than production sees.

**The config is not Kaldi's default in four places** (measured 2026-08-11,
by diffing option sets against sherpa's own frames): ``num_bins`` 80,
``dither`` 0, ``snip_edges`` false, and — the one that is neither obvious nor
guessable — **``high_freq`` 7600 Hz rather than Nyquist**. Getting any single
one of them wrong lands embedding cosine in the 0.66–0.98 band, i.e. looks
like a broken front end rather than one wrong constant; ``high_freq`` at
Nyquist alone costs cosine 0.76–0.93, and the log floor (float32 epsilon)
costs more still at 0.67.

Two facts make this checkable rather than guessable, and both are why this
file exists:

- ``OnlineStream.get_frames(index, n)`` is documented "for comparing FBANK
  features across pipelines", so the front end is verified against ground
  truth instead of inferred from embedding cosine. It aborts the **process**
  on an out-of-range request, so compute the frame count, never probe for it.
- ``kaldi-native-fbank`` on PyPI is the same k2-fsa library sherpa bundles.
  Sweeping option sets against it in a throwaway env (never a dependency)
  names the config in one pass.

Run::

    uv run --group eval eval/fbank_parity.py              # ~120 live clips
    uv run --group eval eval/fbank_parity.py --clips 5    # smoke
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from common import AUDIO_DIR, read_pcm16

from stenograf import assets
from stenograf.audio import to_float32

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = 7600.0  # NOT Nyquist — see the module docstring
PREEMPH = 0.97
FFT_SIZE = 512  # Kaldi rounds the window up to a power of two
LOG_FLOOR = float(np.finfo(np.float32).eps)

CHANNEL = AUDIO_DIR / "ami" / "ES2003c.loop.wav"

COSINE_GATE = 1e-4
"""Embeddings must be identical to sherpa's well inside anything the loop can
notice — its operating threshold is 0.56."""

FEATURE_GATE = 1e-3
"""Log-domain feature residual. Not zero because sherpa computes in float32
and this in float64; measured worst 4.9e-4 (2026-08-11), and a float32 front
end sits in the same band, so this bounds rounding rather than admitting a
real difference."""

SILENCE_RMS = 1e-4
"""Below this a clip is digital silence, which every front end reproduces
whatever its config (verified: silent clips score cosine > 0.999 even with
``high_freq`` wrong). The reference channel is ~45 % exact-zero samples, so
without this filter a third of the short-clip regimes cannot fail."""


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
    so a 1 s clip yields 100 frames, not 98. Verified against sherpa at the
    boundaries, since asking it for one frame too many kills the process."""
    return (num_samples + FRAME_SHIFT // 2) // FRAME_SHIFT


_BANK = mel_filterbank()
_WINDOW = np.power(
    0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LENGTH) / (FRAME_LENGTH - 1)), 0.85
)


def fbank(audio: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Log-mel features, [frames, NUM_BINS]."""
    audio = np.asarray(audio, dtype=dtype)
    n = len(audio)
    starts = np.arange(num_frames(n)) * FRAME_SHIFT + FRAME_SHIFT // 2 - FRAME_LENGTH // 2
    index = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
    # Kaldi mirrors out-of-range indices at both edges rather than zero-padding
    # (feature-window.cc); zero-padding instead costs cosine ~0.9986.
    while (index < 0).any() or (index >= n).any():
        index = np.where(index < 0, -index - 1, index)
        index = np.where(index >= n, 2 * n - index - 1, index)

    frames = audio[index]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    frames[:, 1:] -= PREEMPH * frames[:, :-1]  # numpy reads the RHS first
    frames[:, 0] -= PREEMPH * frames[:, 0]
    frames *= _WINDOW.astype(dtype)
    spectrum = np.abs(np.fft.rfft(frames, n=FFT_SIZE)) ** 2
    return np.log(np.maximum(spectrum @ _BANK.T.astype(dtype), LOG_FLOOR))


def features(audio: np.ndarray, dtype=np.float64) -> np.ndarray:
    """What the ONNX session actually receives: fbank minus its per-call mean
    over time. sherpa applies this outside the graph, and it is not optional —
    without it embeddings collapse to cosine 0.10–0.45 against sherpa's
    (median 0.45, measured 2026-08-11)."""
    f = fbank(audio, dtype)
    return f - f.mean(axis=0, keepdims=True)


def _live(clips: list[np.ndarray]) -> list[np.ndarray]:
    return [c for c in clips if np.sqrt(np.mean(np.square(c))) > SILENCE_RMS]


def _regimes(audio: np.ndarray, count: int, rng) -> list[tuple[str, list[np.ndarray]]]:
    regimes = []
    for seconds in (0.5, 1.0, 4.0, 10.0, 30.0):
        span = int(seconds * SAMPLE_RATE)
        clips: list[np.ndarray] = []
        while len(clips) < count:  # keep drawing until `count` are non-silent
            start = int(rng.integers(0, len(audio) - span - 1))
            clips += _live([audio[start : start + span]])
        regimes.append((f"contiguous {seconds:g} s", clips))

    # What `_pair_embeddings` really builds: runs sliced on the WINDOW/FRAMES
    # scale (~272 samples per frame) inside one chunk, concatenated, with only
    # a summed MIN_EMBED_FRAMES floor — so 10 pieces of ~272 samples (0.17 s
    # total) is a legal production input, an order of magnitude shorter and
    # more fragmented than a naive "few long pieces" sample.
    frame = 272
    production: list[np.ndarray] = []
    while len(production) < count:
        pieces = int(rng.integers(10, 40))
        parts = [
            audio[s : s + int(rng.integers(1, 18)) * frame]
            for s in rng.integers(0, len(audio) - 18 * frame - 1, pieces)
        ]
        production += _live([np.concatenate(parts)])
    regimes.append(("production pair slices", production))
    return regimes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=20, help="live clips per regime")
    args = parser.parse_args()

    import onnxruntime
    import sherpa_onnx

    model = assets.fetch(assets.SPEAKER_EMBEDDING)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1)
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )

    def theirs(clip):
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(clip, dtype=np.float32))
        stream.input_finished()
        n = num_frames(len(clip))  # NEVER probe: out of range aborts the process
        frames = np.reshape(np.asarray(stream.get_frames(0, n), dtype=np.float64), (n, -1))
        vector = np.asarray(extractor.compute(stream), dtype=np.float64)
        return frames, vector / np.linalg.norm(vector)

    def ours(clip, dtype):
        (emb,) = session.run(None, {"x": features(clip, dtype).astype(np.float32)[None]})
        vector = np.asarray(emb, dtype=np.float64).ravel()
        return vector / np.linalg.norm(vector)

    audio = to_float32(read_pcm16(CHANNEL))
    rng = np.random.default_rng(7)

    header = f"{'regime':30s} {'n':>4s} {'worst 1-cos':>12s} {'f32 1-cos':>11s} {'feat |d|':>11s}"
    print(header)
    ok = True
    for label, clips in _regimes(audio, args.clips, rng):
        worst_cos = worst_f32 = worst_feat = 0.0
        for clip in clips:
            reference_frames, reference = theirs(clip)
            worst_feat = max(worst_feat, float(np.abs(fbank(clip) - reference_frames).max()))
            worst_cos = max(worst_cos, 1.0 - float(np.dot(ours(clip, np.float64), reference)))
            worst_f32 = max(worst_f32, 1.0 - float(np.dot(ours(clip, np.float32), reference)))
        ok &= worst_cos < COSINE_GATE and worst_feat < FEATURE_GATE
        print(
            f"{label:30s} {len(clips):4d} {worst_cos:12.3e} {worst_f32:11.3e} {worst_feat:11.3e}"
        )
    print(
        f"\ngate (1-cos < {COSINE_GATE:g} and feature |d| < {FEATURE_GATE:g} everywhere): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
