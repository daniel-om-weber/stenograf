"""Diarization with NVIDIA Nemotron 3 Diarization through its ONNX export.

The export holds one chunk step: log-mel features plus cached encoder frames
in; per-10-ms speaker logits, the chunk's encoder frames and the learned
silence frame out. Everything around it is a numpy port of transformers 5.23's
offline ``Nemotron3DiarizationForAudioFrameClassification`` and
``Nemotron3DiarizationSpeakerCache`` (the eval port, ``eval/nemotron_arm.py``,
matched the PyTorch reference to 7.6e-6 on CPU, 2026-11-15). Two traps the
port keeps:

- the ONNX ``attention_mask`` is per *encoder* frame of the step (cached +
  chunk), not per mel frame;
- the reference front end marks the last mel frame invalid, and the encoder
  mask is sampled at each 8-frame group's first frame — so when that frame
  opens the final group, the last encoder frame is masked, and ignoring it
  moves the final chunk's probabilities by up to 0.06.

The model takes no speaker count. A stated count is applied afterwards as a
cap (:func:`turns_from_probs`): measured on the held-out AMI set it lifts
attribution 97.4 → 97.8 % and makes re-ID perfect at the shipped threshold,
because the extra slots it removes are headset crosstalk (``eval/README.md``).
"""

from __future__ import annotations

import functools
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stenograf import assets
from stenograf.audio import to_float32
from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.diarization.embedding import VoiceEmbedder

N_FFT = 512
WIN_LENGTH = 400
HOP = 160
"""Mel frame step, samples: one probability row per 10 ms."""
NUM_MELS = 128
PREEMPHASIS = 0.97
_LOG_GUARD = 2.0**-24
_FRAMES_PER_SECOND = 100


@functools.cache
def _mel_filters() -> np.ndarray:
    """(128, 257) Slaney-scale, Slaney-normalized triangles, float32 — the
    reference's ``mel_filter_bank``: built in float64, cast to float32
    *before* the area normalization (the order is what makes it bit-exact)."""

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        linear = 3.0 * hz / 200.0
        log = 15.0 + np.log(np.maximum(hz, 1e-12) / 1000.0) * (27.0 / np.log(6.4))
        return np.where(hz >= 1000.0, log, linear)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        linear = 200.0 * mel / 3.0
        log = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel - 15.0))
        return np.where(mel >= 15.0, log, linear)

    low, high = hz_to_mel(np.array([0.0, 8000.0]))
    centers = mel_to_hz(np.linspace(low, high, NUM_MELS + 2))
    bins = np.linspace(0, 8000, N_FFT // 2 + 1)
    slopes = centers[None, :] - bins[:, None]
    widths = np.diff(centers)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    bank = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)
    bank *= (2.0 / (centers[2:] - centers[:-2]))[None, :]
    return np.ascontiguousarray(bank.T)


@functools.cache
def _window() -> np.ndarray:
    """Symmetric Hann of WIN_LENGTH, zero-padded to N_FFT centered (torch.stft)."""
    n = np.arange(WIN_LENGTH, dtype=np.float64)
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (WIN_LENGTH - 1))).astype(np.float32)
    window = np.zeros(N_FFT, dtype=np.float32)
    left = (N_FFT - WIN_LENGTH) // 2
    window[left : left + WIN_LENGTH] = hann
    return window


def num_mel_frames(num_samples: int) -> int:
    """Frames of a centered STFT; the last one is padding-only and masked."""
    return num_samples // HOP + 1


def log_mel(audio: np.ndarray, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Log-mel features of mel frames ``[start, stop)`` of float32 ``audio``.

    Frame for frame what the reference extractor computes over the whole
    recording (preemphasis, centered zero-padded STFT, power, Slaney mel,
    ``log(x + 2**-24)``, the last frame zeroed), so the chunk loop can build
    each step's frames on demand instead of holding a channel's features
    (~2 GB of STFT frames per meeting-hour if built at once). Matches the
    reference to 3e-4 at worst, 1e-7 on average, 90 % of values bit-equal
    (2026-11-15, one 14-min AMI channel); the parity test holds a fixture."""
    total = num_mel_frames(len(audio))
    stop = total if stop is None else min(stop, total)
    if stop <= start:
        return np.zeros((0, NUM_MELS), dtype=np.float32)
    # Frame t reads emphasized samples [t*HOP - N_FFT/2, t*HOP + N_FFT/2),
    # zeros outside the signal (centered STFT, constant padding).
    lo, hi = start * HOP - N_FFT // 2, (stop - 1) * HOP + N_FFT // 2
    padded = np.zeros(hi - lo, dtype=np.float32)
    a, b = max(lo, 0), min(hi, len(audio))
    if b > a:
        tap = 1 if a > 0 else 0  # the sample before the range feeds its first preemphasis
        x = np.asarray(audio[a - tap : b], dtype=np.float32)
        emphasized = x[tap:].copy()
        emphasized[1 - tap :] -= np.float32(PREEMPHASIS) * x[: len(x) - 1]
        padded[a - lo : b - lo] = emphasized
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    spectrum = np.fft.rfft(frames * _window(), axis=1)
    real = spectrum.real.astype(np.float32)
    imag = spectrum.imag.astype(np.float32)
    power = np.sqrt(real * real + imag * imag) ** 2  # the reference's |X| then squared
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mel = _mel_filters() @ power.T  # Accelerate raises spurious FP flags on clean GEMMs
    features = np.log(mel + np.float32(_LOG_GUARD)).T.astype(np.float32)
    if stop == total:
        features[-1] = 0.0
    return np.ascontiguousarray(features)


@dataclass(frozen=True)
class Config:
    """Offline chunking and speaker-cache policy, from the export's config.json."""

    subsampling: int
    chunk: int
    right: int
    left: int
    fifo_length: int
    update_period: int
    cache_length: int
    num_silence: int
    score_threshold: float
    latest_boost: float
    num_speakers: int
    min_positive: int
    num_strong: int
    num_weak: int

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = json.loads(path.read_text())
        stream = raw["streaming_config"]
        num_speakers = stream["num_speakers"]
        cache_length = stream["speaker_cache_length"]
        num_silence = stream["speaker_cache_silence_frames_per_speaker"]
        budget = cache_length // num_speakers - num_silence
        return cls(
            subsampling=raw["audio_config"]["subsampling_factor"],
            chunk=raw["chunk_length"],
            right=raw["chunk_right_context"],
            left=raw.get("chunk_left_context", 0),
            # offline mode overrides the streaming FIFO sizes with the top-level ones
            fifo_length=raw["fifo_length"],
            update_period=raw["speaker_cache_update_period"],
            cache_length=cache_length,
            num_silence=num_silence,
            score_threshold=stream["prediction_score_threshold"],
            latest_boost=stream["latest_frames_score_boost"],
            num_speakers=num_speakers,
            min_positive=math.floor(budget * stream["min_positive_scores_rate"]),
            num_strong=math.floor(budget * stream["strong_boost_rate"]),
            num_weak=math.floor(budget * stream["weak_boost_rate"]),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # exp(-x) → inf for very negative logits, 1/inf = 0
        return 1.0 / (1.0 + np.exp(-x))


class SpeakerCache:
    """Arrival-Order Speaker Cache + FIFO of recent encoder frames (batch 1)."""

    def __init__(self, cfg: Config, dim: int = 512) -> None:
        self.cfg = cfg
        self.embeds = np.zeros((cfg.cache_length, dim), np.float32)
        self.probs = np.zeros((cfg.cache_length, cfg.num_speakers), np.float32)
        self.fifo = np.zeros((cfg.fifo_length + cfg.chunk + cfg.right, dim), np.float32)
        self.num_cache = 0
        self.num_fifo = 0
        self.compressed = False

    def frames(self) -> np.ndarray:
        """The cached frames every chunk attends to: the speaker cache, then the FIFO."""
        return np.concatenate([self.embeds[: self.num_cache], self.fifo[: self.num_fifo]], 0)

    def _num_popped(self, num_fifo: int) -> int:
        if num_fifo <= self.cfg.fifo_length:
            return 0
        return min(max(self.cfg.update_period, num_fifo - self.cfg.fifo_length), num_fifo)

    def update(
        self,
        step_embeds: np.ndarray,
        logits: np.ndarray,
        silence: np.ndarray,
        num_chunk: int,
        num_lookback: int,
        mask: np.ndarray,
    ) -> None:
        """Push a processed chunk into the FIFO, moving its oldest frames to the
        cache when it overflows (``Nemotron3DiarizationSpeakerCache.update``)."""
        cfg = self.cfg
        num_cache, num_fifo = self.num_cache, self.num_fifo
        steps = logits.shape[0] // cfg.subsampling
        pooled = _sigmoid(logits[: steps * cfg.subsampling])
        probs = pooled.reshape(steps, cfg.subsampling, cfg.num_speakers).mean(1)
        probs = (probs * mask[:steps, None]).astype(np.float32)
        start = num_cache + num_fifo + num_lookback
        fifo = np.concatenate([self.fifo[:num_fifo], step_embeds[start : start + num_chunk]], 0)
        popped = self._num_popped(len(fifo))
        if popped:
            fifo_probs = np.concatenate(
                [probs[num_cache : num_cache + num_fifo], probs[start : start + num_chunk]], 0
            )
            # a compressed cache is out of order, so only its stored probabilities are
            # valid; an uncompressed one still holds plain chunk frames this step re-estimated
            stored = self.probs[:num_cache] if self.compressed else probs[:num_cache]
            cache_embeds = np.concatenate([self.embeds[:num_cache], fifo[:popped]], 0)
            cache_probs = np.concatenate([stored, fifo_probs[:popped]], 0)
            fifo = fifo[popped:]
            if len(cache_embeds) > cfg.cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence)
                self.compressed = True
            self.num_cache = len(cache_embeds)
            self.embeds[: self.num_cache] = cache_embeds
            self.probs[: self.num_cache] = cache_probs
        self.num_fifo = len(fifo)
        self.fifo[: self.num_fifo] = fifo

    def _scores(self, probs: np.ndarray) -> np.ndarray:
        threshold = self.cfg.score_threshold
        log_probs = np.log(np.maximum(probs, threshold))
        log_complements = np.log(np.maximum(1 - probs, threshold))
        scores = log_probs - log_complements + log_complements.sum(-1, keepdims=True)
        scores = scores - math.log(0.5)
        speech = probs > 0.5
        scores = np.where(speech, scores, -np.inf)
        positive = scores > 0
        enough = positive.sum(0, keepdims=True) >= self.cfg.min_positive
        return np.where(~positive & speech & enough, -np.inf, scores)

    @staticmethod
    def _boost(scores: np.ndarray, count: int, boost: float) -> np.ndarray:
        top = np.argpartition(-scores, count - 1, axis=0)[:count]
        boosted = scores.copy()
        columns = np.broadcast_to(np.arange(scores.shape[1]), top.shape)
        np.add.at(boosted, (top, columns), boost)
        return boosted

    def _compress(
        self, embeds: np.ndarray, probs: np.ndarray, silence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep the cache_length most important frames, grouped by speaker in
        original order, with silence slots per speaker
        (``Nemotron3DiarizationSpeakerCache._compress``)."""
        cfg = self.cfg
        num_frames = len(probs)
        scores = self._scores(probs)
        scores[cfg.cache_length :] += cfg.latest_boost  # the frames just popped from the FIFO
        scores = self._boost(scores, cfg.num_strong, -2.0 * math.log(0.5))
        scores = self._boost(scores, cfg.num_weak, -math.log(0.5))
        silence_rows = np.full((cfg.num_silence, cfg.num_speakers), np.inf, np.float32)
        scores = np.concatenate([scores, silence_rows], 0)
        embeds = np.concatenate([embeds, silence[None, :]], 0)
        probs = np.concatenate([probs, np.zeros((1, cfg.num_speakers), np.float32)], 0)
        num_scored = num_frames + cfg.num_silence
        sentinel = num_scored * cfg.num_speakers
        flat = scores.T.reshape(-1)  # speaker-major, as the reference flattens it
        top = np.argpartition(-flat, cfg.cache_length - 1)[: cfg.cache_length]
        top = np.sort(np.where(flat[top] == -np.inf, sentinel, top))
        frame = np.where(top == sentinel, num_frames, np.minimum(top % num_scored, num_frames))
        return embeds[frame], probs[frame]


def speaker_probabilities(session: Any, cfg: Config, audio: np.ndarray) -> np.ndarray:
    """Offline inference over a whole recording: float32 16 kHz ``audio`` →
    (mel frames, num_speakers) probabilities, one row per 10 ms. ``session`` is
    an ``onnxruntime.InferenceSession`` over the export (tests pass a double)."""
    factor = cfg.subsampling
    num_mel = num_mel_frames(len(audio))
    num_embeds = math.ceil(num_mel / factor)
    mel_mask = np.arange(num_mel) < num_mel - 1
    embed_mask = mel_mask.astype(np.int64)[::factor]
    cache = SpeakerCache(cfg)
    chunks = []
    for start in range(0, num_embeds, cfg.chunk):
        end = min(start + cfg.chunk, num_embeds)
        context_start = max(start - cfg.left, 0)
        num_lookback = start - context_start
        step_features = log_mel(
            audio, context_start * factor, min(end + cfg.right, num_embeds) * factor
        )
        if len(step_features) % factor:  # the reference zero-pads the last frame group
            pad = factor - len(step_features) % factor
            step_features = np.pad(step_features, ((0, pad), (0, 0)))
        cached = cache.frames()
        num_step = len(step_features) // factor
        step_mask = np.concatenate(
            [np.ones(len(cached), np.int64), embed_mask[context_start : context_start + num_step]]
        )
        logits, chunk_embeds, silence = session.run(
            None,
            {
                "input_features": step_features[None],
                "cached_embeds": cached[None],
                "attention_mask": step_mask[None],
            },
        )
        logits = np.asarray(logits[0], dtype=np.float32)
        step_embeds = np.concatenate([cached, np.asarray(chunk_embeds[0], np.float32)], 0)
        cache.update(
            step_embeds,
            logits,
            np.asarray(silence, np.float32),
            end - start,
            num_lookback,
            step_mask,
        )
        first = (len(cached) + num_lookback) * factor
        chunks.append(logits[first : first + (end - start) * factor])
    return _sigmoid(np.concatenate(chunks, 0)[:num_mel])


def turns_from_probs(probs: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
    """Speaker turns, overlap kept, one ``S<slot>`` label per model slot.

    With ``num_speakers``, the count is a cap: keep the most active slots,
    and frames only dropped slots claimed go to the most probable kept slot
    (a count can only remove speakers — Nemotron never under-counted on the
    held-out set). A count at or above the slots found changes nothing."""
    active = probs > 0.5
    if num_speakers is not None and num_speakers > 0:
        keep = np.argsort(-active.sum(0))[:num_speakers]
        capped = np.zeros_like(active)
        capped[:, keep] = active[:, keep]
        orphan = active.any(1) & ~capped.any(1)
        best = keep[np.argmax(probs[:, keep], 1)]
        capped[np.nonzero(orphan)[0], best[orphan]] = True
        active = capped
    turns: list[SpeakerTurn] = []
    for slot in range(active.shape[1]):
        edges = np.diff(np.concatenate([[0], active[:, slot].astype(np.int8), [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        turns += [
            SpeakerTurn(f"S{slot}", s / _FRAMES_PER_SECOND, e / _FRAMES_PER_SECOND)
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True)
        ]
    turns.sort(key=lambda t: (t.start, t.speaker))
    return turns


class NemotronDiarizer(Diarizer):
    """Nemotron's turns, voices from the shared :class:`VoiceEmbedder`.

    ``model_dir`` overrides the fetched assets (a directory holding
    ``model.onnx``, its ``.onnx_data`` and ``config.json``)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        embedder: VoiceEmbedder | None = None,
        progress: assets.ProgressHook | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._progress = progress
        self.embedder = embedder if embedder is not None else VoiceEmbedder(progress=progress)
        self._session = None
        self._config: Config | None = None
        self._init_lock = threading.Lock()

    def _load(self):
        # The chunk loop is sequential (each step reads the cache the previous
        # one wrote), so intra-op threads are its only parallelism: ORT's
        # default, which the measured 15 s per audio-hour used (M4 Max, fp32,
        # onnxruntime 1.32).
        if self._session is None:
            with self._init_lock:
                if self._session is None:
                    import onnxruntime

                    if self._model_dir is not None:
                        graph = self._model_dir / "model.onnx"
                        config = self._model_dir / "config.json"
                    else:
                        graph, _, config = (
                            assets.fetch(asset, self._progress) for asset in assets.NEMOTRON
                        )
                    self._config = Config.load(config)
                    self._session = onnxruntime.InferenceSession(
                        str(graph), providers=["CPUExecutionProvider"]
                    )
        assert self._config is not None
        return self._session, self._config

    def probabilities(self, samples: np.ndarray) -> np.ndarray:
        """(10-ms frames, slots) speaker probabilities of mono 16 kHz PCM."""
        session, config = self._load()
        return speaker_probabilities(session, config, to_float32(samples))

    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        if len(samples) < HOP:  # no valid mel frame: nothing for the model to score
            return []
        return turns_from_probs(self.probabilities(samples), num_speakers)

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        turns = self.diarize(samples, num_speakers)
        return DiarizationResult(turns, self.embedder.cluster_embeddings(turns, samples))

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        return self.embedder.channel_embedding(samples, turns)
