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

import contextlib
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
from stenograf.diarization.base import (
    DiarizationResult,
    DiarizationStream,
    Diarizer,
    Reader,
    SpeakerTurn,
)
from stenograf.diarization.embedding import VoiceCollector, VoiceEmbedder
from stenograf.log import logger

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


def final_mel_frames(num_samples: int) -> int:
    """How many leading mel frames of a recording still being captured are
    already what they will be in the finished one.

    Frame t reads samples up to ``t*HOP + N_FFT/2 - 1``, so it is complete
    once that sample exists — and then it is never the finished recording's
    last frame, the one the reference zeroes and masks (that one starts at or
    after the final sample)."""
    return max(0, (num_samples - N_FFT // 2) // HOP + 1)


def _sample_span(start: int, stop: int, num_samples: int) -> tuple[int, int]:
    """The samples mel frames ``[start, stop)`` read: their STFT windows plus
    the one before, which feeds the first window's preemphasis."""
    lo = max(start * HOP - N_FFT // 2 - 1, 0)
    return lo, max(lo, min((stop - 1) * HOP + N_FFT // 2, num_samples))


def log_mel(
    audio: np.ndarray,
    start: int = 0,
    stop: int | None = None,
    *,
    offset: int = 0,
    num_samples: int | None = None,
) -> np.ndarray:
    """Log-mel features of mel frames ``[start, stop)`` of float32 ``audio``.

    Frame for frame what the reference extractor computes over the whole
    recording (preemphasis, centered zero-padded STFT, power, Slaney mel,
    ``log(x + 2**-24)``, the last frame zeroed), so the chunk loop can build
    each step's frames on demand instead of holding a channel's features
    (~2 GB of STFT frames per meeting-hour if built at once). Matches the
    reference to 3e-4 at worst, 1e-7 on average, 90 % of values bit-equal
    (2026-11-15, one 14-min AMI channel); the parity test holds a fixture.

    ``audio`` may be a window of a longer recording: it then starts at sample
    ``offset`` of a recording ``num_samples`` long and must cover
    :func:`_sample_span` of the frames asked for. The values do not depend on
    which window holds them — only on the frame range, which picks the
    matrix shapes the BLAS kernels see."""
    num_samples = len(audio) + offset if num_samples is None else num_samples
    total = num_mel_frames(num_samples)
    stop = total if stop is None else min(stop, total)
    if stop <= start:
        return np.zeros((0, NUM_MELS), dtype=np.float32)
    # Frame t reads emphasized samples [t*HOP - N_FFT/2, t*HOP + N_FFT/2),
    # zeros outside the signal (centered STFT, constant padding).
    lo, hi = start * HOP - N_FFT // 2, (stop - 1) * HOP + N_FFT // 2
    padded = np.zeros(hi - lo, dtype=np.float32)
    a, b = max(lo, 0), min(hi, num_samples)
    if b > a:
        tap = 1 if a > 0 else 0  # the sample before the range feeds its first preemphasis
        x = np.asarray(audio[a - tap - offset : b - offset], dtype=np.float32)
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


class ChunkLoop:
    """Nemotron's offline chunk loop, as state that can stop and resume.

    Every step attends to the speaker cache the step before it wrote, so the
    steps run strictly in order — but a step's own inputs (its features and
    attention mask) depend only on audio up to the end of its right context.
    Only the recording's end changes anything: the reference masks the
    finished recording's last mel frame and truncates the final chunk. So a
    meeting can run every step whose inputs are already final while it is
    still being recorded (:meth:`advance`), and only the rest once it ends
    (:meth:`finish`): the same steps on the same inputs in the same order —
    the probabilities are bit-identical to running them all at the end
    (``eval/README.md``, 2026-11-17). :func:`speaker_probabilities` is this
    loop run in one go, so the two paths cannot drift apart.

    Holds the cache (~1.4 MB) and every finished step's logits (~11 MB per
    audio-hour, the size of the result itself)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache = SpeakerCache(cfg)
        self._logits: list[np.ndarray] = []
        self.next_frame = 0
        """First encoder frame of the next step."""
        self.broken = False
        """A step failed after it began writing the cache, so this state is no
        longer the loop's: start over."""

    @property
    def steps(self) -> int:
        """Steps finished so far."""
        return len(self._logits)

    def probabilities(self, first_step: int) -> np.ndarray:
        """The rows the steps from ``first_step`` on produced; they never
        change once a step has finished."""
        logits = self._logits[first_step:]
        if not logits:
            return np.zeros((0, 0), np.float32)
        return _sigmoid(np.concatenate(logits, 0))

    def samples_needed(self) -> int:
        """Samples a recording still being captured must hold before the next
        step's inputs are final (:func:`final_mel_frames` inverted)."""
        cfg = self.cfg
        frames = (self.next_frame + cfg.chunk + cfg.right) * cfg.subsampling
        return (frames - 1) * HOP + N_FFT // 2

    def ready(self, num_samples: int) -> bool:
        return num_samples >= self.samples_needed()

    def steps_left(self, num_samples: int) -> int:
        """Steps :meth:`finish` would run on a finished ``num_samples``-long recording."""
        num_embeds = math.ceil(num_mel_frames(num_samples) / self.cfg.subsampling)
        return max(0, math.ceil((num_embeds - self.next_frame) / self.cfg.chunk))

    def advance(
        self,
        session: Any,
        read: Reader,
        num_samples: int,
        limit: int | None = None,
        run_options: Any = None,
    ) -> int:
        """Run up to ``limit`` steps that ``num_samples`` of a still-growing
        recording already determine; returns how many ran. A step the session
        raises in (``run_options.terminate``) leaves the loop as it was."""
        ran = 0
        while (limit is None or ran < limit) and self.ready(num_samples):
            self._step(session, read, num_samples, ended=False, run_options=run_options)
            ran += 1
        return ran

    def finish(self, session: Any, read: Reader, num_samples: int) -> np.ndarray:
        """Run the remaining steps of the finished ``num_samples``-long
        recording → (mel frames, num_speakers) probabilities."""
        num_mel = num_mel_frames(num_samples)
        while self.next_frame < math.ceil(num_mel / self.cfg.subsampling):
            self._step(session, read, num_samples, ended=True)
        return _sigmoid(np.concatenate(self._logits, 0)[:num_mel])

    def _step(
        self,
        session: Any,
        read: Reader,
        num_samples: int,
        *,
        ended: bool,
        run_options: Any = None,
    ) -> None:
        cfg = self.cfg
        factor = cfg.subsampling
        start = self.next_frame
        if ended:
            num_mel = num_mel_frames(num_samples)
            num_embeds = math.ceil(num_mel / factor)
        else:  # the recording reaches past this step's right context
            num_mel = None
            num_embeds = start + cfg.chunk + cfg.right
        end = min(start + cfg.chunk, num_embeds)
        context_start = max(start - cfg.left, 0)
        num_lookback = start - context_start
        mel_start, mel_stop = context_start * factor, min(end + cfg.right, num_embeds) * factor
        lo, hi = _sample_span(mel_start, mel_stop, num_samples)
        step_features = log_mel(
            read(lo, hi), mel_start, mel_stop, offset=lo, num_samples=num_samples
        )
        if len(step_features) % factor:  # the reference zero-pads the last frame group
            pad = factor - len(step_features) % factor
            step_features = np.pad(step_features, ((0, pad), (0, 0)))
        cached = self.cache.frames()
        num_step = len(step_features) // factor
        # Per encoder frame, sampled from the mel mask at each group's first
        # frame; only the finished recording's last mel frame is invalid.
        groups = np.arange(context_start, context_start + num_step) * factor
        chunk_mask = (
            np.ones(num_step, np.int64) if num_mel is None else (groups < num_mel - 1)
        ).astype(np.int64)
        step_mask = np.concatenate([np.ones(len(cached), np.int64), chunk_mask])
        logits, chunk_embeds, silence = session.run(
            None,
            {
                "input_features": step_features[None],
                "cached_embeds": cached[None],
                "attention_mask": step_mask[None],
            },
            run_options,
        )
        logits = np.asarray(logits[0], dtype=np.float32)
        step_embeds = np.concatenate([cached, np.asarray(chunk_embeds[0], np.float32)], 0)
        first = (len(cached) + num_lookback) * factor
        try:
            self.cache.update(
                step_embeds,
                logits,
                np.asarray(silence, np.float32),
                end - start,
                num_lookback,
                step_mask,
            )
        except BaseException:
            self.broken = True
            raise
        self._logits.append(logits[first : first + (end - start) * factor])
        self.next_frame = end


def speaker_probabilities(session: Any, cfg: Config, audio: np.ndarray) -> np.ndarray:
    """Offline inference over a whole recording: float32 16 kHz ``audio`` →
    (mel frames, num_speakers) probabilities, one row per 10 ms. ``session`` is
    an ``onnxruntime.InferenceSession`` over the export (tests pass a double)."""
    return ChunkLoop(cfg).finish(session, lambda lo, hi: audio[lo:hi], len(audio))


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


class _Models:
    """The export's config and its two sessions, loaded on first use and
    shared by a diarizer and every stream it opened.

    The offline session runs at ORT's default thread count: at the stop the
    chunk loop is all that is left to wait for, and intra-op threads are its
    only parallelism (each step reads the cache the step before wrote). The
    streaming session runs the machine's background plan
    (:func:`stenograf.cpu.background_plan`): on the efficiency cores, one
    thread on macOS, one per efficiency core elsewhere, its pool threads
    pinned there too. Neither spin-waits: at the same speed, spinning cost 2.2×
    the energy on an M4 Max and 1.3× on a Ryzen AI 9 HX 370 (2026-11-17,
    ``eval/README.md``). Thread count and pinning never change the
    probabilities (measured 1–16 threads on both), so a loop may switch
    sessions between steps."""

    def __init__(self, model_dir: Path | None, progress: assets.ProgressHook | None) -> None:
        self._model_dir = model_dir
        self._progress = progress
        self._lock = threading.Lock()
        self._config: Config | None = None
        self._offline: Any = None
        self._stream: Any = None
        self._stream_users = 0

    def _paths(self, *, fetch: bool) -> tuple[Path, Path] | None:
        if self._model_dir is not None:
            return self._model_dir / "model.onnx", self._model_dir / "config.json"
        if not fetch:
            found = [assets.cached_path(asset) for asset in assets.NEMOTRON]
            if any(path is None for path in found):
                return None
            graph, _, config = found
        else:
            graph, _, config = (assets.fetch(asset, self._progress) for asset in assets.NEMOTRON)
        assert graph is not None and config is not None
        return graph, config

    def config(self, *, fetch: bool = True) -> Config | None:
        """The chunking policy; ``fetch=False`` answers ``None`` rather than
        download the model (a meeting never downloads mid-recording)."""
        if self._config is None:
            paths = self._paths(fetch=fetch)
            if paths is None:
                return None
            with self._lock:
                if self._config is None:
                    self._config = Config.load(paths[1])
        return self._config

    def _session(self, *, stream: bool) -> Any:
        import onnxruntime

        from stenograf import cpu

        paths = self._paths(fetch=True)
        assert paths is not None
        options = cpu.session_options(cpu.background_plan() if stream else None)
        return onnxruntime.InferenceSession(
            str(paths[0]), options, providers=["CPUExecutionProvider"]
        )

    def offline(self) -> tuple[Any, Config]:
        config = self.config()
        assert config is not None
        if self._offline is None:
            with self._lock:
                if self._offline is None:
                    self._offline = self._session(stream=False)
        return self._offline, config

    def acquire_stream(self) -> Any:
        """The streaming session, held until every acquirer released it."""
        with self._lock:
            if self._stream is None:
                self._stream = self._session(stream=True)
            self._stream_users += 1
            return self._stream

    def release_stream(self) -> None:
        with self._lock:
            self._stream_users -= 1
            if self._stream_users <= 0:
                self._stream_users = 0
                self._stream = None


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
        self.embedder = embedder if embedder is not None else VoiceEmbedder(progress=progress)
        self._models = _Models(model_dir, progress)
        self._voice_failure: Exception | None = None

    def probabilities(self, samples: np.ndarray) -> np.ndarray:
        """(10-ms frames, slots) speaker probabilities of mono 16 kHz PCM."""
        session, config = self._models.offline()
        return speaker_probabilities(session, config, to_float32(samples))

    def stream(self) -> NemotronStream:
        return NemotronStream(self)

    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        if len(samples) < HOP:  # no valid mel frame: nothing for the model to score
            return []
        return turns_from_probs(self.probabilities(samples), num_speakers)

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        if len(samples) < HOP:
            return DiarizationResult([], {})
        return self._voiced(self.probabilities(samples), samples, num_speakers, None)

    def _voiced(
        self,
        probs: np.ndarray,
        samples: np.ndarray,
        num_speakers: int | None,
        collector: VoiceCollector | None,
    ) -> DiarizationResult:
        """Turns, and each kept slot's voice from ``collector`` once it has
        seen the rest of ``probs`` (a fresh one sees them all). A voice model
        that cannot load or run (no download, a broken file, an ONNX Runtime
        error) costs the voices, never the turns: the failure is kept for
        :meth:`take_voice_failure`."""
        turns = turns_from_probs(probs, num_speakers)
        try:
            voices = self._slot_voices(probs, to_float32(samples), turns, collector)
        except Exception as exc:  # noqa: BLE001 — naming is lost, separation must not be
            logger.warning("voice embedding failed (%s); speakers kept, not named", exc)
            self._voice_failure = exc
            voices = {}
        return DiarizationResult(turns, {f"S{slot}": voice for slot, voice in voices.items()})

    def _slot_voices(
        self,
        probs: np.ndarray,
        audio: np.ndarray,
        turns: list[SpeakerTurn],
        collector: VoiceCollector | None,
    ) -> dict[int, np.ndarray]:
        embedder = self.embedder
        if collector is None or collector.frames > len(probs):
            collector = embedder.collector(HOP)

        def read(lo: int, hi: int) -> np.ndarray:
            return audio[lo:hi]

        collector.feed(probs[collector.frames :] > 0.5, read, embedder.embed, pool=embedder.pool())
        raw: dict[int, list[tuple[int, int]]] = {}
        for turn in turns:
            span = (round(turn.start * _FRAMES_PER_SECOND), round(turn.end * _FRAMES_PER_SECOND))
            raw.setdefault(int(turn.speaker[1:]), []).append((span[0] * HOP, span[1] * HOP))
        return collector.finish(raw, read, embedder.embed, pool=embedder.pool())

    def take_voice_failure(self) -> Exception | None:
        failure, self._voice_failure = self._voice_failure, None
        return failure

    def channel_embedding(
        self, samples: np.ndarray, turns: list[SpeakerTurn]
    ) -> np.ndarray | None:
        return self.embedder.channel_embedding(samples, turns)


STREAM_TAIL_STEPS = 4
"""Most steps the stop finishes on the streaming session. More means the
stream fell behind, and the offline session's threads win back their load:
one step costs 0.29 s on one P-core against 0.13 s on ORT's default threads
(M4 Max, 2026-11-17)."""


class NemotronStream(DiarizationStream):
    """One channel's :class:`ChunkLoop`, advanced while the meeting records.

    Opening the stream loads the streaming session, on the opening thread —
    the meeting's, at its start. Created on the streamer's background thread
    instead, the session held the interpreter lock for 0.7–1.8 s on an
    efficiency core, stalling capture and captions with it; on the meeting's
    thread it takes 0.12 s, and a step afterwards blocks other threads for
    under 10 ms (M4 Max, 2026-11-17). It loads only from the model cache: a
    meeting never downloads, and without the model the stream does nothing.
    :meth:`finish` runs what is left and falls back to the offline loop from
    the start if the streamed state is unusable, so a stream can only ever
    make the stop faster, never lose its result.

    Each step's speech goes to a :class:`VoiceCollector` in the same wake-up,
    so the speakers' voiceprints are mostly built when the meeting stops. An
    embedding that fails or is abandoned costs only itself: the collector
    keeps what it had, the steps go on, and the stop embeds the rest."""

    def __init__(self, diarizer: NemotronDiarizer) -> None:
        self._diarizer = diarizer
        self._models = diarizer._models
        self._session: Any = None
        self._run_options: Any = None
        self._interrupted = False
        self._voices: VoiceCollector | None = None
        self._voiced_steps = 0  # steps of the loop the collector has consumed
        self._voices_failed = False
        config = self._models.config(fetch=False)
        self._loop = ChunkLoop(config) if config is not None else None
        if self._loop is not None:
            import onnxruntime

            self._run_options = onnxruntime.RunOptions()
            self._session = self._models.acquire_stream()
            with contextlib.suppress(Exception):  # the stop embeds instead
                if diarizer.embedder.load(fetch=False):
                    self._voices = diarizer.embedder.collector(HOP)

    def samples_needed(self) -> int | None:
        if self._loop is None or self._loop.broken or self._session is None:
            return None
        return self._loop.samples_needed()

    def advance(self, read: Reader, num_samples: int, limit: int | None = None) -> int:
        needed = self.samples_needed()
        if self._interrupted or needed is None or num_samples < needed:
            return 0
        assert self._loop is not None
        ran = self._loop.advance(self._session, read, num_samples, limit, self._run_options)
        if ran:
            self._collect_voices(read)
        return ran

    def _collect_voices(self, read: Reader) -> None:
        loop, voices = self._loop, self._voices
        if loop is None or voices is None or self._voices_failed:
            return
        embedder = self._diarizer.embedder
        steps = loop.steps
        try:
            voices.feed(
                loop.probabilities(self._voiced_steps) > 0.5,
                read,
                functools.partial(embedder.embed, run_options=self._run_options),
            )
        except Exception:  # noqa: BLE001 — the collector is as it was; the stop goes on from there
            self._voices_failed = not self._interrupted
            return
        self._voiced_steps = steps

    def take_voices(self) -> VoiceCollector | None:
        """The collector for the audio :meth:`finish` just completed, or
        ``None`` when the stop must embed from the start."""
        voices, self._voices = self._voices, None
        return voices

    def interrupt(self) -> None:
        self._interrupted = True
        if self._run_options is not None:
            self._run_options.terminate = True

    def resumed(self) -> Diarizer:
        return _ResumedNemotron(self)

    def finish(self, audio: np.ndarray) -> np.ndarray:
        """Probabilities of the channel's finished float32 ``audio``."""
        read: Reader = lambda lo, hi: audio[lo:hi]  # noqa: E731
        loop = self._loop
        try:
            # A loop already past the audio's end did not stream this audio.
            if loop is not None and loop.next_frame * loop.cfg.subsampling * HOP <= len(audio):
                session = self._session
                if session is None or loop.steps_left(len(audio)) > STREAM_TAIL_STEPS:
                    self.close()  # the two sessions never stack
                    session = self._models.offline()[0]
                with contextlib.suppress(Exception):  # the offline loop below is the backstop
                    if not loop.broken:
                        return loop.finish(session, read, len(audio))
            self.close()
            self._voices = None  # collected over rows this recomputation does not vouch for
            session, config = self._models.offline()
            return speaker_probabilities(session, config, audio)
        finally:
            self.close()

    def close(self) -> None:
        if self._session is not None:
            self._session = None
            self._models.release_stream()


class _ResumedNemotron(NemotronDiarizer):
    """A diarizer whose probabilities finish one stream's loop; voices and
    sessions are its parent's."""

    def __init__(self, stream: NemotronStream) -> None:
        parent = stream._diarizer
        self.embedder = parent.embedder
        self._models = parent._models
        self._stream = stream
        self._voice_failure = None

    def probabilities(self, samples: np.ndarray) -> np.ndarray:
        return self._stream.finish(to_float32(samples))

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        if len(samples) < HOP:
            return DiarizationResult([], {})
        probs = self.probabilities(samples)
        return self._voiced(probs, samples, num_speakers, self._stream.take_voices())
