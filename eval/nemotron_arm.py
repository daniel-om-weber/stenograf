# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2",
#     "scipy>=1.11",
#     "soundfile>=0.12",
#     "huggingface_hub>=0.30",
#     "onnxruntime==1.31.1",
#     "onnxruntime-ep-webgpu>=0.5.0",
#     "transformers==5.23.0",
#     "torch>=2.6",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
# ///
"""NVIDIA Nemotron 3 Diarization through its ONNX export — the held-out-AMI arm.

Measured 2026-11-12 – 11-15 (x86 + Radeon 890M) and recorded in
`eval/README.md`, "The estimate path, and Nemotron 3 Diarization as the one
diarizer"; what was never observed is in `DECISIONS.md`. This script reproduces that
measurement: it writes per-channel speaker turns (and the raw probabilities the
count-cap score needs) for the channels `eval/heldout.py fetch` built, and
`eval/heldout.py score` scores them next to production's arms.

A self-contained uv script (PEP 723 header above): the reference feature
extractor needs transformers 5.x and torch, which the eval group deliberately
does not carry (it pins transformers < 5 for mlx-lm). ``uv run`` builds the
isolated environment on first use; on Linux torch comes from the CPU index.

The model is ``onnx-community/Nemotron-3-Diarization-ONNX``. That export holds
one chunk step — input features plus cached embeddings in; per-10-ms speaker
logits, the chunk's encoder embeddings and the learned silence embedding out.
The offline chunk loop and the Arrival-Order Speaker Cache around it are a
numpy port of transformers 5.23's ``Nemotron3DiarizationForAudioFrameClassification``
(offline mode) and ``Nemotron3DiarizationSpeakerCache``; ``check`` re-verifies
it against the PyTorch reference (5.6e-6 on CPU and WebGPU, 2026-11-14 —
re-run on any transformers or export change). Three traps that cost time:

- the ONNX ``attention_mask`` is per *encoder* frame of the step (cached +
  chunk), not per mel frame;
- the processor marks the last mel frame invalid; that masks the last encoder
  frame, and ignoring it moves the final chunk's probabilities by up to 0.06;
- the fp16 export keeps fp32 inputs and outputs.

A second WebGPU session in one process segfaulted (onnxruntime-ep-webgpu
0.5.0), so one run uses one session.

Usage::

    uv run eval/nemotron_arm.py check [--device webgpu]     # parity vs PyTorch
    uv run eval/nemotron_arm.py run --device webgpu --precision fp16
    uv run eval/nemotron_arm.py fixture   # the library front end's parity fixture
    uv run eval/nemotron_arm.py features --to DIR   # for heldout.py's --reference-features
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
AUDIO_DIR = EVAL_DIR / "audio" / "ami"
OUT_DIR = EVAL_DIR / "out" / "heldout"
MODEL_DIR = EVAL_DIR / "out" / "models" / "nemotron3-onnx"
ONNX_REPO = "onnx-community/Nemotron-3-Diarization-ONNX"
REFERENCE_REPO = "nvidia/Nemotron-3-Diarization"
MODEL_FILES = {"fp32": "onnx/model.onnx", "fp16": "onnx/model_fp16.onnx"}
FRAME_S = 0.01
"""Output frame stride: one probability row per 10 ms mel frame."""


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Config:
    """The offline chunking and speaker-cache policy, read from the export's config.json."""

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        stream = raw["streaming_config"]
        self.subsampling = raw["audio_config"]["subsampling_factor"]
        self.chunk = raw["chunk_length"]
        self.right = raw["chunk_right_context"]
        self.left = raw.get("chunk_left_context", 0)
        # offline mode overrides the streaming FIFO sizes with the top-level ones
        self.fifo_length = raw["fifo_length"]
        self.update_period = raw["speaker_cache_update_period"]
        self.cache_length = stream["speaker_cache_length"]
        self.num_silence = stream["speaker_cache_silence_frames_per_speaker"]
        self.score_threshold = stream["prediction_score_threshold"]
        self.latest_boost = stream["latest_frames_score_boost"]
        self.num_speakers = stream["num_speakers"]
        budget = self.cache_length // self.num_speakers - self.num_silence
        self.min_positive = math.floor(budget * stream["min_positive_scores_rate"])
        self.num_strong = math.floor(budget * stream["strong_boost_rate"])
        self.num_weak = math.floor(budget * stream["weak_boost_rate"])


class SpeakerCache:
    """Arrival-Order Speaker Cache + FIFO of recent encoder frames (batch size 1)."""

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
        """Push a processed chunk into the FIFO, moving its oldest frames to the cache
        when it overflows (``Nemotron3DiarizationSpeakerCache.update``)."""
        cfg = self.cfg
        num_cache, num_fifo = self.num_cache, self.num_fifo
        steps = logits.shape[0] // cfg.subsampling
        pooled = sigmoid(logits[: steps * cfg.subsampling])
        probs = pooled.reshape(steps, cfg.subsampling, cfg.num_speakers).mean(1)
        probs = (probs * mask[:steps, None]).astype(np.float32)
        start = num_cache + num_fifo + num_lookback
        fifo = np.concatenate([self.fifo[:num_fifo], step_embeds[start : start + num_chunk]], 0)
        popped = self._num_popped(len(fifo))
        if popped:
            fifo_probs = np.concatenate(
                [probs[num_cache : num_cache + num_fifo], probs[start : start + num_chunk]], 0
            )
            # a compressed cache is out of order, so only its stored probabilities are valid;
            # an uncompressed one still holds plain chunk frames this step re-estimated
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
        """Keep the cache_length most important frames, grouped by speaker in original
        order, with silence slots per speaker (``Nemotron3DiarizationSpeakerCache._compress``)."""
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


def load_model(precision: str) -> tuple[Path, Config]:
    """Download into a plain folder, not the HF cache: the cache's per-file symlinks
    resolve the graph and its external weights into different blob directories, and
    onnxruntime 1.31 refuses external data outside the model's own directory."""
    from huggingface_hub import hf_hub_download

    files = [MODEL_FILES[precision], MODEL_FILES[precision] + "_data", "config.json"]
    for name in files:
        hf_hub_download(ONNX_REPO, name, local_dir=MODEL_DIR)
    return MODEL_DIR / MODEL_FILES[precision], Config(MODEL_DIR / "config.json")


def session(model: Path, device: str, threads: int):  # returns onnxruntime.InferenceSession
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    if device == "webgpu":
        import onnxruntime_ep_webgpu as webgpu

        ort.register_execution_provider_library(webgpu.get_ep_name(), webgpu.get_library_path())
        devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name()]
        if not devices:
            raise SystemExit("onnxruntime-ep-webgpu found no GPU adapter")
        options.add_provider_for_devices(devices[:1], {})
        return ort.InferenceSession(str(model), options)
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def speaker_probabilities(
    sess, cfg: Config, features: np.ndarray, mel_mask: np.ndarray
) -> np.ndarray:
    """Offline inference over a whole recording: (mel frames, 128) features and the
    processor's mask → (mel frames, num_speakers) probabilities at 10 ms."""
    factor = cfg.subsampling
    num_mel = len(features)
    num_embeds = math.ceil(num_mel / factor)
    embed_mask = mel_mask.astype(np.int64)[::factor]
    cache = SpeakerCache(cfg)
    chunks = []
    for start in range(0, num_embeds, cfg.chunk):
        end = min(start + cfg.chunk, num_embeds)
        context_start = max(start - cfg.left, 0)
        num_lookback = start - context_start
        step_features = features[context_start * factor : min(end + cfg.right, num_embeds) * factor]
        if len(step_features) % factor:  # the reference zero-pads the last frame group
            pad = factor - len(step_features) % factor
            step_features = np.pad(step_features, ((0, pad), (0, 0)))
        cached = cache.frames()
        num_step = len(step_features) // factor
        step_mask = np.concatenate(
            [np.ones(len(cached), np.int64), embed_mask[context_start : context_start + num_step]]
        )
        logits, chunk_embeds, silence = sess.run(
            None,
            {
                "input_features": step_features[None].astype(np.float32),
                "cached_embeds": cached[None].astype(np.float32),
                "attention_mask": step_mask[None],
            },
        )
        logits = logits[0].astype(np.float32)
        step_embeds = np.concatenate([cached, chunk_embeds[0].astype(np.float32)], 0)
        cache.update(
            step_embeds, logits, silence.astype(np.float32), end - start, num_lookback, step_mask
        )
        first = (len(cached) + num_lookback) * factor
        chunks.append(logits[first : first + (end - start) * factor])
    return sigmoid(np.concatenate(chunks, 0)[:num_mel])


def turns_from(probs: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Speaker segments as the processor's ``extract_speaker_dict`` cuts them (overlap kept)."""
    active = (probs > threshold).astype(np.int8)
    turns = []
    for speaker in range(active.shape[1]):
        edges = np.diff(np.concatenate([[0], active[:, speaker], [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        turns += [
            {"speaker": f"S{speaker}", "start": round(s * FRAME_S, 2), "end": round(e * FRAME_S, 2)}
            for s, e in zip(starts, ends, strict=True)
        ]
    return sorted(turns, key=lambda t: (t["start"], t["speaker"]))


def features_of(processor, wav: Path) -> tuple[np.ndarray, np.ndarray, float]:
    import soundfile as sf

    audio, rate = sf.read(wav, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise SystemExit(f"{wav}: need mono 16 kHz")
    inputs = processor(audio, sampling_rate=rate, return_tensors="np")
    return inputs["input_features"][0], inputs["attention_mask"][0], len(audio) / rate


def cmd_check(args: argparse.Namespace) -> int:
    """Parity of this port against the PyTorch reference on one recording's first minute."""
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(ONNX_REPO)
    features, mask, _ = features_of(processor, args.wav)
    features, mask = features[: args.frames], mask[: args.frames]
    reference = AutoModelForAudioFrameClassification.from_pretrained(REFERENCE_REPO).eval()
    with torch.inference_mode():
        logits = reference(
            input_features=torch.from_numpy(features)[None],
            attention_mask=torch.from_numpy(mask)[None],
        ).logits
    expected = logits.sigmoid()[0].numpy()
    model, cfg = load_model(args.precision)
    got = speaker_probabilities(session(model, args.device, args.threads), cfg, features, mask)
    diff = float(np.abs(got - expected).max())
    flips = int(((got > 0.5) != (expected > 0.5)).sum())
    print(f"{args.precision} {args.device}: max |p - reference| = {diff:.2e}, flips {flips}")
    # fp32 reproduced 5.6e-6 with zero flips; fp16 computes internally in half precision
    return 0 if (diff < 1e-4 or args.precision == "fp16") and flips == 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    from transformers import AutoProcessor

    channels = json.loads((args.out / "channels.json").read_text())
    processor = AutoProcessor.from_pretrained(ONNX_REPO)
    model, cfg = load_model(args.precision)
    sess = session(model, args.device, args.threads)
    tag = f"nemotron_{args.precision}_{args.device}"
    probs_dir = args.out / f"probs_{tag}"
    probs_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for channel in channels:
        cid = channel["id"]
        started = time.perf_counter()
        features, mask, seconds = features_of(processor, args.audio_dir / f"{cid}.wav")
        probs = speaker_probabilities(sess, cfg, features, mask)
        elapsed = time.perf_counter() - started
        np.save(probs_dir / f"{cid}.npy", probs.astype(np.float16))
        result[cid] = {"turns": turns_from(probs), "secs": round(elapsed, 2), "audio_s": seconds}
        found = len({t["speaker"] for t in result[cid]["turns"]})
        print(f"{cid}: {elapsed:.1f} s for {seconds / 60:.1f} min, {found} speakers", flush=True)
        (args.out / f"turns_{tag}.json").write_text(json.dumps(result))
    return 0


FIXTURE = EVAL_DIR.parent / "tests" / "data" / "nemotron_logmel.npz"
FIXTURE_REVISION = "353b6f8ad2cac3580e982d7fbdf0a010786b0406"
"""The export revision ``stenograf.assets.NEMOTRON`` pins."""


def cmd_fixture(args: argparse.Namespace) -> int:
    """The reference processor's features on a deterministic signal — chirp, noise and
    a stretch of digital silence at an odd length, so the masked last frame and the
    log guard are both exercised — for ``tests/test_diarization_nemotron.py``."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ONNX_REPO, revision=FIXTURE_REVISION)
    rng = np.random.RandomState(0)  # the legacy generator's stream is frozen across numpy versions
    n = 16_483
    t = np.arange(n) / 16000
    signal = 0.3 * np.sin(2 * np.pi * (200 + 1500 * t) * t) + 0.05 * rng.standard_normal(n)
    signal[6000:9000] = 0.0
    pcm = np.clip(np.round(signal * 32768), -32768, 32767).astype(np.int16)
    inputs = processor(pcm.astype(np.float32) / 32768.0, sampling_rate=16000, return_tensors="np")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE, pcm=pcm, features=inputs["input_features"][0], mask=inputs["attention_mask"][0]
    )
    print(f"wrote {FIXTURE}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """The reference processor's features per held-out channel (~0.17 GB per audio-hour),
    so the library's chunk loop can be run on exactly what ``run`` fed this one."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ONNX_REPO, revision=FIXTURE_REVISION)
    args.to.mkdir(parents=True, exist_ok=True)
    for channel in json.loads((args.out / "channels.json").read_text()):
        target = args.to / f"{channel['id']}.features.npy"
        if not target.exists():
            features, _, _ = features_of(processor, args.audio_dir / f"{channel['id']}.wav")
            np.save(target, features)
            print(target, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--device", choices=["cpu", "webgpu"], default="cpu")
        p.add_argument("--precision", choices=sorted(MODEL_FILES), default="fp32")
        p.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = ORT)")
    check = sub.choices["check"]
    check.add_argument("--wav", type=Path, default=AUDIO_DIR / "IS1009a.loop.wav")
    check.add_argument("--frames", type=int, default=6001, help="mel frames (10 ms) to compare")
    sub.choices["run"].add_argument("--out", type=Path, default=OUT_DIR)
    sub.choices["run"].add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    sub.add_parser("fixture")
    features = sub.add_parser("features")
    features.add_argument("--to", type=Path, required=True)
    features.add_argument("--out", type=Path, default=OUT_DIR)
    features.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    args = parser.parse_args()
    commands = {"check": cmd_check, "run": cmd_run, "fixture": cmd_fixture}
    return {**commands, "features": cmd_features}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
