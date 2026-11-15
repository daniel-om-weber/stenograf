"""Nemotron diarizer: the numpy front end against the reference, the count cap,
and the chunk loop's step inputs.

The front end's parity fixture (``tests/data/nemotron_logmel.npz``) holds the
reference processor's output, written by ``eval/nemotron_arm.py fixture``, so
these need no torch. The chunk loop is checked against the held-out set by the
eval harness; here a fake session pins what each step feeds the model. One
real-model test runs only where the assets are cached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stenograf import assets
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.nemotron import (
    HOP,
    NUM_MELS,
    Config,
    NemotronDiarizer,
    log_mel,
    num_mel_frames,
    speaker_probabilities,
    turns_from_probs,
)

FIXTURE = Path(__file__).parent / "data" / "nemotron_logmel.npz"

_CONFIG = Config(
    subsampling=8,
    chunk=4,
    right=2,
    left=0,
    fifo_length=3,
    update_period=3,
    cache_length=16,
    num_silence=1,
    score_threshold=0.25,
    latest_boost=0.05,
    num_speakers=2,
    min_positive=3,
    num_strong=5,
    num_weak=10,
)
"""Tiny chunks so a few seconds of audio cross every cache path (FIFO pop,
compression) without the model."""


def test_log_mel_matches_the_reference_processor():
    fixture = np.load(FIXTURE)
    audio = fixture["pcm"].astype(np.float32) / 32768.0
    got = log_mel(audio)
    want = fixture["features"]
    assert got.shape == want.shape == (num_mel_frames(len(audio)), NUM_MELS)
    # 2026-11-15: 1.4e-4 worst on this fixture (float32 STFT order differences
    # amplified by the log at near-silent bins), most values bit-equal.
    assert np.abs(got - want).max() < 1e-3
    assert np.abs(got - want).mean() < 1e-5
    # The reference masks the last frame and zeroes its features.
    assert fixture["mask"][-1] == 0 and fixture["mask"][:-1].all()
    assert not got[-1].any()


def test_log_mel_by_ranges_equals_one_pass():
    fixture = np.load(FIXTURE)
    audio = fixture["pcm"].astype(np.float32) / 32768.0
    whole = log_mel(audio)
    for size in (1, 7, 40):
        parts = np.concatenate([log_mel(audio, s, s + size) for s in range(0, len(whole), size)])
        # Equal up to BLAS kernel choice (a one-frame range is a matrix-vector product).
        np.testing.assert_allclose(parts, whole, rtol=0, atol=1e-5)
    assert log_mel(audio, len(whole), len(whole) + 5).shape == (0, NUM_MELS)


def test_turns_without_a_count_keep_every_slot_and_overlap():
    probs = np.zeros((10, 3), dtype=np.float32)
    probs[0:4, 0] = 0.9
    probs[3:6, 1] = 0.8  # overlaps slot 0 at frame 3
    probs[8:10, 2] = 0.7
    turns = turns_from_probs(probs)
    assert turns == [
        SpeakerTurn("S0", 0.0, 0.04),
        SpeakerTurn("S1", 0.03, 0.06),
        SpeakerTurn("S2", 0.08, 0.10),
    ]


def test_count_cap_drops_least_active_slots_and_reassigns_orphans():
    probs = np.zeros((12, 3), dtype=np.float32)
    probs[0:6, 0] = 0.9
    probs[0:6, 1] = 0.2
    probs[6:10, 1] = 0.9
    probs[10:12, 2] = 0.9  # the least active slot: dropped by a cap of 2
    probs[10:12, 0] = 0.4  # ... and its frames go to the more probable kept slot
    turns = turns_from_probs(probs, num_speakers=2)
    assert turns == [
        SpeakerTurn("S0", 0.0, 0.06),
        SpeakerTurn("S1", 0.06, 0.10),
        SpeakerTurn("S0", 0.10, 0.12),
    ]
    # A count at or above the slots found changes nothing.
    assert turns_from_probs(probs, num_speakers=3) == turns_from_probs(probs)
    assert turns_from_probs(probs, num_speakers=8) == turns_from_probs(probs)


class _FakeSession:
    """Echoes a deterministic step: logits from the features, encoder frames
    from the chunk, and records every step's inputs."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.steps: list[dict[str, np.ndarray]] = []

    def run(self, outputs, feeds):
        self.steps.append(feeds)
        features = feeds["input_features"][0]
        num_cached = feeds["cached_embeds"].shape[1]
        num_step = len(features) // self.cfg.subsampling
        logits = np.zeros(((num_cached + num_step) * self.cfg.subsampling, 2), np.float32)
        logits[num_cached * self.cfg.subsampling :, 0] = features.mean(1)
        logits[num_cached * self.cfg.subsampling :, 1] = -features.mean(1)
        chunk = np.repeat(features[:: self.cfg.subsampling, :1], 512, axis=1)
        return logits[None], chunk[None], np.ones((1, 512), np.float32)


@pytest.mark.parametrize(
    ("num_samples", "last_masked"),
    [
        (16_483, False),  # 104 mel frames: the last group starts on a valid frame
        (96 * HOP + 50, True),  # 97 mel frames: the last group is the masked frame alone
    ],
)
def test_chunk_loop_masks_per_encoder_frame_and_covers_every_mel_frame(num_samples, last_masked):
    fixture = np.load(FIXTURE)
    audio = fixture["pcm"][:num_samples].astype(np.float32) / 32768.0
    session = _FakeSession(_CONFIG)
    probs = speaker_probabilities(session, _CONFIG, audio)
    assert probs.shape == (num_mel_frames(len(audio)), 2)
    assert np.all((probs >= 0) & (probs <= 1))
    # Every step's mask is per encoder frame (cached + chunk), sampled from the
    # mel mask at each group's first frame — so only a final group holding
    # nothing but the masked last mel frame is masked.
    for feeds in session.steps:
        mask = feeds["attention_mask"][0]
        assert len(mask) == feeds["cached_embeds"].shape[1] + len(feeds["input_features"][0]) // 8
        assert feeds["input_features"].dtype == np.float32
    assert session.steps[-1]["attention_mask"][0][-1] == (0 if last_masked else 1)
    assert all(f["attention_mask"][0][:-1].all() for f in session.steps)
    # The cache filled and was compressed along the way.
    assert max(f["cached_embeds"].shape[1] for f in session.steps) > _CONFIG.fifo_length


class _Embedder:
    def cluster_embeddings(self, turns, samples):
        return {t.speaker: np.ones(2, np.float32) for t in turns}

    def channel_embedding(self, samples, turns):
        return np.ones(2, np.float32) if turns else None


def test_diarizer_pairs_turns_with_the_shared_embedder(monkeypatch):
    d = NemotronDiarizer(embedder=_Embedder())  # pyright: ignore[reportArgumentType]
    probs = np.zeros((300, 8), np.float32)
    probs[10:100, 0] = 0.9
    probs[150:250, 3] = 0.9
    monkeypatch.setattr(d, "probabilities", lambda samples: probs)
    result = d.diarize_with_embeddings(np.zeros(3 * 16000, np.int16))
    assert [t.speaker for t in result.turns] == ["S0", "S3"]
    assert set(result.embeddings) == {"S0", "S3"}
    assert d.diarize(np.zeros(HOP - 1, np.int16)) == []  # no valid frame: no model call


def test_nemotron_assets_share_one_directory():
    parents = {Path(asset.name).parent for asset in assets.NEMOTRON}
    assert len(parents) == 1
    graph, weights, config = assets.NEMOTRON
    assert weights.name == graph.name + "_data"  # the graph names its weights file
    assert Path(config.name).name == "config.json"


@pytest.mark.skipif(
    any(assets.cached_path(a) is None for a in assets.NEMOTRON),
    reason="needs the Nemotron assets cached (steno setup, or one diarization run)",
)
def test_real_model_runs_on_the_fixture():
    fixture = np.load(FIXTURE)
    probs = NemotronDiarizer().probabilities(fixture["pcm"])
    assert probs.shape == (num_mel_frames(len(fixture["pcm"])), 8)
    assert np.all(np.isfinite(probs)) and np.all((probs >= 0) & (probs <= 1))
