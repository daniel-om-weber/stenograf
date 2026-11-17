"""Nemotron diarizer: the numpy front end against the reference, the count cap,
and the chunk loop's step inputs.

The front end's parity fixture (``tests/data/nemotron_logmel.npz``) holds the
reference processor's output, written by ``eval/nemotron_arm.py fixture``, so
these need no torch. The chunk loop is checked against the held-out set by the
eval harness; here a fake session pins what each step feeds the model, and
that the loop run while audio arrives equals the loop run at the end. One
real-model test runs only where the assets are cached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stenograf import assets
from stenograf.config import MAX_SPEAKERS
from stenograf.diarization import nemotron
from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.nemotron import (
    HOP,
    NUM_MELS,
    STREAM_TAIL_STEPS,
    ChunkLoop,
    Config,
    NemotronDiarizer,
    final_mel_frames,
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

    def run(self, outputs, feeds, run_options=None):
        self.steps.append(feeds)
        features = feeds["input_features"][0]
        num_cached = feeds["cached_embeds"].shape[1]
        num_step = len(features) // self.cfg.subsampling
        logits = np.zeros(((num_cached + num_step) * self.cfg.subsampling, 2), np.float32)
        logits[num_cached * self.cfg.subsampling :, 0] = features.mean(1)
        logits[num_cached * self.cfg.subsampling :, 1] = -features.mean(1)
        chunk = np.repeat(features[:: self.cfg.subsampling, :1], 512, axis=1)
        return logits[None], chunk[None], np.ones(512, np.float32)


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


def _streamed(session, cfg: Config, audio: np.ndarray, sizes) -> tuple[np.ndarray, int]:
    """The loop advanced as ``audio`` arrives in blocks of ``sizes`` samples
    (cycled), then finished; returns the probabilities and the steps streamed."""
    loop = ChunkLoop(cfg)
    read = lambda lo, hi: audio[lo:hi]  # noqa: E731
    n, i, streamed = 0, 0, 0
    while n < len(audio):
        n = min(len(audio), n + sizes[i % len(sizes)])
        i += 1
        streamed += loop.advance(session, read, n)
    return loop.finish(session, read, len(audio)), streamed


@pytest.mark.parametrize(
    "sizes",
    [
        [8000],  # the 0.5 s blocks a meeting's frames add up to
        [HOP * 8 * 4 - 1, HOP * 8 * 4 + 1],  # just under / over one chunk, alternating
        [1],  # every sample: each step runs at the first sample that allows it
        [37, 4999, 160, 1281],
    ],
)
def test_loop_streamed_as_audio_arrives_equals_the_loop_at_the_end(sizes):
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(5 * 16000 + 123) * 0.1).astype(np.float32)
    want = speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio)
    got, streamed = _streamed(_FakeSession(_CONFIG), _CONFIG, audio, sizes)
    assert np.array_equal(got, want)
    # Most steps ran before the end; only the chunks touching it waited.
    total = ChunkLoop(_CONFIG).steps_left(len(audio))
    assert streamed >= total - 2


def test_a_step_runs_exactly_when_its_inputs_are_final():
    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(4 * 16000) * 0.1).astype(np.float32)
    session = _FakeSession(_CONFIG)
    loop = ChunkLoop(_CONFIG)
    needed = loop.samples_needed()
    assert not loop.ready(needed - 1) and loop.ready(needed)
    frames = (_CONFIG.chunk + _CONFIG.right) * _CONFIG.subsampling
    assert final_mel_frames(needed) == frames and final_mel_frames(needed - 1) == frames - 1
    # Run with exactly that much audio and nothing beyond it in memory: the
    # step's features match the offline loop's first step.
    loop.advance(session, lambda lo, hi: audio[:needed][lo:hi], needed)
    offline = _FakeSession(_CONFIG)
    speaker_probabilities(offline, _CONFIG, audio)
    for key in ("input_features", "cached_embeds", "attention_mask"):
        assert np.array_equal(session.steps[0][key], offline.steps[0][key])


class _Sessions:
    """Stands in for onnxruntime: hands out fake sessions, noting which
    kind (``threads``) each was."""

    def __init__(self) -> None:
        self.made: list[int | None] = []
        self.runs: dict[int | None, int] = {}
        self.prepack: dict[int | None, bool] = {}

    def __call__(self, *, threads: int | None, prepack: bool):
        self.made.append(threads)
        self.prepack[threads] = prepack
        sessions = self

        class Session(_FakeSession):
            def run(self, outputs, feeds, run_options=None):
                if run_options is not None and run_options.terminate:
                    raise RuntimeError("Exiting due to terminate flag being set to true.")
                sessions.runs[threads] = sessions.runs.get(threads, 0) + 1
                return super().run(outputs, feeds, run_options)

        return Session(_CONFIG)


@pytest.fixture
def fake_models(monkeypatch):
    sessions = _Sessions()

    def diarizer() -> NemotronDiarizer:
        d = NemotronDiarizer(embedder=_Embedder())  # pyright: ignore[reportArgumentType]
        d._models._config = _CONFIG
        monkeypatch.setattr(d._models, "_session", sessions)
        return d

    return diarizer, sessions


def _audio(seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)


def _feed(stream, audio: np.ndarray, upto: int, block: int = 4000) -> int:
    ran = 0
    for n in range(block, upto + 1, block):
        ran += stream.advance(lambda lo, hi: audio[lo:hi], n)
    return ran


def test_stream_finishes_the_tail_on_its_own_session(fake_models):
    make, sessions = fake_models
    d = make()
    audio = _audio()
    want = speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio)
    stream = d.stream()
    assert sessions.made == [1]  # one thread, loaded as the stream opens
    assert _feed(stream, audio, len(audio)) > 0
    turns = stream.resumed().diarize(audio)
    assert turns == turns_from_probs(want)
    assert np.array_equal(stream.finish(audio), want)  # finishing twice: the same
    assert sessions.made == [1, None]  # ... on the offline session; the stream's is gone
    assert d._models._stream is None and d._models._stream_users == 0
    # Only the meeting-long streaming session drops prepacked weights.
    assert sessions.prepack == {1: False, None: True}


def test_stream_short_of_its_first_step_finishes_on_its_session(fake_models):
    make, sessions = fake_models
    stream = make().stream()
    audio = _audio(0.4)  # the tiny config's first step needs 48 mel frames, 0.48 s
    assert _feed(stream, audio, len(audio), block=800) == 0
    probs = stream.resumed().probabilities(audio)
    assert np.array_equal(probs, speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio))
    assert sessions.made == [1]  # the whole loop is a tail: no offline session


def test_stream_far_behind_finishes_on_the_offline_session(fake_models):
    make, sessions = fake_models
    stream = make().stream()
    audio = _audio(12.0)
    _feed(stream, audio, 16000)  # the streamer stalled after a second of audio
    left = ChunkLoop(_CONFIG).steps_left(len(audio))
    assert left > STREAM_TAIL_STEPS
    before = sessions.runs.get(1, 0)
    probs = stream.finish(audio)
    assert np.array_equal(probs, speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio))
    assert sessions.runs.get(1, 0) == before  # the rest ran offline, from where it stopped
    assert sessions.runs[None] == left - before


def test_interrupted_step_loses_nothing(fake_models):
    make, _ = fake_models
    stream = make().stream()
    audio = _audio()
    _feed(stream, audio, 3 * 16000)
    stream.interrupt()  # the meeting stopped while a step was due
    with pytest.raises(RuntimeError, match="terminate"):
        stream._loop.advance(
            stream._session, lambda lo, hi: audio[lo:hi], len(audio), 1, stream._run_options
        )  # pyright: ignore[reportOptionalMemberAccess]
    assert stream.advance(lambda lo, hi: audio[lo:hi], len(audio)) == 0  # starts no more
    want = speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio)
    assert np.array_equal(stream.finish(audio), want)


def test_broken_stream_falls_back_to_the_offline_loop(fake_models, monkeypatch):
    make, sessions = fake_models
    stream = make().stream()
    audio = _audio()
    _feed(stream, audio, 3 * 16000)

    def fail(*args, **kwargs):
        raise MemoryError

    with monkeypatch.context() as patch, pytest.raises(MemoryError):
        patch.setattr(nemotron.SpeakerCache, "update", fail)
        _feed(stream, audio, len(audio))
    assert stream._loop is not None and stream._loop.broken
    want = speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio)
    assert np.array_equal(stream.finish(audio), want)
    assert None in sessions.made  # recomputed from the start


def test_stream_never_downloads(monkeypatch):
    d = NemotronDiarizer(embedder=_Embedder())  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(assets, "cached_path", lambda asset: None)
    monkeypatch.setattr(assets, "fetch", lambda *a, **k: pytest.fail("downloaded mid-meeting"))
    stream = d.stream()
    assert stream.samples_needed() is None
    assert stream.advance(lambda lo, hi: np.zeros(hi - lo, np.float32), 10 * 16000) == 0


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
    # One column per speaker slot: the settable counts stop at the slot count.
    assert probs.shape == (num_mel_frames(len(fixture["pcm"])), MAX_SPEAKERS)
    assert np.all(np.isfinite(probs)) and np.all((probs >= 0) & (probs <= 1))
