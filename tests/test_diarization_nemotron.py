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
from stenograf.diarization.embedding import VoiceEmbedder, cluster_embeddings
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
from stenograf.diarization.split import SplitResult, to_span, voice_crop, voice_crops

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
    kind (``"stream"`` or ``"offline"``) each was."""

    def __init__(self) -> None:
        self.made: list[str] = []
        self.runs: dict[str, int] = {}

    def __call__(self, *, stream: bool):
        kind = "stream" if stream else "offline"
        self.made.append(kind)
        sessions = self

        class Session(_FakeSession):
            def run(self, outputs, feeds, run_options=None):
                if run_options is not None and run_options.terminate:
                    raise RuntimeError("Exiting due to terminate flag being set to true.")
                sessions.runs[kind] = sessions.runs.get(kind, 0) + 1
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
    assert sessions.made == ["stream"]  # loaded as the stream opens
    assert _feed(stream, audio, len(audio)) > 0
    turns = stream.resumed().diarize(audio)
    assert turns == turns_from_probs(want)
    assert np.array_equal(stream.finish(audio), want)  # finishing twice: the same
    assert sessions.made == ["stream", "offline"]  # finished offline; the stream session is gone
    assert d._models._stream is None and d._models._stream_users == 0


def test_stream_short_of_its_first_step_finishes_on_its_session(fake_models):
    make, sessions = fake_models
    stream = make().stream()
    audio = _audio(0.4)  # the tiny config's first step needs 48 mel frames, 0.48 s
    assert _feed(stream, audio, len(audio), block=800) == 0
    probs = stream.resumed().probabilities(audio)
    assert np.array_equal(probs, speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio))
    assert sessions.made == ["stream"]  # the whole loop is a tail: no offline session


def test_stream_far_behind_finishes_on_the_offline_session(fake_models):
    make, sessions = fake_models
    stream = make().stream()
    audio = _audio(12.0)
    _feed(stream, audio, 16000)  # the streamer stalled after a second of audio
    left = ChunkLoop(_CONFIG).steps_left(len(audio))
    assert left > STREAM_TAIL_STEPS
    before = sessions.runs.get("stream", 0)
    probs = stream.finish(audio)
    assert np.array_equal(probs, speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio))
    assert sessions.runs.get("stream", 0) == before  # the rest ran offline, from where it stopped
    assert sessions.runs["offline"] == left - before


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
    assert "offline" in sessions.made  # recomputed from the start


def test_stream_never_downloads(monkeypatch):
    d = NemotronDiarizer(embedder=_Embedder())  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(assets, "cached_path", lambda asset: None)
    monkeypatch.setattr(assets, "fetch", lambda *a, **k: pytest.fail("downloaded mid-meeting"))
    stream = d.stream()
    assert stream.samples_needed() is None
    assert stream.advance(lambda lo, hi: np.zeros(hi - lo, np.float32), 10 * 16000) == 0


class _Embedder(VoiceEmbedder):
    """The real rule over a model-free ``embed``: a span's vector is its mean
    sample along x and its length along y, so tests can tell spans apart."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fail = False

    def load(self, *, fetch=True):
        return True

    def embed(self, audio, run_options=None):
        if self.fail or (run_options is not None and run_options.terminate):
            raise RuntimeError("Exiting due to terminate flag being set to true.")
        if len(audio) == 0:
            return None
        self.calls += 1
        return np.array([float(audio.mean()), len(audio) / 16000, 1.0], np.float32)

    def pool(self):
        return None


def _two_voices() -> np.ndarray:
    """12 s the fake session hears as two slots taking 1.5 s turns."""
    rng = np.random.default_rng(3)
    return np.concatenate(
        [(rng.standard_normal(24000) * (30.0, 1e-3)[k % 2]).astype(np.float32) for k in range(8)]
    )


def _same_voices(a, b) -> bool:
    return a.turns == b.turns and all(
        np.array_equal(a.embeddings[k], b.embeddings[k]) for k in a.embeddings | b.embeddings
    )


def test_stream_collects_voices_as_it_goes_and_matches_one_pass(fake_models):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize_with_embeddings(audio)
    assert set(want.embeddings) == {"S0", "S1"}
    d = make()
    stream = d.stream()
    assert _feed(stream, audio, len(audio) - 16000) > 0
    during = d.embedder.calls
    assert during > 0  # embedded while "recording"
    got = stream.resumed().diarize_with_embeddings(audio)
    assert _same_voices(got, want)
    assert d.embedder.calls >= during


def test_failing_embedding_never_stops_the_steps_and_the_stop_recovers(fake_models):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize_with_embeddings(audio)
    d = make()
    stream = d.stream()
    d.embedder.fail = True
    ran = _feed(stream, audio, len(audio) - 16000)
    assert ran > 1 and d.embedder.calls == 0  # steps went on without voices
    d.embedder.fail = False
    assert _same_voices(stream.resumed().diarize_with_embeddings(audio), want)


def test_embedding_abandoned_by_the_stop_is_redone_by_it(fake_models):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize_with_embeddings(audio)
    d = make()
    stream = d.stream()
    _feed(stream, audio, 4 * 16000)
    stream.interrupt()
    stream._collect_voices(lambda lo, hi: audio[lo:hi])  # the call the stop cut short
    assert not stream._voices_failed
    assert _same_voices(stream.resumed().diarize_with_embeddings(audio), want)


def test_recomputed_stream_embeds_from_the_start(fake_models, monkeypatch):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize_with_embeddings(audio)
    stream = make().stream()
    _feed(stream, audio, 4 * 16000)
    stream._loop.broken = True  # pyright: ignore[reportOptionalMemberAccess]
    assert _same_voices(stream.resumed().diarize_with_embeddings(audio), want)
    assert stream.take_voices() is None


def _long_voices() -> np.ndarray:
    """30 s the fake session hears as two slots taking 2.5 s turns."""
    rng = np.random.default_rng(4)
    return np.concatenate(
        [(rng.standard_normal(40000) * (30.0, 1e-3)[k % 2]).astype(np.float32) for k in range(12)]
    )


def _long_crops(audio: np.ndarray) -> set:
    turns = turns_from_probs(speaker_probabilities(_FakeSession(_CONFIG), _CONFIG, audio))
    return {voice_crop(to_span(t.start, t.end)) for t in turns if t.end - t.start >= 2.0}


def _forced_split(seen: list):
    """Stands in for the two-voice check: asks for every speaker's turn
    voices, keeps them, and splits S0's later entries off as S2."""

    def check(turns, segments, audio, *, voices, rerun, num_speakers=None):
        by_label: dict[str, list] = {}
        for t in turns:
            by_label.setdefault(t.speaker, []).append(t)
        seen.append({label: voices(voice_crops(own)) for label, own in by_label.items()})
        own = [i for i, (who, _, _) in enumerate(segments) if who == "S0"]
        late = set(own[len(own) // 2 :])
        speakers = ["S2" if i in late else who for i, (who, _, _) in enumerate(segments)]
        moved = {(segments[i][1], segments[i][2]) for i in late}
        out = [SpeakerTurn("S2", t.start, t.end) if (t.start, t.end) in moved else t for t in turns]
        return SplitResult(speakers, out, {"S2": "S0"}, [])

    return check


def test_stream_embeds_each_long_turn_as_it_finishes(fake_models):
    make, _ = fake_models
    audio = _long_voices()
    d = make()
    stream = d.stream()
    _feed(stream, audio, len(audio) - 16000)
    got = dict(stream._turn_voices)
    crops = _long_crops(audio)
    assert got and set(got) <= crops
    assert len(got) >= len(crops) - 3  # only turns open at the last step, or after it, wait
    for (lo, hi), vector in got.items():
        assert np.array_equal(vector, d.embedder.embed(audio[lo:hi]))


def test_streamed_split_matches_one_pass_and_embeds_little_at_the_stop(fake_models, monkeypatch):
    make, _ = fake_models
    seen: list = []
    monkeypatch.setattr(nemotron, "split_two_voice_speakers", _forced_split(seen))
    audio = _long_voices()
    one = make()
    found = one.diarize_with_embeddings(audio)
    segments = [(t.speaker, t.start, t.end) for t in found.turns]
    want = one.split_speakers(audio, found.turns, segments)
    assert want is not None and want.halves == {"S2": "S0"}
    halves = cluster_embeddings(
        want.turns, audio, one.embedder.embed, rule=one.embedder.rule, only={"S0", "S2"}
    )
    assert set(want.embeddings) == {"S0", "S2"}
    for label in want.embeddings:
        assert np.array_equal(want.embeddings[label], halves[label])

    d = make()
    stream = d.stream()
    _feed(stream, audio, len(audio) - 16000)
    resumed = stream.resumed()
    assert resumed.diarize_with_embeddings(audio).turns == found.turns
    missing = _long_crops(audio) - set(stream._turn_voices)
    assert len(missing) <= 3
    got = resumed.split_speakers(audio, found.turns, segments)
    assert got is not None
    assert (got.speakers, got.turns, got.halves) == (want.speakers, want.turns, want.halves)
    for label in want.embeddings:
        assert np.array_equal(got.embeddings[label], want.embeddings[label])
    offline_voices, streamed_voices = seen
    for label in offline_voices:
        for a, b in zip(offline_voices[label], streamed_voices[label], strict=True):
            assert np.array_equal(a, b)


def test_turn_embeddings_failing_mid_meeting_are_made_at_the_stop(fake_models, monkeypatch):
    make, _ = fake_models
    seen: list = []
    monkeypatch.setattr(nemotron, "split_two_voice_speakers", _forced_split(seen))
    audio = _long_voices()
    one = make()
    found = one.diarize_with_embeddings(audio)
    segments = [(t.speaker, t.start, t.end) for t in found.turns]
    want = one.split_speakers(audio, found.turns, segments)
    d = make()
    stream = d.stream()
    d.embedder.fail = True
    assert _feed(stream, audio, len(audio) - 16000) > 0  # the steps went on
    assert stream._turn_voices_failed and not stream._turn_voices
    d.embedder.fail = False
    resumed = stream.resumed()
    resumed.diarize_with_embeddings(audio)
    got = resumed.split_speakers(audio, found.turns, segments)
    assert got is not None and want is not None and got.speakers == want.speakers
    for a, b in zip(seen[0]["S0"], seen[1]["S0"], strict=True):
        assert np.array_equal(a, b)


class _LoudnessEmbedder(_Embedder):
    """Hears two voices: loud and quiet, as the fake session does."""

    def embed(self, audio, run_options=None):
        if self.fail:
            raise RuntimeError("voice model failed")
        if len(audio) == 0:
            return None
        self.calls += 1
        loud = float(np.sqrt(np.mean(audio**2))) > 1.0
        v = np.array([1.0, 0.0, 0.0] if loud else [0.0, 1.0, 0.0], np.float32)
        v[2] = 1e-3 * (self.calls % 7)
        return v / np.linalg.norm(v)


def _merged_speaker():
    """One speaker, S0, whose first fifteen 4 s turns are loud and last
    fifteen quiet, with one transcript entry per turn."""
    rng = np.random.default_rng(9)
    audio = np.zeros(150 * 16000, np.float32)
    turns = []
    for i in range(30):
        lo = i * 5 * 16000
        audio[lo : lo + 4 * 16000] = rng.standard_normal(4 * 16000) * (30.0 if i < 15 else 1e-3)
        turns.append(SpeakerTurn("S0", i * 5.0, i * 5.0 + 4.0))
    return audio, turns, [(t.speaker, t.start, t.end) for t in turns]


def _splitting(make):
    d = make()
    d.embedder = _LoudnessEmbedder()
    return d


def test_the_rerun_on_the_entries_splits_a_merged_speaker(fake_models):
    make, sessions = fake_models
    audio, turns, segments = _merged_speaker()
    d = _splitting(make)
    got = d.split_speakers(audio, turns, segments)
    assert got is not None and got.halves == {"S1": "S0"}
    assert sessions.runs["offline"] > 0  # the rerun ran on the diarizer
    assert got.speakers == ["S0"] * 15 + ["S1"] * 15
    assert [t.speaker for t in got.turns] == ["S0"] * 15 + ["S1"] * 15
    assert set(got.embeddings) == {"S0", "S1"}
    assert not np.allclose(got.embeddings["S0"], got.embeddings["S1"])


def test_halves_whose_voices_fail_stay_split_and_unnamed(fake_models, monkeypatch):
    make, _ = fake_models
    audio, turns, segments = _merged_speaker()
    d = _splitting(make)

    def fail(*args, **kwargs):
        raise RuntimeError("voice model failed")

    monkeypatch.setattr(nemotron, "cluster_embeddings", fail)
    got = d.split_speakers(audio, turns, segments)
    assert got is not None and got.halves == {"S1": "S0"} and got.embeddings == {}
    assert isinstance(d.take_voice_failure(), RuntimeError)


def test_a_voice_model_that_already_failed_is_not_retried_by_the_check(fake_models):
    make, sessions = fake_models
    audio, turns, segments = _merged_speaker()
    d = _splitting(make)
    d._voice_failure = RuntimeError("no voice model")
    assert d.split_speakers(audio, turns, segments) is None
    assert d.embedder.calls == 0 and "offline" not in sessions.runs


def test_a_failing_two_voice_check_keeps_the_speakers_as_found(fake_models, monkeypatch):
    make, _ = fake_models

    def fail(*args, **kwargs):
        raise RuntimeError("check failed")

    monkeypatch.setattr(nemotron, "split_two_voice_speakers", fail)
    audio = _long_voices()
    d = make()
    found = d.diarize_with_embeddings(audio)
    segments = [(t.speaker, t.start, t.end) for t in found.turns]
    assert d.split_speakers(audio, found.turns, segments) is None
    assert d.take_voice_failure() is None


def test_a_failing_voice_model_keeps_the_turns_and_reports_once(fake_models):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize(audio)
    assert len({t.speaker for t in want}) == 2
    d = make()
    d.embedder.fail = True
    result = d.diarize_with_embeddings(audio)
    assert result.turns == want and result.embeddings == {}
    assert isinstance(d.take_voice_failure(), RuntimeError)
    assert d.take_voice_failure() is None  # told once


def test_a_voice_model_that_cannot_download_keeps_the_turns(fake_models, monkeypatch):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize(audio)
    d = make()
    d.embedder = VoiceEmbedder()  # the real one: its load fetches the model

    def offline(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(assets, "cached_path", lambda asset: None)
    monkeypatch.setattr(assets, "fetch", offline)
    result = d.diarize_with_embeddings(audio)
    assert result.turns == want and result.embeddings == {}
    assert isinstance(d.take_voice_failure(), OSError)


def test_a_voice_model_failing_all_meeting_keeps_the_stop_turns(fake_models):
    make, _ = fake_models
    audio = _two_voices()
    want = make().diarize(audio)
    d = make()
    stream = d.stream()
    d.embedder.fail = True
    _feed(stream, audio, len(audio) - 16000)
    resumed = stream.resumed()
    result = resumed.diarize_with_embeddings(audio)
    assert result.turns == want and result.embeddings == {}
    assert resumed.take_voice_failure() is not None


def test_the_stop_keeps_separation_when_the_voice_model_fails(fake_models):
    from conftest import FakeASR

    from stenograf.asr.base import Word
    from stenograf.audio import to_int16
    from stenograf.capture.base import AudioFrame, Channel
    from stenograf.config import MeetingProfile
    from stenograf.session import MeetingRecorder, SessionStore, plan_channels
    from stenograf.view import LiveView

    class Errors(LiveView):
        def __init__(self) -> None:
            self.errors: list[str] = []

        def error(self, message: str) -> None:
            self.errors.append(message)

    make, _ = fake_models
    store = SessionStore({Channel.SYSTEM})
    store.append(AudioFrame(Channel.SYSTEM, 0.0, to_int16(_two_voices())))
    longest: dict[str, SpeakerTurn] = {}
    for turn in make().diarize(store.samples(Channel.SYSTEM)):
        held = longest.get(turn.speaker)
        if held is None or turn.end - turn.start > held.end - held.start:
            longest[turn.speaker] = turn
    assert len(longest) == 2
    words = tuple(
        Word(f"w{k}", (t.start + t.end) / 2 - 0.005, (t.start + t.end) / 2 + 0.005)
        for k, t in enumerate(sorted(longest.values(), key=lambda t: t.start))
    )
    d = make()
    d.embedder.fail = True
    profile = MeetingProfile(local_speakers=0, remote_speakers=None)
    recorder = MeetingRecorder(profile, asr=FakeASR(), diarizer=d)
    view = Errors()
    result = recorder.finalize(
        store, plan_channels(profile), view=view, live_words={Channel.SYSTEM: words}
    )
    assert len({e.speaker for e in result.transcript.entries}) == 2  # not one fallback speaker
    assert result.speaker_embeddings == {}
    assert len(view.errors) == 1 and "voice embedding failed" in view.errors[0]


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
    assert d.diarize_with_embeddings(np.zeros(HOP - 1, np.int16)).turns == []


def test_a_capped_slot_has_no_voice_and_its_overlap_stays_excluded(monkeypatch):
    d = NemotronDiarizer(embedder=_Embedder())  # pyright: ignore[reportArgumentType]
    probs = np.zeros((400, 8), np.float32)
    probs[0:300, 0] = 0.9
    probs[200:260, 5] = 0.9  # crosstalk over slot 0, removed by the count
    monkeypatch.setattr(d, "probabilities", lambda samples: probs)
    audio = np.zeros(4 * 16000, np.float32)
    audio[200 * HOP : 260 * HOP] = 1.0
    result = d.diarize_with_embeddings(audio, num_speakers=1)
    assert {t.speaker for t in result.turns} == {"S0"}
    assert set(result.embeddings) == {"S0"}
    assert result.embeddings["S0"][0] == 0.0  # nothing of the overlapped stretch


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
