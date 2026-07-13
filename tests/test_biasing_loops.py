"""The biased decode loops, against the library loops they replace.

Neither parakeet-mlx nor onnx-asr exposes a hook into its greedy loop, so we run
our own copy with the boosting tree spliced in. A copy silently rots: if upstream
changes its decoder and we keep decoding the old way, nothing fails — we just stop
matching the library, possibly subtly, possibly only on some audio. So every run
diffs the two loops directly, driving both with the same scripted logits.

``alpha=0`` makes the boost a no-op arithmetically while still exercising every
line of our loop, which is what lets us demand *identical* output.
"""

from __future__ import annotations

import numpy as np
import pytest

from stenograf.asr.biasing import BoostingTree

V = 6  # non-blank vocabulary
BLANK = V
_DURATIONS = [0, 1, 2, 3, 4]


def _tree(phrases=((1, 2),)) -> BoostingTree:
    return BoostingTree(phrases, vocab_size=V, unk_score=0.0)


class _FakeAsr:
    """The onnx-asr surface our loop touches, driven by scripted logits.

    Each encoder frame carries the joint logits (V + 1, blank last) followed by
    the TDT duration for that frame — so a test writes the decoder's decisions
    directly instead of standing up a 2.4 GB model.
    """

    _vocab_size = V + 1
    _blank_idx = BLANK
    _max_tokens_per_step = 3
    use_low_precision = False

    def _create_state(self):
        return (0,)

    def _decode(self, prev_tokens, prev_state, encoder_out):
        logits = encoder_out[: V + 1].astype(np.float32)
        step = int(encoder_out[V + 1])
        return logits, step, (len(prev_tokens),)


def _frames(rows: list[tuple[list[float], int]]) -> tuple[np.ndarray, np.ndarray]:
    encodings = np.array([[*logits, duration] for logits, duration in rows], dtype=np.float32)
    return encodings[None, :, :], np.array([len(rows)])


def _library_decoding(asr, encoder_out, lens):
    from onnx_asr.asr import _AsrWithTransducerDecoding

    return list(_AsrWithTransducerDecoding._decoding(asr, encoder_out, lens))


def _our_decoding(asr, encoder_out, lens, tree, alpha):
    from stenograf.asr.parakeet_onnx import _biased_decoding

    return list(_biased_decoding(asr, tree, alpha)(encoder_out, lens))


# A frame emitting token 1 (duration 1), then a frame where token 3 beats token 2
# narrowly, then blank. The phrase (1, 2) is one token from completing.
_LOGITS = [
    ([0.0, 9.0, 1.0, 0.5, 0.0, 0.0, 0.1], 1),
    ([0.0, 0.5, 2.0, 2.4, 0.0, 0.0, 0.2], 1),
    ([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 9.0], 1),
]


def test_our_onnx_loop_matches_the_library_loop():
    # The drift guard. If onnx-asr changes its greedy loop, this fails.
    asr = _FakeAsr()
    encoder_out, lens = _frames(_LOGITS)
    assert _our_decoding(asr, encoder_out, lens, _tree(), alpha=0.0) == _library_decoding(
        asr, encoder_out, lens
    )


def test_our_onnx_loop_matches_the_library_loop_on_blanks_and_frame_skips():
    # Same guard over the paths the happy case misses: blank emission, a
    # zero-duration frame (which forces the max-symbols escape), a multi-frame skip.
    asr = _FakeAsr()
    encoder_out, lens = _frames(
        [
            ([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 5.0], 1),  # blank
            ([0.0, 4.0, 0.2, 0.3, 0.0, 0.0, 0.1], 0),  # token, no frame advance
            ([0.0, 3.0, 0.2, 0.3, 0.0, 0.0, 0.1], 0),  # token, no frame advance
            ([0.0, 2.0, 0.2, 0.3, 0.0, 0.0, 0.1], 0),  # token, hits max_symbols
            ([0.0, 0.1, 5.0, 0.3, 0.0, 0.0, 0.1], 3),  # token, skips 3 frames
            ([0.0, 0.1, 0.2, 9.0, 0.0, 0.0, 0.1], 1),
        ]
    )
    assert _our_decoding(asr, encoder_out, lens, _tree(), alpha=0.0) == _library_decoding(
        asr, encoder_out, lens
    )


def test_boosting_changes_the_emitted_token():
    # The whole point: token 3 wins on the raw logits, but the decoder is one
    # token into the phrase (1, 2), so boosting completes the phrase instead.
    asr = _FakeAsr()
    encoder_out, lens = _frames(_LOGITS)
    tokens_plain = _library_decoding(asr, encoder_out, lens)[0][0]
    tokens_boosted = _our_decoding(asr, encoder_out, lens, _tree(), alpha=1.0)[0][0]
    assert list(tokens_plain) == [1, 3]
    assert list(tokens_boosted) == [1, 2]


def test_boosting_cannot_conjure_a_token_where_the_model_said_blank():
    # The two-stage selection, which is the property that keeps timestamps sound.
    # Blank wins this frame outright. Even an absurd boost on a phrase token must
    # not steal the frame — biasing re-ranks *within* the labels, it never decides
    # label-vs-blank.
    asr = _FakeAsr()
    encoder_out, lens = _frames([([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 9.0], 1)])
    assert list(_our_decoding(asr, encoder_out, lens, _tree(), alpha=50.0)[0][0]) == []


def test_boosting_changes_which_token_is_emitted_never_whether():
    # The alignment invariant, and the mirror of the test above: a dying partial
    # match carries a large *negative* refund, which must not be able to hand its
    # frame to blank either. Boosting re-ranks within the labels; the label-vs-blank
    # call is always the model's, on untouched logits. So however hard we boost, a
    # token is emitted at exactly the frames the stock decoder emits one — which is
    # what keeps word timestamps, and therefore speaker attribution, sound.
    # (The frames here script their own TDT durations, so equal emission frames mean
    # an equal alignment; against the real model a different token still feeds a
    # different decoder state into the duration head.)
    asr = _FakeAsr()
    encoder_out, lens = _frames(
        [
            ([0.0, 9.0, 1.0, 0.5, 0.0, 0.0, 0.1], 1),  # token 1: enters the phrase
            ([0.0, 0.1, 0.2, 0.3, 0.1, 0.0, 9.0], 1),  # blank
            ([0.0, 0.1, 0.2, 5.0, 0.0, 0.0, 0.1], 1),  # token 3: breaks the phrase
            ([0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 9.0], 1),  # blank
        ]
    )
    plain_tokens, plain_frames, _ = _library_decoding(asr, encoder_out, lens)[0]
    boosted_tokens, boosted_frames, _ = _our_decoding(asr, encoder_out, lens, _tree(), 50.0)[0]

    assert list(boosted_frames) == list(plain_frames)  # same frames emit
    assert list(boosted_tokens) != list(plain_tokens)  # ...but the boost was felt


mlx = pytest.importorskip("mlx.core", reason="MLX backend is Apple Silicon only")


def _fake_mlx_model(rows):
    """A stand-in for ParakeetTDT exposing only what decode_greedy touches."""
    import mlx.core as mx

    class FakeModel:
        vocabulary = [f"p{i}" for i in range(V)]
        durations = _DURATIONS
        time_ratio = 0.08
        max_symbols = 3

        def decoder(self, token, state):
            return mx.zeros((1, 1, 1)), (mx.zeros((1, 1)), mx.zeros((1, 1)))

        def joint(self, feature, decoder_out):
            # feature carries the frame index; look up that row's scripted logits.
            step = int(feature[0, 0, 0].item())
            logits, duration = rows[step]
            durations = [10.0 if d == duration else 0.0 for d in _DURATIONS]
            return mx.array([[[[*logits, *durations]]]])

    return FakeModel()


def test_our_mlx_loop_matches_the_library_loop():
    # The same drift guard for parakeet-mlx.
    import mlx.core as mx
    from parakeet_mlx.parakeet import DecodingConfig, ParakeetTDT

    from stenograf.asr.parakeet import _biased_decode_greedy

    rows = [(logits, duration) for logits, duration in _LOGITS]
    model = _fake_mlx_model(rows)
    features = mx.array([[[float(i)] for i in range(len(rows))]])
    config = DecodingConfig()

    expected, _ = ParakeetTDT.decode_greedy(model, features, config=config)
    ours, _ = _biased_decode_greedy(model, _tree(), 0.0, mx)(features, config=config)

    assert [[t.id for t in hyp] for hyp in ours] == [[t.id for t in hyp] for hyp in expected]
    assert [[t.start for t in hyp] for hyp in ours] == [[t.start for t in hyp] for hyp in expected]


def test_mlx_boosting_changes_the_emitted_token():
    import mlx.core as mx

    from stenograf.asr.parakeet import _biased_decode_greedy

    rows = [(logits, duration) for logits, duration in _LOGITS]
    model = _fake_mlx_model(rows)
    features = mx.array([[[float(i)] for i in range(len(rows))]])

    ours, _ = _biased_decode_greedy(model, _tree(), 1.0, mx)(features, config=None)
    assert [t.id for t in ours[0]] == [1, 2]
