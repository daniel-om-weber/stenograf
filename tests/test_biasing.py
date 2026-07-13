"""Boosting-tree unit tests.

The golden values come from NVIDIA NeMo's own boosting-tree test
(``tests/collections/asr/test_boosting_tree.py``, Apache-2.0): phrases ``abc``,
``abd`` and ``c`` over token ids, ``context_score=1.0``, ``depth_scaling=1.0``,
``unk_score=0.0``. Matching them exactly is what says our from-scratch port of
GPU-PB/TurboBias implements the same algorithm and not merely a plausible one —
the refund arithmetic in particular is easy to get subtly, silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from stenograf.asr.biasing import BoostingTree, build, surface_forms

VOCAB = 5
ABC, ABD, C = [1, 2, 3], [1, 2, 4], [3]


def _tree(**kwargs) -> BoostingTree:
    return BoostingTree(
        [ABC, ABD, C],
        vocab_size=VOCAB,
        context_score=1.0,
        depth_scaling=1.0,
        unk_score=0.0,
        **kwargs,
    )


def _walk(tree: BoostingTree, tokens: list[int]) -> list[float]:
    """Score each token as a decoder would: reward from the current state, then
    move to the state that token leads to."""
    state = 0
    scores = []
    for token in tokens:
        reward, next_states = tree.advance(state)
        scores.append(float(reward[token]))
        state = int(next_states[token])
    return scores


def test_tree_has_one_state_per_prefix_plus_root():
    # a, ab, abc, abd, c — shared prefixes are not duplicated — plus the root.
    assert _tree().num_states == 6


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        # Walking "abc" pays 1.0, then 1+ln2, then 1+ln3; leaving the *completed*
        # phrase costs nothing (0.0) — an earned reward is kept, not refunded —
        # and the decoder is back at the root, free to enter "abc" again (1.0).
        ([1, 2, 3, 2, 1], [1.0, 1.6931, 2.0986, 0.0, 1.0]),
        # Tokens that start no phrase are worth the unknown score (0.0 here).
        ([2, 2, 1, 2, 4], [0.0, 0.0, 1.0, 1.6931, 2.0986]),
        # The refund: "ab" accumulated 2.6931, then token 1 breaks the match, so
        # it hands all of it back and pays the fresh start-of-"abc" arc (+1.0).
        ([3, 1, 2, 1], [1.0, 1.0, 1.6931, -1.6931]),
    ],
)
def test_matches_nemo_golden_scores(tokens, expected):
    assert _walk(_tree(), tokens) == pytest.approx(expected, abs=1e-4)


def test_abandoned_prefix_nets_to_zero():
    # The property the refund exists for: a partial match that never completes
    # must leave the decoder exactly where it found it, or the boost would credit
    # a phrase that was never emitted.
    assert sum(_walk(_tree(), [1, 2, 0])) == pytest.approx(0.0, abs=1e-4)


def test_completed_phrase_keeps_its_reward():
    # ...and the mirror image: a phrase the decoder *did* emit stays rewarded.
    assert sum(_walk(_tree(), [1, 2, 3, 0])) == pytest.approx(4.7918, abs=1e-4)


def test_depth_scaling_front_loads_the_pull_through_a_phrase():
    # Greedy holds one hypothesis, so later tokens of a phrase must be rewarded
    # harder than the first, or the decoder abandons a name mid-word.
    scores = _walk(BoostingTree([ABC], vocab_size=VOCAB, unk_score=0.0), ABC)
    assert scores[0] < scores[1] < scores[2]


def test_unknown_score_lifts_every_non_phrase_token():
    # The paper's greedy recommendation: with unk == c0 the root is uniform, so
    # entering a phrase carries no advantage over any other word — only
    # *continuing* one does. This is what keeps biasing from degrading the words
    # that are not in the glossary.
    tree = BoostingTree([ABC], vocab_size=VOCAB, context_score=1.0, unk_score=1.0)
    reward, _ = tree.advance(0)
    assert reward[1] == pytest.approx(reward[0])  # phrase-initial == unknown


def test_blank_is_outside_the_reward_vector():
    # The reward vector spans the non-blank vocabulary only; blank (id == vocab
    # size) has no entry, so a caller cannot boost or refund it by accident.
    reward, next_states = _tree().advance(0)
    assert reward.shape == (VOCAB,)
    assert next_states.shape == (VOCAB,)


def test_a_phrase_that_is_a_prefix_of_another_still_ends():
    # "Kafka" inside "Kafka Streams": completing the short phrase must be
    # rewarded even though its node also continues into the long one.
    tree = BoostingTree([[1, 2], [1, 2, 3]], vocab_size=VOCAB, unk_score=0.0)
    assert sum(_walk(tree, [1, 2, 0])) == pytest.approx(sum(_walk(tree, [1, 2])), abs=1e-4)


def test_build_tokenizes_every_surface_form():
    # Boosting is token-level and the model is case-sensitive, so a lowercase
    # term must also be boosted in its sentence-start spelling.
    seen: list[str] = []

    def tokenize(text: str) -> list[list[int]]:
        seen.append(text)
        return [[len(text)]]

    build(["deployen"], tokenize, vocab_size=VOCAB)
    assert seen == ["deployen", "Deployen"]


def test_build_adds_every_tokenization_of_a_term():
    # A term reaches the decoder in more than one shape — at a word start, and
    # buried in a compound with no word-start marker. Both have to become phrases
    # or the compound (the case German actually gets wrong) is unreachable.
    tree = build(["X"], lambda text: [[1, 2], [3, 4]], vocab_size=VOCAB)
    assert tree is not None
    reward, _ = tree.advance(0)
    assert reward[1] > 0 and reward[3] > 0  # both tokenizations can be entered


def test_build_returns_none_without_terms():
    # No glossary must cost nothing at all — not an empty tree the decode loop
    # would still have to consult on every emitted token.
    assert build([], lambda text: [[1]], vocab_size=VOCAB) is None
    assert build(["  "], lambda text: [[1]], vocab_size=VOCAB) is None


def test_surface_forms_keeps_the_spelling_the_user_wrote():
    assert surface_forms("Kubernetes") == ["Kubernetes"]
    assert surface_forms("iOS") == ["iOS"]
    assert surface_forms("grafana") == ["grafana", "Grafana"]


def test_advance_is_cached_per_state():
    tree = _tree()
    first, _ = tree.advance(0)
    second, _ = tree.advance(0)
    assert first is second  # the decode loop calls this on every emitted token


def test_scores_are_float32_for_the_logits_they_add_to():
    reward, _ = _tree().advance(0)
    assert reward.dtype == np.float32
