"""Decode-time contextual biasing — a phrase-boosting tree over token logits.

Parakeet has no prompt or hotword parameter, but it does not need one: biasing a
transducer is a *decoder* concern, and both of our decode loops are a few lines of
Python around an ``argmax``. This module is the model-agnostic half — an
Aho-Corasick trie over BPE token ids that answers, for any decode state, "how much
should each token be rewarded right now". The backends supply the logits; see
``stenograf.asr.parakeet``/``parakeet_onnx`` for the loops that call it.

This is a from-scratch implementation of NVIDIA's GPU-PB / TurboBias (Apache-2.0,
arXiv:2508.07014), which is the variant that works under *greedy* decoding — the
one property we cannot give up (beam search costs the live pass its power budget,
and sherpa-onnx's beam decoder silently drops sentences: k2-fsa/sherpa-onnx#3267).
Verified against NeMo's own golden vectors in ``tests/test_biasing.py``.

Two rules carry the whole design, and both are load-bearing:

**Depth-scaled arc scores.** The reward for continuing a phrase grows with depth
(:func:`_token_score`). Greedy keeps exactly one hypothesis, so a flat per-token
bonus is too weak to pull the decoder through a multi-token name before the
evidence for it accumulates. Depth scaling front-loads that pull.

**Telescoping refund on fail arcs.** A partial match that dies must hand its
accumulated bonus back, or the decoder keeps credit for a phrase it never emitted.
The refund is not a separate undo step: it is priced into the backoff weight of
every node (:attr:`_State.backoff_w`), so abandoning a prefix costs exactly what
entering it paid — except at phrase *ends*, whose backoff is zero so a completed
phrase keeps its reward.

The caller must also honour the two-stage token selection documented on
:meth:`BoostingTree.advance`. Skipping it does not merely weaken biasing, it
corrupts the alignment.

What biasing can and cannot reach (measured against the real model, 2026-07-13):
it re-ranks the token the decoder is about to emit, so it fixes a term the model
*nearly* heard — "Grafana-Dashboot" becomes "Grafana-Dashboard" at a boost as low
as 0.5. It cannot conjure a word boundary the model did not hear: where the
decoder glued two words into "Prometheusalord", boosting "Alert" alone changes
nothing, because a completed phrase returns the tree to its root and "Alert" never
starts a word there. Listing the compound — "Prometheus-Alert" — fixes it. Hence
the rule the settings scaffold states: write terms as they appear in the sentence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

DEFAULT_CONTEXT_SCORE = 1.0
"""``c0`` — the reward on a phrase's first token. NeMo's recommended default."""

DEFAULT_DEPTH_SCALING = 2.0
"""``beta`` — how much harder later tokens of a phrase are rewarded. 2.0 is NeMo's
recommendation for RNN-T/TDT models (1.0 is for attention-decoder models)."""

DEFAULT_UNK_SCORE = 1.0
"""Reward for a token that starts no phrase at all.

NeMo's *code* defaults this to 0.0, but the TurboBias paper explicitly recommends
"an unknown score value close to the score of the first transition along the tree"
**for greedy decoding** — which is our case, so we follow the paper, not the code.
With ``unk_score == context_score`` the root's arcs are uniform, so biasing no
longer yanks the decoder toward a glossary term at every word start; only genuine
*continuations* (depth >= 2) keep a differential reward. That is precisely where
biasing disambiguates, and it is the documented cure for the failure mode NVIDIA
warns about: degrading the words that are *not* in the glossary."""

DEFAULT_ALPHA = 1.0
"""Shallow-fusion weight — the knob to tune. NeMo's maintainer reports optimal
greedy values "closer to 1 and even higher"; 0.5 is described as very small.
Above ~3 the boost starts overwriting correctly-recognized words."""

_ROOT = 0


@dataclass
class _Node:
    """A trie node during construction (flattened into :class:`_State` after)."""

    token: int = -1  # -1 marks the root
    token_score: float = 0.0  # reward on the arc *into* this node
    node_score: float = 0.0  # accumulated reward, root -> here
    level: int = 0
    is_end: bool = False  # a phrase terminates here
    next: dict[int, _Node] = field(default_factory=dict)
    fail: _Node | None = None
    state: int = -1  # index assigned during flattening


@dataclass(frozen=True)
class _State:
    """A flattened decode state: where each token leads, and what it pays."""

    arcs: dict[int, tuple[int, float]]  # token -> (next state, reward)
    backoff_to: int
    backoff_w: float


def _token_score(depth: int, context_score: float, depth_scaling: float) -> float:
    """Reward on the arc entering a node at ``depth`` (1-based).

    Flat ``c0`` on a phrase's first token; from the second token on, the reward is
    scaled *and* grows logarithmically with depth, so the deeper the decoder is
    into a phrase, the more it costs to abandon it.
    """
    if depth <= 1:
        return context_score
    return context_score * depth_scaling + math.log(depth)


class BoostingTree:
    """Aho-Corasick phrase-boosting over BPE token ids.

    Build it once per meeting from the glossary (see :func:`build`), then drive it
    from the decode loop: hold one integer state per utterance, call
    :meth:`advance` to get the per-token reward vector, and move the state along
    the token the decoder actually emits.
    """

    def __init__(
        self,
        phrases: Iterable[Sequence[int]],
        *,
        vocab_size: int,
        context_score: float = DEFAULT_CONTEXT_SCORE,
        depth_scaling: float = DEFAULT_DEPTH_SCALING,
        unk_score: float = DEFAULT_UNK_SCORE,
    ) -> None:
        self.vocab_size = vocab_size
        self.unk_score = unk_score
        root = _Node()
        for tokens in phrases:
            self._add(root, tokens, context_score, depth_scaling)
        _link_failures(root)
        self._states = _flatten(root, vocab_size, unk_score)
        # advance() is a pure function of the state and there are only as many
        # states as trie nodes, so memoize: the decode loop then pays one dict
        # lookup per emitted token instead of walking the fail chain.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def num_states(self) -> int:
        return len(self._states)

    def _add(
        self, root: _Node, tokens: Sequence[int], context_score: float, depth_scaling: float
    ) -> None:
        node = root
        for i, token in enumerate(tokens):
            depth = i + 1
            is_end = depth == len(tokens)
            child = node.next.get(token)
            if child is None:
                score = _token_score(depth, context_score, depth_scaling)
                child = _Node(
                    token=token,
                    token_score=score,
                    node_score=node.node_score + score,
                    level=depth,
                    is_end=is_end,
                    fail=root,
                )
                node.next[token] = child
            else:
                # A shared prefix of two phrases. Scores are a pure function of
                # depth, so the existing arc already carries the right reward;
                # only the end-marker can be new (one phrase being a prefix of
                # another, e.g. "Kafka" inside "Kafka Streams").
                child.is_end = child.is_end or is_end
            node = child

    def advance(self, state: int) -> tuple[np.ndarray, np.ndarray]:
        """Rewards and successor states for every non-blank token, from ``state``.

        Returns ``(scores[vocab_size], next_states[vocab_size])``. ``scores`` is
        what to *add* to the token logits; it is negative for tokens that abandon a
        partial match (the refund).

        The caller MUST apply this in two stages, per NVIDIA's algorithm:

        1. Pick ``argmax`` over labels **and blank** on the *unbiased* logits. If
           that is blank, emit blank and leave this tree's state untouched.
        2. Only otherwise, re-pick ``argmax`` over the **non-blank** labels with
           ``logits[:vocab_size] + alpha * scores`` and emit that token.

        Blank is never boosted and never refunded. Folding blank into a single
        biased ``argmax`` instead would let the refund — the only large negative
        score in play — hand frames to blank, dropping tokens and wrecking the
        alignment the whole pipeline hangs its timestamps on.
        """
        cached = self._cache.get(state)
        if cached is not None:
            return cached

        scores = np.zeros(self.vocab_size, dtype=np.float32)
        next_states = np.full(self.vocab_size, -1, dtype=np.int32)
        accumulated = 0.0
        current = state
        while True:
            for token, (to, weight) in self._states[current].arcs.items():
                # Deepest match wins: a token already resolved at a deeper state
                # must not be overwritten by the same token found further up the
                # fail chain.
                if next_states[token] == -1:
                    scores[token] = accumulated + weight
                    next_states[token] = to
            if current == _ROOT:
                break
            accumulated += self._states[current].backoff_w
            current = self._states[current].backoff_to

        # The root arcs cover only phrase-initial tokens; everything else falls
        # back to the root itself and earns the flat unknown-token score.
        unresolved = next_states == -1
        scores[unresolved] = accumulated + self.unk_score
        next_states[unresolved] = _ROOT

        self._cache[state] = (scores, next_states)
        return scores, next_states


def _link_failures(root: _Node) -> None:
    """Aho-Corasick fail links: ``fail(n)`` is the longest proper suffix of n's
    path that is itself a path in the trie (the root when there is none)."""
    root.fail = root
    queue = list(root.next.values())
    for child in queue:
        child.fail = root
    while queue:
        node = queue.pop(0)
        for token, child in node.next.items():
            fail = node.fail
            assert fail is not None
            while token not in fail.next and fail.token != -1:
                assert fail.fail is not None
                fail = fail.fail
            target = fail.next.get(token)
            # A node never fails to itself: that would make the backoff walk
            # in advance() spin forever.
            child.fail = target if target is not None and target is not child else root
            queue.append(child)


def _flatten(root: _Node, vocab_size: int, unk_score: float) -> list[_State]:
    """Number the trie in BFS order and price every arc and backoff."""
    order: list[_Node] = [root]
    root.state = _ROOT
    i = 0
    while i < len(order):
        node = order[i]
        i += 1
        for child in node.next.values():
            child.state = len(order)
            order.append(child)

    states: list[_State] = []
    for node in order:
        arcs = {token: (child.state, child.token_score) for token, child in node.next.items()}
        if node is root:
            # The root answers for the whole vocabulary: any token that starts no
            # phrase is a self-loop paying the flat unknown score. advance() fills
            # those in rather than materializing 8192 arcs per tree.
            states.append(_State(arcs=arcs, backoff_to=_ROOT, backoff_w=0.0))
            continue
        assert node.fail is not None
        # Zero backoff out of a phrase end: the reward for a phrase the decoder
        # actually completed is earned, not lent. Charging the usual refund here
        # would claw it straight back on the next token and the whole boost would
        # net to nothing.
        backoff_w = 0.0 if node.is_end else node.fail.node_score - node.node_score
        states.append(_State(arcs=arcs, backoff_to=node.fail.state, backoff_w=backoff_w))
    return states


def surface_forms(term: str) -> list[str]:
    """The spellings of ``term`` worth boosting.

    Boosting is token-level and Parakeet-v3 is case-sensitive, so "grafana" and
    "Grafana" are different token paths and only the spelling we put in the tree
    is rewarded. German capitalizes nouns and *any* word can start a sentence, so
    an all-lowercase term also needs its capitalized form. A term that already
    carries capitals is left exactly as written: its spelling is deliberate
    (``Kubernetes``, ``iOS``), and "correcting" it would boost a spelling nobody
    uses — ``IOS`` — which is a phrase that can only ever fire wrongly.
    """
    forms = [term]
    if term.islower():
        forms.append(term[:1].upper() + term[1:])
    return forms


def boost_terms(glossary: Iterable[str], attendee_names: Iterable[str] = ()) -> list[str]:
    """Everything worth boosting for a run: the glossary, each attendee's full
    name, and each part of it.

    Names are registered whole *and* per part because a name is usually misheard
    one part at a time — the same reasoning ``stenograf.glossary.build_terms``
    applies to post-correction. Registering the full name too is not redundant:
    a phrase's reward grows with its depth, so "Ada Lovelace" pulls the decoder
    through the surname far harder than "Lovelace" alone can.
    """
    terms: list[str] = [term for term in glossary if term.strip()]
    for name in attendee_names:
        if not name.strip():
            continue
        terms.append(name)
        parts = name.split()
        if len(parts) > 1:
            terms.extend(parts)
    return terms


def build(
    terms: Iterable[str],
    tokenize: Callable[[str], list[list[int]]],
    *,
    vocab_size: int,
    context_score: float = DEFAULT_CONTEXT_SCORE,
    depth_scaling: float = DEFAULT_DEPTH_SCALING,
    unk_score: float = DEFAULT_UNK_SCORE,
) -> BoostingTree | None:
    """Compile glossary terms into a boosting tree, or ``None`` when there is
    nothing to boost — so a meeting without a glossary pays literally nothing.

    ``tokenize`` is the model's own text-to-ids callable, returning every
    tokenization of a term the decoder might emit (word-initial and compound-tail
    — see :func:`stenograf.asr.tokens.load_encoder`). It must be the model's real
    encoder, not an approximation: the trie speaks token ids, so a segmentation
    the decoder never emits is a set of arcs it never walks — biasing would be a
    silent no-op rather than a visible error.
    """
    assert callable(tokenize)
    phrases: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for term in terms:
        for form in surface_forms(term.strip()):
            if not form:
                continue
            for tokenization in tokenize(form):
                tokens = tuple(tokenization)
                if not tokens or tokens in seen:
                    continue
                seen.add(tokens)
                phrases.append(tokens)
    if not phrases:
        return None
    return BoostingTree(
        phrases,
        vocab_size=vocab_size,
        context_score=context_score,
        depth_scaling=depth_scaling,
        unk_score=unk_score,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONTEXT_SCORE",
    "DEFAULT_DEPTH_SCALING",
    "DEFAULT_UNK_SCORE",
    "BoostingTree",
    "boost_terms",
    "build",
    "surface_forms",
]
