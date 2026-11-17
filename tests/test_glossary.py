import random
from difflib import SequenceMatcher

import pytest

from stenograf import glossary as glossary_module
from stenograf.asr.base import Word
from stenograf.glossary import apply_glossary, build_terms
from stenograf.transcript import TranscriptEntry

LOOSE = 0.82
"""A threshold loose enough for these fixtures' near-misses to fire.

Stated explicitly, and deliberately *not* the shipped default: the tests below are
about the mechanism — timing, punctuation, multi-word windows, wordless entries —
and a mechanism test that silently rides on a product default breaks every time the
default is tuned, which tells you nothing. The default's own behaviour is pinned by
``test_default_threshold_leaves_a_distant_near_miss_alone``; 0.82 is what it used to
be, and the eval that moved it is ``eval/bias.py --post``."""


def _w(text: str, start: float) -> Word:
    return Word(text, start, start + 0.4)


def _entry(words: tuple[Word, ...], *, speaker: str = "S0") -> TranscriptEntry:
    return TranscriptEntry(
        speaker=speaker,
        text=" ".join(w.text for w in words),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )


def test_corrects_word_keeping_timing_and_punctuation():
    entry = _entry((_w("wir", 0.0), _w("waren", 0.5), _w("in", 1.0), _w("Grafswald,", 1.5)))
    out = apply_glossary([entry], glossary=["Greifswald"], threshold=LOOSE)[0]
    assert out.text == "wir waren in Greifswald,"
    corrected = out.words[3]
    assert corrected.text == "Greifswald,"  # trailing comma preserved
    assert (corrected.start, corrected.end) == (1.5, 1.9)  # timing untouched
    # Words that did not match are the same objects (no needless churn).
    assert out.words[:3] == entry.words[:3]


def test_corrects_attendee_name_token():
    entry = _entry((_w("danke", 0.0), _w("Danjel", 0.5)))
    out = apply_glossary([entry], attendee_names=["Daniel Weber"], threshold=LOOSE)[0]
    assert out.text == "danke Daniel"


def test_fixes_only_casing_when_already_spelled():
    entry = _entry((_w("aus", 0.0), _w("greifswald", 0.5)))
    out = apply_glossary([entry], glossary=["Greifswald"])[0]
    assert out.words[1].text == "Greifswald"


def test_lowercase_term_does_not_impose_its_case_on_the_transcript():
    """An all-lowercase glossary line asserts spelling, not case.

    Parakeet capitalizes sentence-initial words and German nouns correctly, so
    forcing the term's own case here would *invent* an error out of a spelling the
    user never meant to assert — and it would put this layer at odds with
    `asr.biasing.surface_forms`, which boosts both forms of a lowercase term rather
    than forcing one. The letters get corrected; the capital survives.
    """
    entry = _entry((_w("Kubernets", 0.0), _w("läuft", 0.5)))
    out = apply_glossary([entry], glossary=["kubernetes"], threshold=LOOSE)[0]
    assert out.words[0].text == "Kubernetes"  # not "kubernetes"

    # …and mid-sentence, where the model chose lowercase, it stays lowercase.
    mid = _entry((_w("auf", 0.0), _w("kubernets", 0.5)))
    assert apply_glossary([mid], glossary=["kubernetes"], threshold=LOOSE)[0].words[1].text == (
        "kubernetes"
    )


def test_capitalized_term_is_imposed_verbatim():
    """The converse: capitals in the term ARE the assertion, so they win."""
    entry = _entry((_w("wir", 0.0), _w("nutzen", 0.5), _w("kubernetes", 1.0)))
    out = apply_glossary([entry], glossary=["Kubernetes"])[0]
    assert out.words[2].text == "Kubernetes"


def test_does_not_overcorrect_a_dissimilar_common_word():
    entry = _entry((_w("das", 0.0), _w("ist", 0.5), _w("gut", 1.0)))
    out = apply_glossary([entry], glossary=["Git", "GitHub"])
    assert out[0] is entry  # nothing matched → the very same entry object


def test_wordless_entry_is_corrected_via_text():
    # A Whisper/Voxtral entry carries no word timings — only its text is fixed.
    entry = TranscriptEntry("S0", "wir nutzen Kubernetis heute", 0.0, 3.0)
    out = apply_glossary([entry], glossary=["Kubernetes"], threshold=LOOSE)[0]
    assert out.text == "wir nutzen Kubernetes heute"
    assert out.words == ()


def test_partial_word_coverage_is_not_truncated():
    # Backend emitted fewer words than the text has tokens: text must survive whole.
    entry = TranscriptEntry(
        "S0", "in Grafswald bei allen", 1.0, 3.0, words=(_w("in", 1.0), _w("Grafswald", 1.5))
    )
    out = apply_glossary([entry], glossary=["Greifswald"], threshold=LOOSE)[0]
    assert out.text == "in Greifswald bei allen"
    assert [w.text for w in out.words] == ["in", "Greifswald"]


def test_multiword_term_matches_a_window():
    entry = _entry((_w("die", 0.0), _w("Mett", 0.5), _w("Unie", 1.0), _w("meldete", 1.5)))
    out = apply_glossary([entry], glossary=["Met Uni"], threshold=LOOSE)[0]
    assert out.text == "die Met Uni meldete"


def test_default_threshold_leaves_a_distant_near_miss_alone():
    """The shipped default declines this correction, and that is the point.

    "Grafswald" → "Greifswald" is a real misrecognition a user would want fixed, and
    at the old 0.82 default this layer fixed it. It also, on the same setting,
    rewrote words that were never wrong — U-WER +6.5 % on German and +86 % on
    English, 84 false insertions against decode-time biasing's 3 (``eval/bias.py
    --post``, 2026-07-13). Fuzzy matching answers to no acoustics, so it cannot tell
    the two cases apart; the decoder can, which is why biasing is the layer that
    should be catching this one, upstream, while it can still hear the audio.

    So the default buys precision with recall, deliberately. If this test ever starts
    failing because someone lowered the threshold, the number to look at is not B-WER.
    """
    entry = _entry((_w("wir", 0.0), _w("waren", 0.5), _w("in", 1.0), _w("Grafswald,", 1.5)))
    assert apply_glossary([entry], glossary=["Greifswald"])[0] is entry


def test_threshold_gates_correction():
    entry = _entry((_w("wort", 0.0), _w("Kubernets", 0.5)))
    # A loose threshold snaps the near-miss; a strict one leaves it alone.
    assert apply_glossary([entry], glossary=["Kubernetes"], threshold=0.7)[0].text.endswith(
        "Kubernetes"
    )
    assert apply_glossary([entry], glossary=["Kubernetes"], threshold=0.99)[0] is entry


def test_no_terms_returns_the_same_list():
    entries = [_entry((_w("hallo", 0.0),))]
    assert apply_glossary(entries) is entries


def test_build_terms_dedup_expands_names_and_drops_short():
    terms = build_terms(["Greifswald", "greifswald", "ab"], ["Daniel Weber"])
    norms = sorted(t.norm for t in terms)
    # "Greifswald" once (dedup by key), "ab" dropped (too short), name whole + per token.
    assert norms == ["daniel", "danielweber", "greifswald", "weber"]


def test_first_term_wins_a_tie():
    entry = _entry((_w("abcx", 0.0),))
    # "abcx" scores exactly 0.75 against both terms.
    assert apply_glossary([entry], glossary=["Abcd", "Abce"], threshold=0.7)[0].text == "Abcd"
    assert apply_glossary([entry], glossary=["Abce", "Abcd"], threshold=0.7)[0].text == "Abce"


_ALPHABET = "aaabbcdeäöß"


def _random_entries(rng: random.Random) -> tuple[list[TranscriptEntry], list[str], list[str]]:
    """Near-misses of random multi-token terms over a tiny alphabet, so corrections,
    ties and terms competing for the same window are all common."""

    def token(lo: int, hi: int) -> str:
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(lo, hi)))
        return text.capitalize() if rng.random() < 0.3 else text

    def mutate(tok: str) -> str:
        i = rng.randrange(len(tok) + 1)
        edit = rng.choice(("keep", "delete", "substitute", "insert"))
        if edit == "delete":
            return tok[:i] + tok[i + 1 :] or tok
        if edit == "substitute":
            return tok[:i] + rng.choice(_ALPHABET) + tok[i + 1 :]
        if edit == "insert":
            return tok[:i] + rng.choice(_ALPHABET) + tok[i:]
        return tok

    glossary = [
        " ".join(token(2, 6) for _ in range(rng.choice((1, 1, 2, 3))))
        for _ in range(rng.randint(1, 12))
    ]
    names = [f"{token(3, 6)} {token(3, 6)}" for _ in range(rng.randint(0, 2))]
    vocabulary = [tok for phrase in glossary + names for tok in phrase.split()]
    entries = []
    for _ in range(rng.randint(1, 4)):
        texts = [
            mutate(rng.choice(vocabulary)) if rng.random() < 0.6 else token(1, 7)
            for _ in range(rng.randint(1, 25))
        ]
        texts = [rng.choice(("", "(", "„")) + t + rng.choice(("", "", ",", ".")) for t in texts]
        words = tuple(_w(t, 0.5 * i) for i, t in enumerate(texts))
        text = " ".join(texts)
        if rng.random() < 0.2:
            words = ()  # a wordless backend
        elif rng.random() < 0.2:
            text += f" {mutate(rng.choice(vocabulary))}"  # words cover only part of the text
        entries.append(TranscriptEntry(speaker="S0", text=text, start=0.0, end=1.0, words=words))
    return entries, glossary, names


def _score_every_term(self: glossary_module._Matcher, window: str, size: int):
    best, best_score = None, self.threshold
    for term in self._by_size[size]:
        score = SequenceMatcher(None, window, term.norm).ratio()
        if score >= self.threshold and (best is None or score > best_score):
            best, best_score = term, score
    return best


def test_pruned_search_picks_exactly_what_scoring_every_term_picks(monkeypatch: pytest.MonkeyPatch):
    """The length and character-multiset bounds and the per-window memo are only
    allowed to save work: every text, word and tie must come out as an exhaustive
    ``difflib`` search would have it."""
    cases = []
    for seed in range(150):
        rng = random.Random(seed)
        entries, glossary, names = _random_entries(rng)
        for threshold in (0.5, 0.7, 0.82, 0.9, 0.95):
            pruned = apply_glossary(
                entries, glossary=glossary, attendee_names=names, threshold=threshold
            )
            cases.append((entries, glossary, names, threshold, pruned))
    monkeypatch.setattr(glossary_module._Matcher, "best", _score_every_term)
    fired = 0
    for entries, glossary, names, threshold, pruned in cases:
        exhaustive = apply_glossary(
            entries, glossary=glossary, attendee_names=names, threshold=threshold
        )
        assert pruned == exhaustive, (glossary, names, threshold)
        fired += any(a is not b for a, b in zip(entries, exhaustive, strict=True))
    # Guards the guard: a generator that never triggers a correction proves nothing.
    assert fired > len(cases) // 2
