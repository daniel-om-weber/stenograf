from stenograf.asr.base import Word
from stenograf.glossary import apply_glossary, build_terms
from stenograf.transcript import TranscriptEntry


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
    out = apply_glossary([entry], glossary=["Greifswald"])[0]
    assert out.text == "wir waren in Greifswald,"
    corrected = out.words[3]
    assert corrected.text == "Greifswald,"  # trailing comma preserved
    assert (corrected.start, corrected.end) == (1.5, 1.9)  # timing untouched
    # Words that did not match are the same objects (no needless churn).
    assert out.words[:3] == entry.words[:3]


def test_corrects_attendee_name_token():
    entry = _entry((_w("danke", 0.0), _w("Danjel", 0.5)))
    out = apply_glossary([entry], attendee_names=["Daniel Weber"])[0]
    assert out.text == "danke Daniel"


def test_fixes_only_casing_when_already_spelled():
    entry = _entry((_w("aus", 0.0), _w("greifswald", 0.5)))
    out = apply_glossary([entry], glossary=["Greifswald"])[0]
    assert out.words[1].text == "Greifswald"


def test_does_not_overcorrect_a_dissimilar_common_word():
    entry = _entry((_w("das", 0.0), _w("ist", 0.5), _w("gut", 1.0)))
    out = apply_glossary([entry], glossary=["Git", "GitHub"])
    assert out[0] is entry  # nothing matched → the very same entry object


def test_wordless_entry_is_corrected_via_text():
    # A Whisper/Voxtral entry carries no word timings — only its text is fixed.
    entry = TranscriptEntry("S0", "wir nutzen Kubernetis heute", 0.0, 3.0)
    out = apply_glossary([entry], glossary=["Kubernetes"])[0]
    assert out.text == "wir nutzen Kubernetes heute"
    assert out.words == ()


def test_partial_word_coverage_is_not_truncated():
    # Backend emitted fewer words than the text has tokens: text must survive whole.
    entry = TranscriptEntry(
        "S0", "in Grafswald bei allen", 1.0, 3.0, words=(_w("in", 1.0), _w("Grafswald", 1.5))
    )
    out = apply_glossary([entry], glossary=["Greifswald"])[0]
    assert out.text == "in Greifswald bei allen"
    assert [w.text for w in out.words] == ["in", "Greifswald"]


def test_multiword_term_matches_a_window():
    entry = _entry((_w("die", 0.0), _w("Mett", 0.5), _w("Unie", 1.0), _w("meldete", 1.5)))
    out = apply_glossary([entry], glossary=["Met Uni"])[0]
    assert out.text == "die Met Uni meldete"


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
