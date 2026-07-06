from stenograf.config import Language
from stenograf.lid import detect_language


def test_detects_german_from_function_words():
    text = "Ich glaube, das ist wirklich eine gute Idee und wir sollten das machen."
    assert detect_language(text) == Language.GERMAN


def test_detects_german_from_umlauts_even_with_few_function_words():
    text = "Können wir über die Prüfung sprechen? Das wäre schön."
    assert detect_language(text) == Language.GERMAN


def test_detects_english():
    text = "I think this is really a good idea and we should just do that now."
    assert detect_language(text) == Language.ENGLISH


def test_empty_text_is_undecidable():
    assert detect_language("") is None


def test_too_little_evidence_is_undecidable():
    # A couple of words with no clear markers — not enough to call.
    assert detect_language("okay ja hmm") is None


def test_non_de_en_text_is_undecidable():
    assert detect_language("Foobar 123 xyz qwerty") is None
