"""SentencePiece token → word merging (shared by the MLX and ONNX backends)."""

from stenograf.asr.tokens import Token, merge_tokens


def _texts(tokens):
    return [w.text for w in merge_tokens(tokens)]


def test_merge_tokens_leading_space_starts_a_word():
    toks = [Token(" hal", 0.0, 0.2), Token("lo", 0.2, 0.3), Token(" welt", 0.4, 0.7)]
    assert _texts(toks) == ["hallo", "welt"]


def test_merge_tokens_word_confidence_is_its_weakest_token():
    # A word is only as certain as its least certain piece: averaging would bury the
    # one token the model fumbled under its confident neighbours.
    toks = [Token(" hal", 0.0, 0.2, 0.9), Token("lo", 0.2, 0.3, 0.4), Token(" welt", 0.4, 0.7, 0.8)]
    words = merge_tokens(toks)
    assert [w.confidence for w in words] == [0.4, 0.8]


def test_merge_tokens_missing_confidence_is_absent_not_zero():
    # A backend that reports no confidence must leave it None — not drag the word to 0.
    assert merge_tokens([Token(" hallo", 0.0, 0.2)])[0].confidence is None
    mixed = merge_tokens([Token(" hal", 0.0, 0.2, 0.7), Token("lo", 0.2, 0.3)])
    assert mixed[0].confidence == 0.7


def test_merge_tokens_bare_space_token_is_a_word_boundary():
    # Numbers arrive as a bare " " boundary token followed by digit pieces
    # (real decode of de-2: " und", " ", "1", "5", ".", "7", "."). The empty
    # token has no visible text but must still break the word — this used to
    # render "und15.7.".
    toks = [
        Token(" und", 0.0, 0.2),
        Token(" ", 0.2, 0.25),
        Token("1", 0.25, 0.3),
        Token("5", 0.3, 0.35),
        Token(".", 0.35, 0.4),
        Token("7", 0.4, 0.45),
        Token(".", 0.45, 0.5),
    ]
    merged = merge_tokens(toks)
    assert [w.text for w in merged] == ["und", "15.7."]
    assert merged[1].start == 0.25  # the number's time span, not the space's
    assert merged[1].end == 0.5


def test_merge_tokens_empty_text_token_is_not_a_boundary():
    toks = [Token(" ge", 0.0, 0.1), Token("", 0.1, 0.1), Token("sagt", 0.1, 0.3)]
    assert _texts(toks) == ["gesagt"]
