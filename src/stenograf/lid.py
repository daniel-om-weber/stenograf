"""Spoken-language identification for stenograf's German/English scope.

Phase 1 resolves the meeting language from the *finalized transcript* with a
function-word + diacritic vote. For a binary de/en decision over
meeting-length text this separates the two languages far more reliably than a
small acoustic LID model would, and it needs no extra model download — the
transcript already exists, because the default ASR (Parakeet) is multilingual
and nothing needs the language before transcription runs.

The plan's acoustic path — LID on the first confident speech segment, locked
before any text exists (sherpa-onnx exposes a Whisper-based
``SpokenLanguageIdentification``) — is the upgrade the live pass needs
(PLAN.md §2, Phase 2) and what a language-*requiring* backend (Whisper/Voxtral)
would use. Either would slot in behind :func:`detect_language`.
"""

from __future__ import annotations

import re

from stenograf.config import Language

# Frequent, low-collision function words. Words that also occur in the other
# language ("will"/"was" are English but common German verbs; German
# "die"/"war"/"man"/"in"/"so"/"hat") are deliberately left out — the aggregate
# vote plus the umlaut/ß signal carries the decision without them.
_GERMAN = frozenset(
    {
        "und", "ist", "nicht", "ich", "auch", "sind", "oder", "aber", "wenn",
        "weil", "dass", "haben", "hatte", "werden", "wird", "wurde", "kann",
        "können", "muss", "müssen", "soll", "jetzt", "hier", "sehr", "mehr",
        "schon", "noch", "eine", "einen", "einem", "eines", "für", "über",
        "unter", "durch", "gegen", "ohne", "zwischen", "während", "vielleicht",
        "genau", "natürlich", "eigentlich", "wirklich",
    }
)  # fmt: skip
_ENGLISH = frozenset(
    {
        "the", "and", "is", "are", "were", "this", "that", "these", "those",
        "with", "from", "have", "has", "had", "would", "could", "should",
        "there", "their", "about", "because", "which", "what", "when",
        "where", "your", "they", "them", "our", "been", "being", "just",
        "really", "actually", "maybe", "something", "anything",
    }
)  # fmt: skip
_GERMAN_CHARS = frozenset("äöüß")
_UMLAUT_WEIGHT = 3
"""An umlaut/ß is a strong German signal (English virtually never has one)."""

_MIN_EVIDENCE = 3
"""Below this many weighted votes the text is too short to call — return None."""

_WORD = re.compile(r"[^\W\d_]+")


def detect_language(text: str) -> Language | None:
    """Classify ``text`` as German or English, or ``None`` if undecidable.

    Undecidable means too little evidence (short/empty text) or a tie (neither
    language, or heavy code-switching). Callers then leave the language unset;
    the transcript is unaffected, since the default ASR model is multilingual.
    """
    tokens = _WORD.findall(text.lower())
    german = sum(token in _GERMAN for token in tokens)
    english = sum(token in _ENGLISH for token in tokens)
    german += _UMLAUT_WEIGHT * sum(char in _GERMAN_CHARS for char in text.lower())
    if german + english < _MIN_EVIDENCE or german == english:
        return None
    return Language.GERMAN if german > english else Language.ENGLISH
