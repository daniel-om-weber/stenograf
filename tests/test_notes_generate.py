import json

import pytest

from stenograf.config import Language, MeetingProfile
from stenograf.notes import NotesBackendUnavailableError, NotesGenerationError
from stenograf.notes.generate import MAX_SINGLE_SHOT_CHARS, generate_notes
from stenograf.notes.prompt import NOTES_SCHEMA, build_messages, chunk_entries
from stenograf.transcript import Transcript, TranscriptEntry


def entry(text: str, speaker="Local-1", start=0.0) -> TranscriptEntry:
    return TranscriptEntry(speaker=speaker, text=text, start=start, end=start + 5.0)


def transcript(entries=None, *, title=None, language=Language.GERMAN) -> Transcript:
    return Transcript(
        language=language,
        profile=MeetingProfile(
            title=title, attendee_names=("Anna", "Ben"), glossary=("Stenograf",)
        ),
        entries=entries if entries is not None else [entry("Wir planen das Quartal.")],
    )


NOTES_JSON = json.dumps(
    {
        "title": "Quartalsplanung",
        "summary": "Es wurde das Quartal geplant.",
        "decisions": ["Juli-Release"],
        "action_items": [
            {"task": "Budget entwerfen", "owner": "Anna", "due": None, "timestamp": 12.0},
        ],
        "highlights": [],
        "open_questions": ["Einstellungen in Q3?"],
    }
)


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses=None, available=True):
        self.responses = list(responses) if responses is not None else [NOTES_JSON]
        self.available = available
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def is_available(self) -> bool:
        return self.available

    def complete(self, messages, schema) -> str:
        self.calls.append((messages, schema))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


# ---- prompt -------------------------------------------------------------------


def test_build_messages_injects_context_and_language():
    messages = build_messages(transcript(title="Planung"))
    system = messages[0]["content"]
    assert "German" in system
    assert "Title: Planung" in system
    assert "Anna, Ben" in system
    assert "Stenograf" in system
    assert "never invent" in system.lower() or "Never invent" in system
    assert "Local-1 [0:00]: Wir planen das Quartal." in messages[1]["content"]


def test_build_messages_appends_user_instructions():
    messages = build_messages(transcript(), instructions="Immer Du-Form verwenden.")
    system = messages[0]["content"]
    assert system.index("Never invent") < system.index("Immer Du-Form verwenden.")


def test_chunk_entries_keeps_whole_turns_and_drops_nothing():
    entries = [entry(f"Satz {i} " + "x" * 90, start=float(i)) for i in range(50)]
    chunks = chunk_entries(entries, max_chars=1000)
    assert len(chunks) > 1
    assert [e for chunk in chunks for e in chunk] == entries
    assert all(sum(len(e.text) for e in chunk) <= 1000 for chunk in chunks)


def test_chunk_entries_single_oversized_turn_is_its_own_chunk():
    entries = [entry("kurz"), entry("y" * 5000, start=10.0), entry("auch kurz", start=20.0)]
    chunks = chunk_entries(entries, max_chars=1000)
    assert [e for chunk in chunks for e in chunk] == entries
    assert any(chunk == [entries[1]] for chunk in chunks)


# ---- generate -----------------------------------------------------------------


def test_single_shot_populates_notes_and_provenance():
    backend = FakeBackend()
    notes = generate_notes(transcript(), backend)
    assert notes.title == "Quartalsplanung"
    assert notes.summary == "Es wurde das Quartal geplant."
    assert notes.decisions == ("Juli-Release",)
    assert notes.action_items[0].owner == "Anna"
    assert notes.action_items[0].timestamp == 12.0
    assert notes.provenance.backend == "fake"
    assert notes.provenance.model == "fake-model"
    assert notes.provenance.strategy == "single-shot"
    assert notes.provenance.language == "de"
    assert len(backend.calls) == 1
    assert backend.calls[0][1] is NOTES_SCHEMA


def test_profile_title_wins_over_derived_title():
    notes = generate_notes(transcript(title="Weekly Sync"), FakeBackend())
    assert notes.title == "Weekly Sync"


def test_over_budget_forces_map_reduce():
    turn_chars = 2000
    n_turns = (MAX_SINGLE_SHOT_CHARS // turn_chars) + 5
    entries = [entry("w" * turn_chars, start=float(i)) for i in range(n_turns)]
    backend = FakeBackend()
    notes = generate_notes(transcript(entries), backend)
    assert len(backend.calls) > 2  # at least two map calls + one reduce
    assert notes.provenance.strategy.startswith("map-reduce")
    # The reduce call carries the partial notes, not raw transcript entries.
    reduce_messages = backend.calls[-1][0]
    assert "Portion 1 notes:" in reduce_messages[1]["content"]


def test_fenced_json_is_extracted():
    fenced = f"Here are your notes!\n```json\n{NOTES_JSON}\n```\nAnything else?"
    notes = generate_notes(transcript(), FakeBackend([fenced]))
    assert notes.title == "Quartalsplanung"


def test_garbage_response_raises_generation_error():
    with pytest.raises(NotesGenerationError, match="no JSON object"):
        generate_notes(transcript(), FakeBackend(["I'm afraid I can't do that."]))


def test_missing_required_field_raises():
    with pytest.raises(NotesGenerationError, match="summary"):
        generate_notes(transcript(), FakeBackend(['{"title": "T"}']))


def test_unavailable_backend_raises_before_any_completion():
    backend = FakeBackend(available=False)
    with pytest.raises(NotesBackendUnavailableError):
        generate_notes(transcript(), backend)
    assert backend.calls == []


def test_empty_transcript_is_an_error():
    with pytest.raises(NotesGenerationError, match="no entries"):
        generate_notes(transcript([]), FakeBackend())
