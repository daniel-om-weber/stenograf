import pytest

from stenograf.config import Language, MeetingProfile
from stenograf.notes import NotesBackendUnavailableError, NotesGenerationError
from stenograf.notes.generate import generate_notes
from stenograf.notes.prompt import build_messages, chunk_entries, template_instruction
from stenograf.notes.template import DEFAULT_TEMPLATE
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


NOTES_MD = """\
# Quartalsplanung

Es wurde das Quartal geplant.

## Decisions

- Juli-Release

## Action items

- [ ] Budget entwerfen — **Anna** [0:12]

## Highlights

- **Local-1**: Der Juli-Release steht.

## Open questions

- Einstellungen in Q3?
"""

PARTIAL_MD = "- Quartal geplant\n- Anna entwirft das Budget"


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses=None, available=True, max_input_chars=48_000):
        self.responses = list(responses) if responses is not None else [NOTES_MD]
        self.available = available
        self.max_input_chars = max_input_chars
        self.calls: list[list[dict[str, str]]] = []

    def is_available(self) -> bool:
        return self.available

    def complete(self, messages) -> str:
        self.calls.append(messages)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


class RoutingBackend(FakeBackend):
    """Answers map calls with bullets and everything else with the note —
    chunk counts may shift with the template, the routing must not."""

    def complete(self, messages) -> str:
        self.calls.append(messages)
        if "portion of a longer meeting" in messages[0]["content"]:
            return PARTIAL_MD
        return NOTES_MD


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


def test_the_template_rides_last_in_the_user_message():
    # Recency is load-bearing: the format spec must not sit in front of up to
    # 400k chars of transcript, so it is the last thing in the prompt — the
    # seat the old schema instruction had in both grammarless backends.
    messages = build_messages(transcript(), template=DEFAULT_TEMPLATE)
    user = messages[1]["content"]
    assert user.index("Wir planen das Quartal.") < user.index("Fill in this markdown template")
    assert user.rstrip().endswith(DEFAULT_TEMPLATE.rstrip())


def test_a_map_portion_asks_for_bullets_not_the_template():
    # Content in map, shape in reduce: a portion filling the template would
    # give the reduce step N conflicting layouts to un-merge.
    messages = build_messages(transcript(), partial=True, template=None)
    user = messages[1]["content"]
    assert "Fill in this markdown template" not in user
    assert 'lines starting with "- "' in user


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


# ---- generate: the happy paths -------------------------------------------------


def test_single_shot_populates_notes_and_provenance():
    backend = FakeBackend()
    notes = generate_notes(transcript(), backend)
    assert notes.title == "Quartalsplanung"
    assert "Es wurde das Quartal geplant." in notes.body
    assert "## Decisions" in notes.body
    assert "- [ ] Budget entwerfen" in notes.body
    assert not notes.body.startswith("#")  # the H1 left the body — one H1, the renderer's
    assert notes.provenance.backend == "fake"
    assert notes.provenance.model == "fake-model"
    assert notes.provenance.strategy == "single-shot"
    assert notes.provenance.language == "de"
    assert notes.provenance.warnings == ()
    assert len(backend.calls) == 1


def test_profile_title_wins_over_derived_title():
    notes = generate_notes(transcript(title="Weekly Sync"), FakeBackend())
    assert notes.title == "Weekly Sync"


def test_over_backend_budget_forces_map_reduce():
    # The budget is the BACKEND's property — a small-window backend map-reduces
    # a meeting that a frontier backend takes in one pass.
    entries = [entry("w" * 2000, start=float(i)) for i in range(10)]
    backend = RoutingBackend(max_input_chars=8000)
    progress: list[str] = []
    notes = generate_notes(transcript(entries), backend, on_progress=progress.append)
    assert len(backend.calls) > 2  # at least two map calls + one reduce
    assert notes.provenance.strategy.startswith("map-reduce")
    # The reduce call carries the partial notes, not raw transcript entries —
    # and the template arrives only there.
    reduce_messages = backend.calls[-1]
    assert "Portion 1 notes:" in reduce_messages[1]["content"]
    assert "Fill in this markdown template" in reduce_messages[1]["content"]
    # One progress line per model call — a slow run must not look like a hang.
    assert len(progress) == len(backend.calls)
    assert any("portion 1/" in line for line in progress)
    assert "merging" in progress[-1]


def test_same_meeting_is_single_shot_for_a_large_window_backend():
    entries = [entry("w" * 500, start=float(i)) for i in range(10)]
    backend = FakeBackend(max_input_chars=400_000)
    notes = generate_notes(transcript(entries), backend)
    assert len(backend.calls) == 1
    assert notes.provenance.strategy == "single-shot"


# ---- generate: budget ----------------------------------------------------------


def test_system_prompt_counts_against_the_chunk_budget():
    # Same entries, same backend window: a large instructions file must shrink
    # what one chunk may carry — the system prompt does not ride for free.
    entries = [entry("w" * 500, start=float(i)) for i in range(10)]
    lean = RoutingBackend(max_input_chars=9000)
    generate_notes(transcript(entries), lean)
    assert len(lean.calls) == 1  # one pass with a lean system prompt
    fat = RoutingBackend(max_input_chars=9000)
    generate_notes(transcript(entries), fat, instructions="x" * 2500)
    assert len(fat.calls) > 1  # the instructions ate the single-pass budget


def test_oversized_instructions_fail_loudly_before_any_model_call():
    # A system prompt that eats the input window must not degenerate into one
    # model call per speaker turn — it names the cause and stops.
    backend = FakeBackend(max_input_chars=48_000)
    with pytest.raises(NotesGenerationError, match="instructions"):
        generate_notes(transcript(), backend, instructions="x" * 47_000)
    assert backend.calls == []


# ---- generate: the gates that replaced the schema ------------------------------


def test_fenced_response_with_chatter_is_unwrapped():
    # The old {-scan's fenced-JSON case, in markdown: preamble, a fenced note,
    # suffix chatter. Everything outside the fence is not the note.
    fenced = f"Here are your notes!\n```markdown\n{NOTES_MD}```\nAnything else?"
    notes = generate_notes(transcript(), FakeBackend([fenced]))
    assert notes.title == "Quartalsplanung"
    assert "## Decisions" in notes.body
    assert "Anything else?" not in notes.body
    assert "```" not in notes.body


def test_a_think_block_h1_is_not_the_title():
    # The reasoning block is stripped wherever it sits — a `# ` line inside it
    # must not become the note's title (the old strip was \A-anchored).
    wrapped = f"<think>\n# Planungs-Gedanken\nhmm\n</think>\n{NOTES_MD}"
    notes = generate_notes(transcript(), FakeBackend([wrapped]))
    assert notes.title == "Quartalsplanung"
    assert "Gedanken" not in notes.body


def test_a_refusal_is_a_hard_fail_not_a_note():
    with pytest.raises(NotesGenerationError, match="none of the template's headings"):
        generate_notes(transcript(), FakeBackend(["I'm afraid I can't do that."]))


def test_the_template_echoed_back_is_a_hard_fail_not_an_empty_note():
    # Every heading matches, so heading presence alone would pass the very
    # response the gate exists to catch — the silently empty note.
    with pytest.raises(NotesGenerationError, match="template unchanged"):
        generate_notes(transcript(), FakeBackend([DEFAULT_TEMPLATE]))


def test_a_missing_section_is_a_warning_on_the_provenance():
    without_questions = NOTES_MD[: NOTES_MD.index("## Open questions")]
    notes = generate_notes(transcript(), FakeBackend([without_questions]))
    assert 'section "## Open questions" is missing' in notes.provenance.warnings


def test_an_empty_section_is_a_warning_on_the_provenance():
    empty_decisions = NOTES_MD.replace("- Juli-Release\n", "")
    notes = generate_notes(transcript(), FakeBackend([empty_decisions]))
    assert 'section "## Decisions" is empty' in notes.provenance.warnings


def test_a_response_without_a_title_falls_back_with_a_warning():
    headless = NOTES_MD[NOTES_MD.index("## Decisions") :]
    notes = generate_notes(transcript(), FakeBackend([headless]), fallback_title="2026-07-10")
    assert notes.title == "2026-07-10"
    assert any("no title" in w for w in notes.provenance.warnings)


def test_a_refused_map_portion_fails_with_its_portion_number():
    # The map path has no template to validate against; the bullet gate is its
    # refusal detector (weaker than the old JSON parse, and says so in code).
    entries = [entry("w" * 2000, start=float(i)) for i in range(10)]

    class RefusingBackend(FakeBackend):
        def complete(self, messages):
            self.calls.append(messages)
            return "I cannot summarize this portion."

    with pytest.raises(NotesGenerationError, match="portion 1"):
        generate_notes(transcript(entries), RefusingBackend(max_input_chars=8000))


def test_unavailable_backend_raises_before_any_completion():
    backend = FakeBackend(available=False)
    with pytest.raises(NotesBackendUnavailableError):
        generate_notes(transcript(), backend)
    assert backend.calls == []


def test_empty_transcript_is_an_error():
    with pytest.raises(NotesGenerationError, match="no entries"):
        generate_notes(transcript([]), FakeBackend())


def test_template_instruction_forbids_translation():
    # Body language and heading language are decoupled on purpose: a German
    # meeting against the English default template must not hard-fail on
    # translated headings.
    assert "do not translate" in template_instruction(DEFAULT_TEMPLATE)
