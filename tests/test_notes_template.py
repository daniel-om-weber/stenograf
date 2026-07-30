"""The template module (headings, sections, content) and the markdown unwrap."""

from stenograf.notes.markdown import strip_reasoning, unwrap_markdown
from stenograf.notes.template import DEFAULT_TEMPLATE, content_lines, h1, headings, sections

NOTE = """\
# Title

Summary line.

## Decisions

- one

## Open questions

- any?
"""


# ---- template parsing ----------------------------------------------------------


def test_default_template_headings():
    assert headings(DEFAULT_TEMPLATE) == [
        "Decisions",
        "Action items",
        "Highlights",
        "Open questions",
    ]


def test_headings_skip_fenced_blocks():
    template = "# T\n\n## Real\n\n```\n## Phantom\n# Also phantom\n```\n\n## Also real\n"
    assert headings(template) == ["Real", "Also real"]


def test_h1_is_fence_aware():
    assert h1("```\n# in a fence\n```\n# Real title\n") == "Real title"
    assert h1("no heading here") is None


def test_sections_map_heading_to_lines():
    got = sections(NOTE)
    assert [line for line in got["Decisions"] if line.strip()] == ["- one"]
    assert "Open questions" in got
    assert "Title" not in got  # the H1 is the title slot, not a section


def test_content_lines_ignore_template_echo():
    # A response that is the template verbatim has zero content — the
    # template's own placeholder lines don't count.
    assert content_lines(DEFAULT_TEMPLATE, DEFAULT_TEMPLATE) == []
    assert content_lines(NOTE, DEFAULT_TEMPLATE) == ["Summary line.", "- one", "- any?"]


# ---- reasoning strip -----------------------------------------------------------


def test_strip_reasoning_is_unanchored():
    assert strip_reasoning("a\n<think>x</think>\nb") == "a\n\nb"


def test_strip_reasoning_drops_an_unclosed_block():
    # An unclosed <think> means the output was cut mid-reasoning — everything
    # from it on is reasoning, not note.
    assert strip_reasoning("note text\n<think>never closed").rstrip() == "note text"


# ---- unwrap --------------------------------------------------------------------


def test_unwrap_plain_note():
    title, body = unwrap_markdown(NOTE)
    assert title == "Title"
    assert body.startswith("Summary line.")


def test_unwrap_drops_preamble_before_the_h1():
    title, body = unwrap_markdown("Sure! Here are the notes.\n\n" + NOTE)
    assert title == "Title"
    assert "Sure!" not in body


def test_unwrap_takes_the_fenced_note_and_drops_chatter_both_sides():
    raw = f"Here you go:\n```markdown\n{NOTE}```\nAnything else?"
    title, body = unwrap_markdown(raw)
    assert title == "Title"
    assert "Anything else?" not in body
    assert "```" not in body
    assert "## Decisions" in body


def test_unwrap_survives_an_unclosed_fence():
    raw = f"```markdown\n{NOTE}"  # the closing fence was cut off
    title, body = unwrap_markdown(raw)
    assert title == "Title"
    assert "```" not in body


def test_unwrap_keeps_a_legitimate_code_block_inside_the_note():
    note = "# T\n\nIntro.\n\n## Decisions\n\n```sql\nSELECT 1; -- # not a heading\n```\n\n- done\n"
    title, body = unwrap_markdown(note)
    assert title == "T"
    assert "SELECT 1;" in body
    assert body.count("```") == 2


def test_unwrap_without_any_h1_returns_no_title():
    title, body = unwrap_markdown("## Decisions\n\n- one\n")
    assert title is None
    assert body.startswith("## Decisions")
