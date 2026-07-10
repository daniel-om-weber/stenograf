"""Cross-channel dedup: the backstop for echo the canceller leaves behind."""

from __future__ import annotations

from stenograf.session import drop_echo_duplicates
from stenograf.transcript import TranscriptEntry


def mic(text: str, start: float, end: float) -> TranscriptEntry:
    return TranscriptEntry(speaker="Local-1", text=text, start=start, end=end)


def remote(text: str, start: float, end: float) -> TranscriptEntry:
    return TranscriptEntry(speaker="Remote-1", text=text, start=start, end=end)


class TestRealResidual:
    """The three lines a live speakers-on capture actually leaked past 30 dB of AEC."""

    SYSTEM = [
        remote("Uh when when we say dynamics, uh we are mainly talking about the "
               "motion model for the prediction part of the algorithm.", 8.0, 14.5),
        remote("And there then at the moment we are using some okay, um but we "
               "cannot say sorry.", 15.0, 22.0),
        remote("Yeah, but but that's a good idea.", 23.0, 26.0),
    ]  # fmt: skip

    def test_drops_a_trailing_fragment(self) -> None:
        entries = [mic("prediction part of the algorithm.", 13.0, 14.8)]
        assert drop_echo_duplicates(entries, self.SYSTEM) == []

    def test_drops_despite_asr_disagreement(self) -> None:
        # "we can't not say sorry" vs the remote's "we cannot say sorry"
        entries = [mic("But we can't not say sorry.", 20.0, 22.5)]
        assert drop_echo_duplicates(entries, self.SYSTEM) == []

    def test_drops_a_three_word_fragment(self) -> None:
        entries = [mic("Yeah but but", 23.0, 24.0)]
        assert drop_echo_duplicates(entries, self.SYSTEM) == []


class TestKeepsRealSpeech:
    def test_keeps_the_local_speaker_talking_over_the_remote(self) -> None:
        system = [remote("So the main methodologies can be kept here.", 0.0, 4.0)]
        entries = [mic("Sorry, could you go back one slide?", 1.0, 3.5)]
        assert drop_echo_duplicates(entries, system) == entries

    def test_keeps_short_agreement_even_when_the_remote_says_it_too(self) -> None:
        """'Yeah' on both channels is two people agreeing, not an echo."""
        system = [remote("Yeah.", 5.0, 5.4)]
        entries = [mic("Yeah.", 5.1, 5.5)]
        assert drop_echo_duplicates(entries, system) == entries

    def test_keeps_the_same_words_said_much_later(self) -> None:
        system = [remote("we are using some okay but we cannot say sorry", 10.0, 14.0)]
        entries = [mic("we are using some okay but we cannot say sorry", 60.0, 64.0)]
        assert drop_echo_duplicates(entries, system) == entries

    def test_keeps_everything_with_no_remote_channel(self) -> None:
        entries = [mic("a genuinely local sentence here", 1.0, 3.0)]
        assert drop_echo_duplicates(entries, []) == entries

    def test_keeps_a_partial_quote(self) -> None:
        """Repeating three words of a ten-word remote line is not an echo of it."""
        system = [remote("the motion model for the prediction part of the algorithm", 8.0, 14.0)]
        entries = [mic("the motion model, right, and then what about latency", 9.0, 13.0)]
        assert drop_echo_duplicates(entries, system) == entries


class TestChanceSubsequence:
    """Short local lines vs a long remote monologue — the measured data-loss bug.

    Almost any generic utterance is a chance character-subsequence of a long
    enough remote line: normalized by the mic line's length alone, these scored
    0.80–0.95 and were destroyed. Their matches scatter across the monologue,
    though, while a real echo lands in one dense stretch — which is what the
    span denominator in ``_covered_by`` measures.
    """

    MONOLOGUE = [
        remote(
            "And I think what we should probably do here is to go through the "
            "notes from the last meeting first, because there were some open "
            "questions about who is responsible for the deployment and whether "
            "we can move the date, and I don't want us to lose track of that "
            "again like we did last time.",
            10.0,
            30.0,
        )
    ]

    def test_keeps_generic_lines_that_chance_match_a_monologue(self) -> None:
        entries = [
            mic("No, I don't think so.", 12.0, 13.5),
            mic("Yeah, I think so.", 15.0, 16.0),
            mic("Who is taking notes today?", 18.0, 19.5),
            mic("Sorry, could you go back one slide?", 22.0, 24.0),
        ]
        assert drop_echo_duplicates(entries, self.MONOLOGUE) == entries

    def test_still_drops_an_echo_of_one_sentence_inside_the_monologue(self) -> None:
        """The density denominator must not over-correct: an echo that is a
        contiguous slice of a longer remote line is still an echo."""
        entries = [
            mic("because there were some open questions about who is responsible", 14.0, 18.0)
        ]
        assert drop_echo_duplicates(entries, self.MONOLOGUE) == []


class TestWindow:
    def test_respects_the_overlap_window(self) -> None:
        system = [remote("we are mainly talking about the motion model", 0.0, 4.0)]
        near = [mic("we are mainly talking about the motion model", 5.0, 9.0)]
        far = [mic("we are mainly talking about the motion model", 20.0, 24.0)]
        assert drop_echo_duplicates(near, system) == []  # within 2 s of the remote span
        assert drop_echo_duplicates(far, system) == far  # far outside it

    def test_preserves_order_of_survivors(self) -> None:
        system = [remote("the prediction part of the algorithm", 8.0, 14.0)]
        entries = [
            mic("first local sentence", 1.0, 2.0),
            mic("the prediction part of the algorithm", 13.0, 14.0),
            mic("second local sentence", 20.0, 21.0),
        ]
        kept = drop_echo_duplicates(entries, system)
        assert [e.text for e in kept] == ["first local sentence", "second local sentence"]
