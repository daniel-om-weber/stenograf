import json

from stenograf.asr.base import Word
from stenograf.config import Language, MeetingProfile
from stenograf.transcript import Transcript, TranscriptEntry


def make_transcript() -> Transcript:
    return Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(language=Language.GERMAN, local_speakers=1, remote_speakers=2),
        entries=[
            TranscriptEntry(speaker="Local-1", text="Guten Morgen zusammen.", start=0.5, end=2.1),
            TranscriptEntry(
                speaker="Remote-1", text="Morgen!", start=2.0, end=2.6, provisional=True
            ),
        ],
    )


def test_markdown_rendering():
    md = make_transcript().to_markdown()
    assert "**Local-1** [0:00]: Guten Morgen zusammen." in md
    assert "*(overlap)*" in md


def test_json_roundtrip():
    data = json.loads(make_transcript().to_json())
    assert data["language"] == "de"
    assert data["profile"]["remote_speakers"] == 2
    assert len(data["entries"]) == 2
    assert data["entries"][1]["provisional"] is True


def test_json_serializes_word_timestamps():
    transcript = Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(language=Language.GERMAN, local_speakers=1, remote_speakers=1),
        entries=[
            TranscriptEntry(
                speaker="Local-1",
                text="guten morgen",
                start=0.0,
                end=0.9,
                words=(Word("guten", 0.0, 0.4), Word("morgen", 0.5, 0.9)),
            )
        ],
    )
    words = json.loads(transcript.to_json())["entries"][0]["words"]
    assert [w["text"] for w in words] == ["guten", "morgen"]
    assert words[0]["start"] == 0.0 and words[1]["end"] == 0.9


def test_entry_without_words_serializes_empty_list():
    # A wordless backend (Whisper/Voxtral) still produces a valid entry.
    data = json.loads(make_transcript().to_json())
    assert data["entries"][0]["words"] == []


def _long_turn(speaker: str, n: int, *, step: float = 0.6, dur: float = 0.5, t0: float = 0.0):
    words = tuple(Word(f"wort{i}", start=t0 + i * step, end=t0 + i * step + dur) for i in range(n))
    return TranscriptEntry(
        speaker=speaker,
        text=" ".join(w.text for w in words),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )


def _srt_blocks(srt: str) -> list[list[str]]:
    return [block.splitlines() for block in srt.strip().split("\n\n")]


def test_srt_reflows_a_long_turn_into_multiple_cues():
    # 20 words over 12 s must not be one cue — it exceeds the duration budget.
    transcript = Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(language=Language.GERMAN, local_speakers=1),
        entries=[_long_turn("Local-1", 20)],
    )
    blocks = _srt_blocks(transcript.to_srt())
    assert len(blocks) > 1  # re-flowed, not one giant cue
    # Numbered from 1, monotonic; standard SRT `HH:MM:SS,mmm --> HH:MM:SS,mmm`.
    assert [b[0] for b in blocks] == [str(i) for i in range(1, len(blocks) + 1)]
    assert blocks[0][1] == "00:00:00,000 --> 00:00:05,900"
    assert blocks[0][2].startswith("Local-1: ")  # speaker prefix
    # No single line runs past the wrap width.
    assert all(len(line) <= 42 for b in blocks for line in b[2:] if not line.startswith("Local-1"))


def test_vtt_header_voice_tags_and_escaping():
    transcript = Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(language=Language.GERMAN, local_speakers=1, remote_speakers=1),
        entries=[
            TranscriptEntry(
                speaker="Remote-1",
                text="a < b & c",
                start=1.0,
                end=2.5,
                words=(Word("a", 1.0, 1.2), Word("<", 1.3, 1.4), Word("b & c", 1.5, 2.5)),
            )
        ],
    )
    vtt = transcript.to_vtt()
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:01.000 --> 00:00:02.500" in vtt  # dot separator, not comma
    assert "<v Remote-1>" in vtt and "</v>" in vtt
    assert "&lt;" in vtt and "&amp;" in vtt  # VTT-significant chars escaped
    assert "< b &" not in vtt  # ...and the raw forms are gone


def test_wordless_entry_becomes_one_cue_on_its_span():
    # A Whisper/Voxtral entry has no word times: the whole turn is one cue.
    transcript = Transcript(
        language=Language.ENGLISH,
        profile=MeetingProfile(language=Language.ENGLISH, local_speakers=1),
        entries=[
            TranscriptEntry(speaker="Local-1", text="hello there everyone", start=4.0, end=9.0)
        ],
    )
    blocks = _srt_blocks(transcript.to_srt())
    assert len(blocks) == 1
    assert blocks[0][1] == "00:00:04,000 --> 00:00:09,000"
    assert "hello there everyone" in " ".join(blocks[0][2:])


def test_cues_from_multiple_speakers_are_ordered_by_start():
    # Overlapping Local/Remote cues are legal; they interleave by start time.
    transcript = Transcript(
        language=Language.GERMAN,
        profile=MeetingProfile(language=Language.GERMAN, local_speakers=1, remote_speakers=1),
        entries=[
            _long_turn("Local-1", 20),
            TranscriptEntry(
                speaker="Remote-1",
                text="ja genau",
                start=3.0,
                end=3.9,
                words=(Word("ja", 3.0, 3.3), Word("genau", 3.4, 3.9)),
            ),
        ],
    )
    starts = [
        line.split(" --> ")[0] for line in transcript.to_srt().splitlines() if " --> " in line
    ]
    assert starts == sorted(starts)
    assert "00:00:03,000" in starts  # the remote cue sits between the two local ones


def test_timestamp_formats_hours():
    transcript = Transcript(
        language=None,
        profile=MeetingProfile(),
        entries=[TranscriptEntry(speaker="Local-1", text="spät", start=3661.5, end=3661.9)],
    )
    assert "01:01:01,500 --> 01:01:01,900" in transcript.to_srt()
    assert "01:01:01.500 --> 01:01:01.900" in transcript.to_vtt()
