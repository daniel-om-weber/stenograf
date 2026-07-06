import json

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
