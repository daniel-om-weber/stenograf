"""Unit tests for the corpus reference builder (eval/ami.py) — pure parts only.

The interval math, channel mixing, and annotation parsing are what turn corpus
files into references; a bug here silently corrupts every downstream number, so
they are pinned against hand-built inputs. Network fetching and audio synthesis
are exercised by running the harness itself, not here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from ami import (  # noqa: E402
    ICSI_SESSIONS,
    MrtMeeting,
    apply_mask,
    crosstalk_masks,
    dominant_speaker,
    icsi_speakers,
    merge_spans,
    mix_pcm,
    parse_meetings_xml,
    parse_mrt,
    parse_words_xml,
)
from rttm import Turn  # noqa: E402


class TestMergeSpans:
    def test_gap_under_threshold_merges(self):
        assert merge_spans([(0.0, 1.0), (1.25, 2.0)], gap=0.3) == [(0.0, 2.0)]

    def test_gap_over_threshold_stays_split(self):
        assert merge_spans([(0.0, 1.0), (1.31, 2.0)], gap=0.3) == [(0.0, 1.0), (1.31, 2.0)]

    def test_unsorted_and_contained_spans(self):
        assert merge_spans([(5.0, 6.0), (0.0, 4.0), (1.0, 2.0)], gap=0.3) == [
            (0.0, 4.0),
            (5.0, 6.0),
        ]

    def test_empty(self):
        assert merge_spans([]) == []


class TestMixPcm:
    def test_sum_without_clipping_is_exact(self):
        a = np.array([100, -200, 300], dtype=np.int16)
        b = np.array([1, 2, 3], dtype=np.int16)
        assert mix_pcm([a, b]).tolist() == [101, -198, 303]

    def test_shorter_channels_are_zero_padded(self):
        a = np.array([10, 20, 30], dtype=np.int16)
        b = np.array([1], dtype=np.int16)
        assert mix_pcm([a, b]).tolist() == [11, 20, 30]

    def test_clipping_rescales_to_peak(self):
        a = np.array([30000, 0], dtype=np.int16)
        b = np.array([30000, 15000], dtype=np.int16)
        mixed = mix_pcm([a, b])
        assert mixed[0] == 32767  # the 60000 peak lands exactly at full scale
        assert mixed[1] == round(15000 * 32767 / 60000)

    def test_negative_peak_clipping(self):
        a = np.array([-30000], dtype=np.int16)
        b = np.array([-30000], dtype=np.int16)
        assert mix_pcm([a, b])[0] == -32767


def _tone(windows: list[float], win: int = 800) -> np.ndarray:
    """Per-window constant-amplitude signal (one value per 50 ms window)."""
    rng = np.repeat(np.array(windows), win)
    return (rng * np.sin(np.arange(len(rng)) * 0.5)).astype(np.int16)


class TestCrosstalkMasks:
    def test_bleed_is_gated_and_own_speech_kept(self):
        # ch0 speaks in windows 0-3 (loud) with -15 dB bleed on ch1; ch1
        # speaks in windows 10-13. Hangover keeps ±2 windows around speech.
        ch0 = _tone([20000] * 4 + [0] * 6 + [3500] * 4 + [0] * 4)
        ch1 = _tone([3500] * 4 + [0] * 6 + [20000] * 4 + [0] * 4)
        m0, m1 = crosstalk_masks([ch0, ch1])
        assert m0[:4].all() and not m1[:4].any()  # ch0's speech: ch1 gated
        assert m1[10:14].all() and not m0[10:14].any()
        assert not m0[7].any() and not m1[7].any()  # silence gated everywhere

    def test_overlap_survives_on_both_channels(self):
        # Both channels near their own typical level in windows 0-3.
        ch0 = _tone([18000] * 4 + [0] * 4 + [20000] * 4)
        ch1 = _tone([15000] * 4 + [0] * 4 + [16000] * 4)
        m0, m1 = crosstalk_masks([ch0, ch1])
        assert m0[:4].all() and m1[:4].all()

    def test_quiet_talker_is_not_gated_by_a_loud_neighbor(self):
        # ch1's voice is intrinsically 4x quieter; normalization must keep it.
        ch0 = _tone([20000] * 4 + [0] * 4 + [1200] * 4)
        ch1 = _tone([1200] * 4 + [0] * 4 + [5000] * 4)
        _, m1 = crosstalk_masks([ch0, ch1])
        assert m1[8:12].all()

    def test_apply_mask_zeroes_only_gated_windows(self):
        pcm = _tone([1000] * 4)
        mask = np.array([True, False, True, False])
        gated = apply_mask(pcm, mask)
        assert not gated[800:1600].any() and not gated[2400:3200].any()
        assert (gated[:800] == pcm[:800]).all()


class TestDominantSpeaker:
    def test_majority_overlap_wins(self):
        ref = [Turn("A", 0, 10), Turn("B", 10, 20)]
        cluster = [Turn("S0", 8, 14)]  # 2 s on A, 4 s on B
        assert dominant_speaker(cluster, ref) == "B"

    def test_no_overlap_is_none(self):
        assert dominant_speaker([Turn("S0", 30, 40)], [Turn("A", 0, 10)]) is None

    def test_accumulates_across_turns(self):
        ref = [Turn("A", 0, 10), Turn("B", 10, 20)]
        cluster = [Turn("S0", 0, 3), Turn("S0", 9, 12)]  # 4 s on A, 2 s on B
        assert dominant_speaker(cluster, ref) == "A"


MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting nite:id="meet_41" type="scenario" observation="ES2003a" duration="1242">
    <speaker nite:id="ES2003a_2" channel="1" nxt_agent="B" global_name="FEE005" role="PM"/>
    <speaker nite:id="ES2003a_1" channel="0" nxt_agent="A" global_name="MEE006" role="ID"/>
  </meeting>
</nite:root>
"""


WORDS_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="ES2003a.A.words0" starttime="77.44" endtime="77.74">Hi</w>
  <w nite:id="ES2003a.A.words1" starttime="77.74" endtime="77.74" punc="true">,</w>
  <vocalsound nite:id="ES2003a.A.words2" starttime="80.0" endtime="81.0" type="laugh"/>
  <w nite:id="ES2003a.A.words3" starttime="82.0" endtime="82.5">there</w>
  <w nite:id="ES2003a.A.words4" starttime="" endtime="">um</w>
</nite:root>
"""


MRT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bmr021">
  <Preamble>
    <Channels>
      <Channel Name="chan0" Mic="c1" AudioFile="chan0.sph"/>
      <Channel Name="chan3" Mic="c2" AudioFile="chan3.sph"/>
    </Channels>
    <Participants>
      <Participant Name="me011" Channel="chan0" Seat="9"/>
      <Participant Name="fe008" Channel="chan3" Seat="2"/>
      <Participant Name="mn055" Seat="4"/>
    </Participants>
  </Preamble>
  <Transcript>
    <Segment StartTime="0.530" EndTime="1.430" Participant="me011">We're on.</Segment>
    <Segment StartTime="2.000" EndTime="3.000" Participant="fe008">Yes.</Segment>
    <Segment StartTime="4.000" EndTime="5.000" Participant="mn055">Off-mic.</Segment>
    <Segment StartTime="6.000" EndTime="6.500">Unattributed noise.</Segment>
    <Segment StartTime="7.000" EndTime="7.000" Participant="me011">zero-length</Segment>
  </Transcript>
</Meeting>
"""


class TestParsers:
    def test_meetings_xml_keys_by_agent_not_document_order(self, tmp_path):
        path = tmp_path / "meetings.xml"
        path.write_text(MEETINGS_XML)
        meetings = parse_meetings_xml(path)
        assert meetings["ES2003a"] == {"A": (0, "MEE006"), "B": (1, "FEE005")}

    def test_words_xml_keeps_only_real_timed_words(self, tmp_path):
        path = tmp_path / "words.xml"
        path.write_text(WORDS_XML)
        # punctuation (zero-duration), vocal sounds, and time-less words drop
        assert parse_words_xml(path) == [(77.44, 77.74), (82.0, 82.5)]

    def test_mrt_channel_map_and_segments(self, tmp_path):
        path = tmp_path / "meeting.mrt"
        path.write_text(MRT_XML)
        mrt = parse_mrt(path)
        assert mrt.channel_files == {"chan0": "chan0.sph", "chan3": "chan3.sph"}
        # mn055 has no close-talk channel → excluded from map and segments
        assert mrt.participant_channels == {"me011": "chan0", "fe008": "chan3"}
        assert mrt.segments == {"me011": [(0.53, 1.43)], "fe008": [(2.0, 3.0)]}


class TestIcsiConventions:
    def test_session_letters_follow_meeting_order(self):
        # The letter remap only preserves "enroll on a, trial on the rest" if
        # letter order equals meeting order.
        meetings = list(ICSI_SESSIONS.values())
        assert list(ICSI_SESSIONS) == sorted(ICSI_SESSIONS)
        assert meetings == sorted(meetings)

    def test_icsi_speakers_is_the_single_universe(self):
        # A close-talk participant with no transcribed speech is not a speaker;
        # sorted order fixes the mic wearer ([0]) and the convention stranger
        # ([-1]) for every consumer at once.
        mrt = MrtMeeting(
            channel_files={"chan0": "chan0.sph", "chan1": "chan1.sph", "chan2": "chan2.sph"},
            participant_channels={"me011": "chan0", "fe008": "chan1", "mn017": "chan2"},
            segments={"me011": [(0.0, 1.0)], "mn017": [(2.0, 3.0)]},
        )
        assert icsi_speakers(mrt) == {"me011": "chan0", "mn017": "chan2"}
