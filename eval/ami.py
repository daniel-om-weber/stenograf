"""AMI + ICSI → labelled diarization/re-ID references in stenograf's topology.

Public meeting corpora (CC-BY-4.0, per-speaker headset channels, human
annotations) stand in for the hand-labelling of private audio that stays
declined: for each meeting, one participant's headset is the **mic** channel
and the other N−1 headsets mixed are the **loopback** channel — stenograf's
exact two-channel shape, with real human labels. References carry the corpus'
*global* person ids so the same voice keeps one name across sessions, which is
what makes enroll-on-session-a / trial-on-b–d re-ID scoring possible.

Corpus facts this module is built on (verified against the mirror 2026-08-06):

- Identity resolves ONLY through ``meetings.xml``'s ``global_name``; agent
  letters permute between sessions in some series and ``segments/@channel``
  reproduces transcriber errors the audio doesn't have.
- ES2003 + IS1009: same 4 participants across all four sessions and zero
  entries in the corpus data-problems table (ES2002/IS1000 have documented
  headset faults and dropouts; ES2004d has dropouts).
- ICSI close-talk channels are NIST SPHERE with embedded shorten — ffmpeg
  decodes them natively, no sph2pipe. The channel↔speaker map and the only
  reliable ICSI timing layer (utterance segments) both live in the ``.mrt``
  transcript, so ICSI references are utterance-granular where AMI's are built
  from word timings; compare ICSI numbers only to ICSI numbers.
- ICSI participant channels are non-contiguous per meeting: the download list
  comes from the ``.mrt`` participant map, never a fixed ``chan0..N`` range.

AMI reference turns are the words layer (``punc``/zero-duration dropped,
vocal sounds excluded) merged per speaker across gaps ≤ ``MERGE_GAP_S`` — the
RT smoothing convention; our DER numbers are comparable only to our own
(``PLAN-DIARIZATION.md``'s calibration note).

Commands (downloads ~2.5 GB into ``eval/audio/ami/raw/`` on first use)::

    uv run --group eval eval/ami.py fetch   # download + build channels & refs
    uv run --group eval eval/ami.py run     # fetch, diarize, score DER + naming
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from common import AUDIO_DIR, OUT_DIR, REFS_DIR, read_pcm16, to_wav16k
from rttm import Turn, parse_rttm, write_rttm

MIRROR = "https://groups.inf.ed.ac.uk/ami"
AMI_ANNOTATIONS_ZIP = "ami_public_manual_1.6.2.zip"
ICSI_TRANSCRIPTS_ZIP = "ICSI_original_transcripts.zip"

AMI_GROUPS = {"ES2003": "abcd", "IS1009": "abcd"}
ICSI_MEETINGS = ("Bmr021", "Bmr025")
"""Bmr025 (33 min, 8 close-talk participants) over Bed009 (54 min, 7): a
larger group in fewer minutes is the better large-group condition."""

MERGE_GAP_S = 0.3
"""Same-speaker word/utterance spans closer than this merge into one turn (the
RT eval smoothing convention for pauses)."""

ENROLL_SESSION = "a"
"""Re-ID galleries enroll from each group's first session; trials come from the
rest. Per group the alphabetically-last participant stays unenrolled so their
clusters are stranger (false-accept) trials."""

RAW_DIR = AUDIO_DIR / "ami" / "raw"
CHANNELS_DIR = AUDIO_DIR / "ami"
AMI_REFS_DIR = REFS_DIR / "ami"
CHANNELS_MANIFEST = CHANNELS_DIR / "channels.json"


# -- pure: interval + mixing math ------------------------------------------

CROSSTALK_ALPHA = 0.35
"""A window survives on a channel when its level-normalized RMS is within this
factor (≈ −9 dB) of the loudest channel's. Headset mics are open mics in one
room: measured on ES2003a before gating, neighbor bleed put 68.9 % false alarm
on the mic channel (DER 75.7 %) — the reference only carries the wearer.
Real stenograf channels are bleed-free (loopback is digital; AEC removes the
far end from the mic), so gating is what makes the synthesis honest.

Swept 0.10–0.35 on ES2003a.mic (2026-08-06): lower values trade missed speech
for bleed false alarm almost 1:1 (VAD-span DER flat at 21–23 %), so the choice
is made on false alarm alone — 3.0 % here vs 10.4 % at 0.10. The channel's
missed-speech floor (12.8 % on the *ungated* headset) is the VAD against
verbatim references, not the gate."""

CROSSTALK_STRONG = 0.3
CROSSTALK_WEAK = 0.03
CROSSTALK_EXPAND = 40
"""Hysteresis on the channel's own speech level (90th percentile of its RMS
over windows where it is the loudest mic): windows above STRONG seed, windows
above WEAK survive only within EXPAND windows (2 s) of a seed. Swept jointly
on IS1009c.mic + ES2003a.mic (2026-08-06, VAD-span DER): this point gives
11.6 % / 25.6 %, false alarm ≤ 2.5 % on both; every stricter or looser
variant trades one room against the other. The ES2003 number is miss-heavy by
construction — the gate uniformly costs ~15 % of soft word-tails there on top
of the VAD-vs-verbatim floor (12.8 % on the ungated channel) — and the
program only reads deltas, in which a uniform gate cost cancels; false alarm
is the axis that would actually corrupt references, and it is ~0."""

CROSSTALK_WIN_S = 0.05
CROSSTALK_HANGOVER = 2
"""Windows each side of an active window that stay open, so soft onsets and
tails survive the gate."""


def _window_rms(pcm: np.ndarray, win: int) -> np.ndarray:
    """Windowed RMS in bounded memory (an hour-long channel would otherwise
    take a full float copy; seven of those at once killed the build)."""
    n = (len(pcm) + win - 1) // win
    out = np.empty(n, dtype=np.float32)
    chunk = 8192  # windows per slice
    for i in range(0, n, chunk):
        seg = pcm[i * win : (i + chunk) * win].astype(np.float32)
        if len(seg) % win:
            seg = np.concatenate([seg, np.zeros(win - len(seg) % win, dtype=np.float32)])
        out[i : i + chunk] = np.sqrt((seg.reshape(-1, win) ** 2).mean(axis=1))
    return out


def crosstalk_masks(channels: list[np.ndarray], win: int = 800) -> list[np.ndarray]:
    """Per-channel boolean window masks: True where the channel is dominant.

    Each channel's windowed RMS is normalized by its own 95th percentile (its
    typical speech level, so quiet talkers are not gated by loud neighbors),
    then a window is active where the channel is within ``CROSSTALK_ALPHA`` of
    the loudest normalized channel, dilated by ``CROSSTALK_HANGOVER`` windows.
    Overlapping speakers are near their own speech level simultaneously, so
    overlap survives on every involved channel."""
    rms = [_window_rms(c, win) for c in channels]
    n = max(len(r) for r in rms)
    mat = np.zeros((len(channels), n), dtype=np.float32)
    for i, r in enumerate(rms):
        mat[i, : len(r)] = r
    # Raw cross-channel comparison, deliberately unnormalized. Close-talk wins
    # its own windows whatever the wearer's absolute level, so a quiet talker
    # is safe; per-channel level normalization was tried and REVERTED — it
    # scales a loud neighbor's bleed up by exactly the wearer's quietness
    # (IS1009, talk-time-asymmetric group: 161–198 % false alarm, unchanged by
    # normalizing against dominant-window levels only). Cost of raw: a quiet
    # speaker's overlap with a much louder one is gated on the quiet side.
    loudest = mat.max(axis=0)
    active = mat >= CROSSTALK_ALPHA * np.maximum(loudest, 1e-12)
    active &= mat > 0
    # And an own-speech level test with hysteresis: dominance alone lets a
    # close mic's breath win every window in which the actual speaker pauses —
    # scattered near-parity seeds the morphology then chains into minutes of
    # phantom speech (IS1009c.mic: 1105 s false, median seed RMS 189 vs 1541
    # during real speech). A flat floor cannot fix it: her breath sits at 0.12
    # of her speech level while ES2003a's wearer has real soft speech below
    # 0.06 of his (a 0.12 floor cost him 13 pts missed). So, VAD-style: only
    # unmistakably-strong windows seed, and weaker windows survive only within
    # a bounded expansion of a seed — soft syllables neighbor strong ones,
    # sustained breath does not.
    argmax = mat.argmax(axis=0)
    for i, r in enumerate(rms):
        own = r[(argmax[: len(r)] == i) & (r > 0)]
        level = (
            float(np.percentile(own, 90))
            if len(own)
            else float(np.percentile(r[r > 0], 95)) if (r > 0).any() else 0.0
        )
        dominant = active[i, : len(r)]
        strong = dominant & (r >= CROSSTALK_STRONG * level)
        weak = dominant & (r >= CROSSTALK_WEAK * level)
        kept = strong.copy()
        for _ in range(CROSSTALK_EXPAND):
            grown = kept.copy()
            grown[1:] |= kept[:-1] & weak[1:]
            grown[:-1] |= kept[1:] & weak[:-1]
            if (grown == kept).all():
                break
            kept = grown
        active[i, : len(r)] = kept
    masks = []
    for i in range(len(channels)):
        mask = active[i].copy()
        for shift in range(1, CROSSTALK_HANGOVER + 1):
            mask[:-shift] |= active[i][shift:]
            mask[shift:] |= active[i][:-shift]
        # Close gaps ≤ MERGE_GAP_S: a brief louder interjection elsewhere must
        # not chop the wearer's stream mid-utterance — fragmentation makes the
        # VAD drop the pieces (measured +7 pts missed on ES2003a.mic).
        open_spans = [
            (float(s * CROSSTALK_WIN_S), float(e * CROSSTALK_WIN_S))
            for s, e in _mask_spans(mask)
        ]
        closed = mask.copy()
        for s, e in merge_spans(open_spans, MERGE_GAP_S):
            closed[round(s / CROSSTALK_WIN_S) : round(e / CROSSTALK_WIN_S)] = True
        masks.append(closed)
    return masks


def _mask_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of ``mask`` as half-open [start, end) window spans."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def apply_mask(pcm: np.ndarray, mask: np.ndarray, win: int = 800) -> np.ndarray:
    """Zero every window whose mask is False (bleed becomes digital silence —
    matching the loopback channel, which is digitally silent between turns)."""
    gated = pcm.copy()
    for w in np.flatnonzero(~mask):
        gated[w * win : (w + 1) * win] = 0
    return gated


def merge_spans(spans: list[tuple[float, float]], gap: float = MERGE_GAP_S) -> list[
    tuple[float, float]
]:
    """Sort spans and merge any pair separated by ``gap`` seconds or less."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def mix_pcm(channels: Iterable[np.ndarray]) -> np.ndarray:
    """Sum int16 channels (zero-padded to the longest); rescale only if the sum
    clips, so the common no-clip case stays sample-exact. Accepts a lazy
    iterable so callers can gate hour-long channels one at a time."""
    total = np.zeros(0, dtype=np.int32)
    for c in channels:
        if len(c) > len(total):
            grown = np.zeros(len(c), dtype=np.int32)
            grown[: len(total)] = total
            total = grown
        total[: len(c)] += c.astype(np.int32)
    peak = int(np.abs(total).max()) if len(total) else 0
    if peak > 32767:
        total = (total.astype(np.float64) * (32767.0 / peak)).round().astype(np.int32)
    return total.astype(np.int16)


def dominant_speaker(cluster: list[Turn], ref: list[Turn]) -> str | None:
    """The reference speaker overlapping ``cluster``'s turns the most, or None
    when nothing in the reference overlaps them (a noise cluster)."""
    overlap: dict[str, float] = {}
    for turn in cluster:
        for r in ref:
            shared = min(turn.end, r.end) - max(turn.start, r.start)
            if shared > 0:
                overlap[r.speaker] = overlap.get(r.speaker, 0.0) + shared
    if not overlap:
        return None
    return max(overlap.items(), key=lambda kv: kv[1])[0]


# -- pure: annotation parsing ----------------------------------------------


def parse_meetings_xml(path: Path) -> dict[str, dict[str, tuple[int, str]]]:
    """``corpusResources/meetings.xml`` → {observation: {agent: (headset, global_name)}}.

    Keyed by ``nxt_agent`` because speaker elements are not in agent order."""
    meetings: dict[str, dict[str, tuple[int, str]]] = {}
    for meeting in ElementTree.parse(path).getroot().iter("meeting"):
        obs = meeting.get("observation")
        if obs is None:
            continue
        agents: dict[str, tuple[int, str]] = {}
        for spk in meeting.iter("speaker"):
            agent, chan, name = spk.get("nxt_agent"), spk.get("channel"), spk.get("global_name")
            if agent and chan is not None and name:
                agents[agent] = (int(chan), name)
        meetings[obs] = agents
    return meetings


def parse_words_xml(path: Path) -> list[tuple[float, float]]:
    """``words/<MEETING>.<AGENT>.words.xml`` → real spoken-word spans.

    Only ``<w>`` elements count (vocal sounds excluded); punctuation tokens
    (``punc="true"``, zero duration) and time-less words are dropped."""
    spans: list[tuple[float, float]] = []
    for w in ElementTree.parse(path).getroot().iter("w"):
        if w.get("punc"):
            continue
        start, end = w.get("starttime"), w.get("endtime")
        if not start or not end:
            continue
        s, e = float(start), float(end)
        if e > s:
            spans.append((s, e))
    return spans


@dataclass(frozen=True)
class MrtMeeting:
    """The channel map and utterance segments of one ICSI ``.mrt`` transcript."""

    channel_files: dict[str, str]
    """channel name (``chan0``) → audio file (``chan0.sph``)."""
    participant_channels: dict[str, str]
    """participant (``me011``) → channel name; close-talk participants only."""
    segments: dict[str, list[tuple[float, float]]]
    """participant → utterance spans (only participants with a channel)."""


def parse_mrt(path: Path) -> MrtMeeting:
    root = ElementTree.parse(path).getroot()
    channel_files = {
        c.get("Name"): c.get("AudioFile")
        for c in root.iter("Channel")
        if c.get("Name") and c.get("AudioFile")
    }
    participant_channels = {
        p.get("Name"): chan
        for p in root.iter("Participant")
        if p.get("Name") and (chan := p.get("Channel")) in channel_files
    }
    segments: dict[str, list[tuple[float, float]]] = {}
    for seg in root.iter("Segment"):
        who = seg.get("Participant")
        start, end = seg.get("StartTime"), seg.get("EndTime")
        if who in participant_channels and start and end and float(end) > float(start):
            segments.setdefault(who, []).append((float(start), float(end)))
    return MrtMeeting(channel_files, participant_channels, segments)


# -- fetching --------------------------------------------------------------


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.rename(dst)


def _ami_headset(meeting: str, headset: int) -> Path:
    dst = RAW_DIR / meeting / f"{meeting}.Headset-{headset}.wav"
    _download(f"{MIRROR}/AMICorpusMirror/amicorpus/{meeting}/audio/{dst.name}", dst)
    return dst


def _ami_annotations() -> Path:
    """Download the manual-annotations release and extract the members we read."""
    zip_path = RAW_DIR / AMI_ANNOTATIONS_ZIP
    _download(f"{MIRROR}/AMICorpusAnnotations/{AMI_ANNOTATIONS_ZIP}", zip_path)
    out = RAW_DIR / "annotations"
    meetings = {group + s for group, sessions in AMI_GROUPS.items() for s in sessions}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            wanted = name == "corpusResources/meetings.xml" or (
                name.startswith("words/") and name.split("/")[-1].split(".")[0] in meetings
            )
            if wanted and not (out / name).exists():
                zf.extract(name, out)
    return out


def _icsi_mrt(meeting: str) -> Path:
    zip_path = RAW_DIR / ICSI_TRANSCRIPTS_ZIP
    _download(f"{MIRROR}/ICSICorpusAnnotations/{ICSI_TRANSCRIPTS_ZIP}", zip_path)
    out = RAW_DIR / "icsi-transcripts"
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.endswith(f"{meeting}.mrt")]
        if not matches:
            raise FileNotFoundError(f"{meeting}.mrt not in {ICSI_TRANSCRIPTS_ZIP}")
        if not (out / matches[0]).exists():
            zf.extract(matches[0], out)
    return out / matches[0]


def _icsi_channel_wav(meeting: str, sph_name: str) -> Path:
    """One ICSI close-talk channel as 16 kHz mono s16 WAV (ffmpeg decodes the
    shorten-compressed SPHERE natively)."""
    sph = RAW_DIR / meeting / sph_name
    _download(f"{MIRROR}/ICSIsignals/SPH/{meeting}/{sph_name}", sph)
    dst = sph.with_suffix(".wav")
    if not dst.exists():
        to_wav16k(sph, dst)
    return dst


# -- building channels + references ----------------------------------------


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setframerate(16_000)
        w.setsampwidth(2)
        w.writeframes(pcm.astype(np.int16).tobytes())


@dataclass(frozen=True)
class Channel:
    """One synthesized channel: its WAV, reference, and diarization inputs."""

    id: str  # e.g. "ES2003a.mic"
    group: str | None  # re-ID group (AMI series), None for ICSI
    session: str | None
    num_speakers: int

    @property
    def wav_path(self) -> Path:
        return CHANNELS_DIR / f"{self.id}.wav"

    @property
    def ref_path(self) -> Path:
        return AMI_REFS_DIR / f"{self.id}.rttm"


def _build_meeting(
    meeting: str,
    group: str | None,
    session: str | None,
    speakers: dict[str, tuple[Path, list[tuple[float, float]]]],
) -> list[Channel]:
    """Write mic/loop WAVs + refs for one meeting.

    ``speakers`` maps global name → (headset wav, merged speech spans). The mic
    participant is the alphabetically-first name — deterministic, and the same
    person across a group's sessions since the participant set is identical."""
    mic_name = sorted(speakers)[0]
    others = {n: v for n, v in speakers.items() if n != mic_name}

    channels = [
        Channel(f"{meeting}.mic", group, session, 1),
        Channel(f"{meeting}.loop", group, session, len(others)),
    ]
    mic, loop = channels
    AMI_REFS_DIR.mkdir(parents=True, exist_ok=True)

    if not (mic.wav_path.exists() and loop.wav_path.exists()):
        names = sorted(speakers)
        pcms = [read_pcm16(speakers[n][0]) for n in names]
        masks = dict(zip(names, crosstalk_masks(pcms), strict=True))
        by_name = dict(zip(names, pcms, strict=True))
        _write_wav(mic.wav_path, apply_mask(by_name[mic_name], masks[mic_name]))
        _write_wav(loop.wav_path, mix_pcm(apply_mask(by_name[n], masks[n]) for n in others))

    write_rttm(mic.ref_path, [Turn(mic_name, s, e) for s, e in speakers[mic_name][1]], mic.id)
    write_rttm(
        loop.ref_path,
        [Turn(name, s, e) for name, (_, spans) in others.items() for s, e in spans],
        loop.id,
    )
    return channels


def fetch() -> list[Channel]:
    """Download everything, synthesize mic/loop channels, write references and
    the channel manifest. Idempotent; already-present files are kept."""
    annotations = _ami_annotations()
    meetings_map = parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    channels: list[Channel] = []

    for group, sessions in AMI_GROUPS.items():
        for session in sessions:
            meeting = group + session
            speakers: dict[str, tuple[Path, list[tuple[float, float]]]] = {}
            for agent, (headset, name) in meetings_map[meeting].items():
                wav = _ami_headset(meeting, headset)
                spans = parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
                speakers[name] = (wav, merge_spans(spans))
            channels += _build_meeting(meeting, group, session, speakers)
            print(f"[{meeting}] {len(speakers)} speakers → mic + loop")

    for meeting in ICSI_MEETINGS:
        mrt = parse_mrt(_icsi_mrt(meeting))
        speakers = {}
        for who, chan in mrt.participant_channels.items():
            spans = mrt.segments.get(who)
            if not spans:
                continue  # a close-talk participant with no transcribed speech
            wav = _icsi_channel_wav(meeting, mrt.channel_files[chan])
            speakers[who] = (wav, merge_spans(spans))
        shared = len(mrt.participant_channels) - len(set(mrt.participant_channels.values()))
        if shared:
            raise RuntimeError(
                f"{meeting}: {shared} participants share a close-talk channel; "
                "pick a different ICSI meeting (channel audio is not per-speaker there)"
            )
        channels += _build_meeting(meeting, None, None, speakers)
        print(f"[{meeting}] {len(speakers)} speakers → mic + loop")

    CHANNELS_MANIFEST.write_text(
        json.dumps(
            [
                {
                    "id": c.id,
                    "group": c.group,
                    "session": c.session,
                    "num_speakers": c.num_speakers,
                }
                for c in channels
            ],
            indent=2,
        )
    )
    print(f"{len(channels)} channels → {CHANNELS_MANIFEST}")
    return channels


def load_channels() -> list[Channel]:
    entries = json.loads(CHANNELS_MANIFEST.read_text())
    return [Channel(**e) for e in entries]


# -- re-ID trials ----------------------------------------------------------


def build_galleries(embed) -> dict[str, dict[str, np.ndarray]]:
    """Per-group enrollment galleries from session ``a`` raw headsets.

    Enrollment slices the participant's *reference* turns from their own
    headset (the clean solo signal our rename-once flow would capture); the
    alphabetically-last participant stays unenrolled, so every later cluster
    of theirs is a stranger trial by construction. ``embed`` is the production
    embedding callable (``SherpaOnnxDiarizer.embed``)."""
    from stenograf.diarization.base import SpeakerTurn
    from stenograf.diarization.sherpa import cluster_embeddings

    annotations = RAW_DIR / "annotations"
    meetings_map = parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    galleries: dict[str, dict[str, np.ndarray]] = {}
    for group in AMI_GROUPS:
        meeting = group + ENROLL_SESSION
        by_name = {
            name: (agent, headset) for agent, (headset, name) in meetings_map[meeting].items()
        }
        gallery: dict[str, np.ndarray] = {}
        for name in sorted(by_name)[:-1]:
            agent, headset = by_name[name]
            spans = merge_spans(
                parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
            )
            pcm = read_pcm16(RAW_DIR / meeting / f"{meeting}.Headset-{headset}.wav")
            turns = [SpeakerTurn(name, s, e) for s, e in spans]
            embedded = cluster_embeddings(turns, pcm, embed)
            gallery[name] = embedded[name]
        galleries[group] = gallery
        print(f"[{group}] enrolled {sorted(gallery)} from session {ENROLL_SESSION}")
    return galleries


def build_trials() -> None:
    """Enroll galleries (:func:`build_galleries`), score every later cluster
    against them, write ``out/reid/trials.json``."""
    from reid_score import Trial, save_trials

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    galleries = build_galleries(SherpaOnnxDiarizer().embed)

    hyp_dir = OUT_DIR / "diar" / "ami"
    trials: list[Trial] = []
    for channel in load_channels():
        if channel.session == ENROLL_SESSION:
            continue  # enrollment source is never a trial
        emb_path = hyp_dir / f"{channel.id}.emb.json"
        if not emb_path.exists():
            print(f"  no embeddings for {channel.id} — run diarize.py --ami", file=sys.stderr)
            continue
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads(emb_path.read_text()).items()
        }
        hyp = parse_rttm(hyp_dir / f"{channel.id}.rttm")
        ref = parse_rttm(channel.ref_path)
        for cluster, vector in embeddings.items():
            cluster_turns = [t for t in hyp if t.speaker == cluster]
            true = dominant_speaker(cluster_turns, ref)
            for group, gallery in galleries.items():
                scores = {name: float(vector @ emb) for name, emb in gallery.items()}
                trials.append(Trial(f"{group}:{channel.id}/{cluster}", true, scores))

    save_trials(OUT_DIR / "reid" / "trials.json", trials)
    known = sum(1 for t in trials if t.known)
    print(
        f"{len(trials)} trials ({known} known, {len(trials) - known} unknown) "
        "→ out/reid/trials.json"
    )


# -- the one-command matrix ------------------------------------------------


def run(channel_ids: set[str] | None = None) -> int:
    """fetch → diarize+finalize every channel → DER + word attribution → re-ID.

    Every heavy stage runs as a child process, and each loop channel gets its
    own: a dense ~40-min channel runs near 10 GB RSS with its ASR pass in the
    same process (observed 2026-08-06), and stacked peaks got the run SIGKILLed
    three times back when sherpa's embedding leak made that ~31 GB. Memory dies
    with each channel's process. Mic channels never diarize, so they share one
    child."""
    started = time.monotonic()
    eval_dir = Path(__file__).parent

    def phase(*argv: str) -> None:
        subprocess.run([sys.executable, *argv], cwd=eval_dir, check=True)

    phase("ami.py", "fetch")
    wanted = [c for c in load_channels() if channel_ids is None or c.id in channel_ids]
    mics = [c.id for c in wanted if c.num_speakers == 1]
    if mics:
        phase("diarize.py", "--ami", "--segments", ",".join(mics))
    for channel in (c for c in wanted if c.num_speakers > 1):
        phase("diarize.py", "--ami", "--segments", channel.id)
    phase("ami.py", "trials")

    import der
    import reid_score

    der.main()
    code = reid_score.main()
    print(f"matrix total: {(time.monotonic() - started) / 60:.1f} min")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "trials", "run"])
    parser.add_argument("--channels", help="comma-separated channel ids (run: a subset)")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
        return 0
    if args.command == "trials":
        build_trials()
        return 0
    return run(set(args.channels.split(",")) if args.channels else None)


if __name__ == "__main__":
    raise SystemExit(main())
