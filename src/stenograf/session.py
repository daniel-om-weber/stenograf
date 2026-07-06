"""Meeting orchestrator: capture frames → in-RAM store → finalized transcript.

This is the spine that turns the accuracy core (``pipeline.finalize_channel``)
into a live meeting. It:

1. consumes ``AudioFrame`` objects from any :class:`CaptureProvider`,
2. accumulates each channel's PCM in a bounded in-RAM store (never disk),
3. on stop, runs the finalize pass **per channel** with that channel's known
   speaker count — the biggest diarization accuracy lever (PLAN.md §2) — then
4. interleaves the two channels' entries into one timeline, labelling the mic
   channel ``Local-N`` and the system channel ``Remote-N``.

The channel prior is free separation: diarization never has to tell local from
remote voices, only voices *within* a channel. The provider is swappable
(Swift helper on macOS, sounddevice on Linux, file replay for dev/test), so the
whole orchestrator is exercisable without native capture.

Hybrid-mode cross-channel dedup (room speaker bleed on the mic) is not done
here yet: its primary mitigation is AEC in the capture helper, which does not
exist yet, so there is nothing to dedup in the synthetic paths. Tracked for
when the macOS helper lands (PLAN.md §2 "Hybrid-mode caveats").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from stenograf.asr.base import ASRBackend
from stenograf.audio import to_float32
from stenograf.capture.base import (
    ORDER_TOLERANCE_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
    CaptureProvider,
    Channel,
)
from stenograf.config import Language, MeetingProfile
from stenograf.diarization.base import Diarizer
from stenograf.lid import detect_language
from stenograf.pipeline import finalize_channel, relabel_speakers
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.vad import SileroVAD

_CHANNEL_LABEL = {Channel.MIC: "Local-{n}", Channel.SYSTEM: "Remote-{n}"}


class SessionStore:
    """Per-channel in-RAM PCM accumulation — int16, ~115 MB/hour/channel.

    Frames are placed by their session timestamp, not merely concatenated, so a
    gap between frames pads silence and both channels share one clock anchored
    at t=0. That shared clock is what lets the two channels' finalized entries
    interleave correctly. No audio is ever written to disk.
    """

    def __init__(self, channels: set[Channel]) -> None:
        self._chunks: dict[Channel, list[np.ndarray]] = {ch: [] for ch in channels}
        self._lengths: dict[Channel, int] = dict.fromkeys(channels, 0)

    def append(self, frame: AudioFrame) -> None:
        """Store a frame at its timestamp; frames must arrive in order per channel."""
        chunks = self._chunks.get(frame.channel)
        if chunks is None:
            return  # a channel we're not recording — ignore
        offset = round(frame.timestamp * SAMPLE_RATE)
        length = self._lengths[frame.channel]
        if offset < length - ORDER_TOLERANCE_SAMPLES:
            # A backward jump past jitter tolerance means the stream desynced;
            # appending here would silently misalign every later frame.
            raise ValueError(
                f"{frame.channel} frame went backwards "
                f"{(length - offset) / SAMPLE_RATE:.3f}s (timestamp {frame.timestamp:.3f}s "
                f"< buffered {length / SAMPLE_RATE:.3f}s); frames must arrive in order"
            )
        if offset > length:  # gap since the last frame → pad silence
            chunks.append(np.zeros(offset - length, dtype=np.int16))
            length = offset
        # A minor overlap (within tolerance) just appends at the tail, keeping
        # the buffer contiguous and the clock monotonic.
        samples = np.asarray(frame.samples, dtype=np.int16)
        chunks.append(samples)
        self._lengths[frame.channel] = length + len(samples)

    def channels(self) -> list[Channel]:
        return list(self._chunks)

    def samples(self, channel: Channel) -> np.ndarray:
        """The channel's full audio as mono 16 kHz float32 (empty if none)."""
        chunks = self._chunks[channel]
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return to_float32(pcm)

    def duration(self, channel: Channel) -> float:
        return self._lengths[channel] / SAMPLE_RATE


@dataclass(frozen=True)
class ChannelPlan:
    """How one channel is transcribed: its speaker count and display label."""

    channel: Channel
    num_speakers: int | None  # None = estimate, 1 = single (no diarization), >1 = known
    label_template: str


def plan_channels(profile: MeetingProfile) -> list[ChannelPlan]:
    """Resolve which channels to record and each channel's speaker count.

    The mic is recorded unless there is explicitly no local speaker
    (``local_speakers == 0``, a listen-only session); an unknown local count
    (``None``) defaults to 1 — a single local user — because far-field
    local-count estimation is Phase 3 (PLAN.md §2). The system tap is recorded
    unless the meeting is in-room (``remote_speakers == 0``); an unknown remote
    count (``None``) means record it and estimate. ``MeetingProfile`` forbids
    both counts being 0, so at least one channel is always planned.
    """
    plans = []
    if profile.local_speakers != 0:
        local = 1 if profile.local_speakers is None else profile.local_speakers
        plans.append(ChannelPlan(Channel.MIC, local, _CHANNEL_LABEL[Channel.MIC]))
    if profile.remote_speakers != 0:
        plans.append(
            ChannelPlan(Channel.SYSTEM, profile.remote_speakers, _CHANNEL_LABEL[Channel.SYSTEM])
        )
    return plans


def interleave(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Merge per-channel entries into one timeline, ordered by start time."""
    return sorted(entries, key=lambda e: (e.start, e.end))


class MeetingRecorder:
    """Drives a capture session and produces the merged, labelled transcript.

    Construct with the finalize backends (shared across channels — the diarizer
    is stateless per call and takes each channel's count as an argument), then
    call :meth:`run` with a provider. Backends are reused, so a wrong meeting
    parameter is corrected by re-running :meth:`finalize` on the same store.
    """

    def __init__(
        self,
        profile: MeetingProfile,
        *,
        asr: ASRBackend,
        vad: SileroVAD | None = None,
        diarizer: Diarizer | None = None,
        language: Language | None = None,
    ) -> None:
        self.profile = profile
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.language = language or profile.language

    def run(
        self,
        provider: CaptureProvider,
        *,
        on_frame: Callable[[AudioFrame], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_checkpoint: Callable[[Transcript], None] | None = None,
        checkpoint_interval: float = 180.0,
        max_seconds: float | None = None,
    ) -> Transcript:
        """Capture until the provider stops (or Ctrl-C), then finalize.

        ``on_frame`` sees every stored frame (used by the audio tee); a
        ``KeyboardInterrupt`` ends capture gracefully rather than aborting, so
        an interrupted meeting still yields a transcript of what was captured.
        ``max_seconds`` stops capture automatically after that much audio.

        If ``on_checkpoint`` is given, the completed audio is re-finalized every
        ``checkpoint_interval`` seconds of capture and the transcript handed to
        the callback — the crash-recovery safety net (PLAN.md §3): a crash then
        loses at most one interval of *text*, and audio is still never persisted.
        The authoritative transcript remains the full finalize returned on stop.
        """
        plans = plan_channels(self.profile)
        store = SessionStore({p.channel for p in plans})
        checkpointing = on_checkpoint is not None and checkpoint_interval > 0
        next_checkpoint = checkpoint_interval
        provider.start({p.channel for p in plans})
        try:
            for frame in provider.frames():
                store.append(frame)
                if on_frame is not None:
                    on_frame(frame)
                captured = max(store.duration(p.channel) for p in plans)
                if checkpointing and captured >= next_checkpoint:
                    # Inline re-finalize of the whole captured-so-far. Fine for
                    # the file/replay providers; when the real-time capture
                    # helper lands this must move off the consume thread (its
                    # backpressure design) and finalize only the new tail.
                    on_checkpoint(self.finalize(store, plans))
                    while captured >= next_checkpoint:
                        next_checkpoint += checkpoint_interval
                if max_seconds is not None and captured >= max_seconds:
                    break
        except KeyboardInterrupt:
            if on_status is not None:
                on_status("interrupted — finalizing captured audio")
        finally:
            provider.stop()
        return self.finalize(store, plans, on_status=on_status)

    def finalize(
        self,
        store: SessionStore,
        plans: list[ChannelPlan] | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> Transcript:
        """Run the finalize pass on every stored channel and interleave them."""
        plans = plans or plan_channels(self.profile)
        entries: list[TranscriptEntry] = []
        for plan in plans:
            if plan.channel not in store.channels():
                continue
            if on_status is not None:
                on_status(f"finalizing {plan.channel} ({_speaker_note(plan.num_speakers)})")
            samples = store.samples(plan.channel)
            diarizer = None if plan.num_speakers == 1 else self.diarizer
            raw = finalize_channel(
                samples,
                asr=self.asr,
                language=self.language,
                vad=self.vad,
                diarizer=diarizer,
                num_speakers=plan.num_speakers,
            )
            entries.extend(relabel_speakers(raw, plan.label_template))
        interleaved = interleave(entries)
        language = self._resolve_language(interleaved, on_status=on_status)
        return Transcript(language=language, profile=self.profile, entries=interleaved)

    def _resolve_language(
        self,
        entries: list[TranscriptEntry],
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> Language | None:
        """Fill the meeting language by LID over the transcript, at most once.

        An explicit user setting always wins. Otherwise detect from the
        finalized text and lock the result on ``self.language`` so later
        checkpoints stay consistent (PLAN.md §2 "auto-detect once … then lock").
        """
        if self.language is not None:
            return self.language
        detected = detect_language(" ".join(e.text for e in entries))
        if detected is not None:
            self.language = detected  # lock for the session
            if on_status is not None:
                on_status(f"detected language: {detected.value}")
        return detected


def _speaker_note(num_speakers: int | None) -> str:
    if num_speakers is None:
        return "estimating speakers"
    if num_speakers == 1:
        return "single speaker"
    return f"{num_speakers} speakers"
