"""The workflows a *UI* runs: meeting, transcribe, notes — without any UI in them.

:class:`MeetingRun` is **the** meeting: ``steno start`` and the Qt app's
Start button (:mod:`stenograf.gui`) are both thin adapters over it — the CLI
translates flags into a :class:`MeetingRequest` plus :class:`RunOptions`, the
app builds the request from its setup form and takes every option's default.
There is exactly one assembly sequence (resolve → plan channels → capture →
load backends → record → persist → notes tail); a front-end that wants to
run it differently has nothing to fork, which is what keeps the two from
drifting into two products. The screens keep only what a screen is:
gathering inputs, showing progress, handling the answer. Everything here is
expressed against :class:`~stenograf.view.LiveView` and plain callbacks, so
it is drivable from a Qt worker thread, a terminal, or a test with no UI at
all — progress goes through the view, never ``click.echo``.

Ordering matters twice in :meth:`MeetingRun.run`:

- the *slow* assembly (model loading) runs after the capture provider is
  created **and started**, so the Stop control is wired almost immediately
  (:meth:`~stenograf.view.LiveView.set_stop`) and the meeting's first seconds
  buffer in the provider's queue through a slow (cold) model load instead of
  being lost;
- the transcript is persisted at the ``finalized`` event (the
  :class:`~stenograf.output.PersistOnce` contract the CLI path shares), so a
  force-quit on the "done" screen — or even mid-finalize — never loses the
  meeting.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from stenograf import loaders, output
from stenograf.audio import SAMPLE_RATE, load_audio
from stenograf.config import Language, MeetingProfile
from stenograf.notes import run as notes_run
from stenograf.pipeline import (
    STAGE_ASR,
    STAGE_DIARIZATION,
    finalize_file,
    resolve_split_channels,
    transcribe_split_channels,
)
from stenograf.recording import WavTee
from stenograf.session import (
    LIVE_FLUSH_INTERVAL_S,
    CheckpointConfig,
    MeetingRecorder,
    plan_channels,
)
from stenograf.settings import (
    Settings,
    SettingsError,
    apply_meeting_preset,
    load_settings,
    settings_path,
    settings_rows,
)
from stenograf.transcript import DEFAULT_FORMATS, Transcript
from stenograf.view import CallbackView, LiveView
from stenograf.vocab import collect_terms

if TYPE_CHECKING:
    from collections.abc import Callable

    from stenograf.session import MeetingResult


class MeetingRequestError(Exception):
    """A setup form's inputs (plus settings.toml) do not describe a runnable meeting.

    One type for every way that can happen — an unreadable settings file, a
    stale ``[vocab] glossary_file`` path, both sources switched off — so a form
    has one thing to catch and one message to show. The message is
    user-facing text, already phrased for a dialog."""


@dataclass(frozen=True)
class MeetingRequest:
    """What a setup form resolved: the profile to record plus the run-level extras.

    ``settings`` rides along so the run uses the exact values in force when the
    user pressed Start — not whatever the file says seconds later.
    ``record_audio`` is the CLI's ``--record-audio``: keep the raw capture as
    the meeting folder's ``audio.wav`` (or :attr:`RunOptions.audio_path`)."""

    profile: MeetingProfile
    settings: Settings
    notes: bool
    record_audio: bool


@dataclass(frozen=True)
class RunOptions:
    """The per-run knobs beyond a setup form's controls — ``steno start``'s
    developer and tuning flags. Every default is the button UI's fixed
    choice, so a default-constructed ``RunOptions`` and a clicked Start
    describe the same run; the CLI adapter fills in whatever its flags say.
    ``None`` throughout means "the settings.toml value" (or the built-in
    default behind it)."""

    replay: str | None = None
    """Dev: replay audio file(s) as the mic (and optional system) channel."""
    out: Path | None = None
    """Use this directory as the meeting's folder (``--out``); ``None``
    allocates a fresh date-named folder that cannot collide."""
    force: bool = False
    """Let ``out`` overwrite a folder already holding a transcript."""
    live: bool = True
    """Stream live captions; ``False`` is the silent batch capture."""
    aec: bool = True
    """Echo-cancel the mic against the system channel (and dedup at merge)."""
    aec_dump: Path | None = None
    """Write the canceller's mic/lpb/enh WAV triple here for offline scoring."""
    flush_interval: float | None = None
    """Seconds between live crash checkpoints (cheap file I/O); ``None`` picks
    the default cadence. Batch runs never checkpoint."""
    max_seconds: float | None = None
    """Stop capture automatically after this much audio."""
    full_finalize: bool = False
    """Re-transcribe from scratch at stop instead of reusing the live pass."""
    use_reid: bool = True
    """Relabel diarized speakers from the saved voiceprint store."""
    reid_threshold: float | None = None
    reid_store: Path | None = None
    glossary_threshold: float | None = None
    formats: tuple[str, ...] | None = None
    """Transcript formats to write; ``None`` = ``[transcript] formats``."""
    audio_path: Path | None = None
    """Where ``record_audio`` writes; ``None`` = ``audio.wav`` in the folder."""
    notes_backend: str | None = None
    notes_model: str | None = None
    notes_instructions: Path | None = None
    """The per-run notes trio — one meeting with a different notes setup."""
    transport_stderr: bool = False
    """Leave the capture transports' diagnostics on inherited stderr (a
    terminal command owns its stderr and a raw write corrupts nothing);
    ``False`` buffers them and routes problem lines to the view — a GUI
    process may own no usable stderr at all."""

    def resolved_flush_interval(self) -> float:
        """The live checkpoint cadence: the explicit value (0 = off), else the
        default beside ``CheckpointConfig``. Batch runs never checkpoint."""
        if self.flush_interval is not None:
            return self.flush_interval
        return LIVE_FLUSH_INTERVAL_S


def standing_settings() -> Settings:
    """The settings a setup form's controls start from; a broken file reads as defaults.

    A form must open even when settings.toml is unusable — the error belongs on
    the Start attempt (:func:`resolve_meeting_request` reports it), not on a
    screen the user cannot yet act on."""
    try:
        return load_settings()
    except SettingsError:
        return Settings()


def resolve_meeting_request(
    *,
    mic: bool,
    system: bool,
    diarize: bool,
    local_speakers: int | None = None,
    remote_speakers: int | None = None,
    language: Language | None = None,
    title: str = "",
    notes: bool = False,
    record_audio: bool = False,
    preset: str | None = None,
) -> MeetingRequest:
    """Turn a setup form's controls into a runnable :class:`MeetingRequest`.

    ``mic``/``system`` are the source switches, ``diarize`` the "tell speakers
    apart" switch, and the two counts the per-channel choices that only exist
    while it is on (``None`` = estimate the count). They collapse into the
    profile's counts the way the CLI's flags do: a source switched off is 0
    speakers, and without diarization a live source is exactly 1 — which is what
    keeps the diarizer model unloaded.

    ``preset`` applies a ``[meetings.<name>]`` section exactly as the CLI's
    ``--preset`` does: its notes setup rides in the returned request's
    ``settings``, its vocabulary merges into the profile, and its
    title/language are defaults a form's typed values (``title``,
    ``language``) still beat. One known precedence gap on this UI path only:
    ``STENOGRAF_NOTES_BACKEND`` still beats a preset's *backend* inside
    ``create_backend`` (the CLI passes the backend explicitly; this path has
    no override channel) — the env var is a developer escape hatch, so
    documented rather than plumbed.

    Raises :class:`MeetingRequestError` with a message meant for the user."""
    try:
        settings = load_settings()
        preset_obj = None
        if preset is not None:
            settings, preset_obj = apply_meeting_preset(settings, preset)
        glossary_terms, attendee_names = collect_terms(
            (),
            None,
            (),
            vocab=settings.vocab,
            extra_vocab=preset_obj.vocab if preset_obj is not None else None,
        )
    except SettingsError as exc:  # e.g. a stale [vocab] glossary_file
        raise MeetingRequestError(str(exc)) from exc
    if preset_obj is not None:
        title = title or (preset_obj.title or "")
        if language is None and preset_obj.language is not None:
            language = Language(preset_obj.language)

    def count(enabled: bool, chosen: int | None) -> int | None:
        """The profile count a source's controls mean: 0 = off, 1 = one speaker,
        None = diarize and estimate."""
        if not enabled:
            return 0
        return chosen if diarize else 1

    try:
        profile = MeetingProfile(
            language=language,
            local_speakers=count(mic, local_speakers),
            remote_speakers=count(system, remote_speakers),
            glossary=glossary_terms,
            attendee_names=attendee_names,
            title=title,
        )
    except ValueError as exc:  # e.g. both sources switched off
        raise MeetingRequestError(str(exc)) from exc
    return MeetingRequest(
        profile=profile, settings=settings, notes=notes, record_audio=record_audio
    )


class MeetingRun:
    """One meeting: its folder, its persistence hook, and the run itself.

    Split in two on purpose. Construction allocates the output folder and builds
    the persist callback, both of which the UI needs *before* the run starts —
    the view is constructed with ``persist`` so the transcript reaches disk at
    the ``finalized`` event, and the folder path is what the "saved" message
    names. :meth:`run` then does the slow work on whatever thread the UI gives
    it."""

    def __init__(
        self,
        request: MeetingRequest,
        *,
        options: RunOptions | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        self.request = request
        self.options = options or RunOptions()
        self.created_at = datetime.now()
        # A fresh date-named folder under the visible output home (which cannot
        # collide with an existing meeting), or options.out as the folder —
        # refused if it already holds a transcript, unless options.force.
        # Nothing is created until the first write.
        self.out_dir, self.basename, audio_default = output.prepare_output(
            self.options.out, self.created_at, request.settings, force=self.options.force
        )
        self.audio_path = self.options.audio_path or audio_default
        """Where ``record_audio`` lands — resolved here so a front-end can
        name (or banner) the file before the run starts."""
        self.result: MeetingResult | None = None
        """The recorder's full result (speaker counts, echo-dedup stats),
        for reporting a view's events don't carry. Set once :meth:`run` has it."""
        self.elapsed: float | None = None
        """Wall-clock seconds from capture start to the persisted transcript."""
        formats = list(
            self.options.formats or request.settings.transcript.formats or DEFAULT_FORMATS
        )

        def write(transcript: Transcript) -> list[Path]:
            paths = output.write_transcript(transcript, self.out_dir, self.basename, formats)
            output.cleanup_checkpoints(self.out_dir, self.basename)
            return paths

        self.persist = output.PersistOnce(write)
        """Write-the-transcript-once callback; hand it to the view so the files
        land at the ``finalized`` event rather than after the user closes the
        screen."""
        self.abandon_notes = threading.Event()
        """Set by a front-end that is going away (window closed, app quitting):
        a notes step that has not started yet is skipped. The transcript is
        already persisted at the ``finalized`` event, so skipping notes loses
        nothing that ``steno notes --last`` cannot regenerate — while a quit
        that waited on an agentic notes backend could block for its full
        ``[notes] timeout_s``."""
        self.notes_running = False
        """Whether the run is currently inside the notes step — the only phase a
        departing front-end may abandon instead of joining (see
        :attr:`abandon_notes`; capture teardown and finalize must be waited on,
        or the meeting itself is lost)."""
        self.abort = abort or threading.Event()
        """Set while capture is still being *built* to cancel the run outright.

        Between the Start gesture and ``view.set_stop`` there is nothing
        installed to stop — and provider construction is exactly where a wedged
        CoreAudio can hang. A front-end passes its own event in (so the flag
        exists before the worker thread does) and sets it from Stop/Escape/quit;
        :meth:`run` checks it around construction and returns ``None`` instead
        of starting the meeting. Nothing is lost: no audio has been captured
        yet. Once capture is up the stop callback is the way to end a run, and
        a stale flag is ignored."""

    def run(self, view: LiveView) -> Transcript | None:
        """Capture, finalize, persist, and (if asked) write notes, reporting
        through ``view``.

        Blocking, and meant for a worker thread: it returns when the meeting is
        over. Ends when the view's stop callback — installed here, on the first
        line that has something to stop — is invoked."""
        settings, profile = self.request.settings, self.request.profile
        options = self.options
        plans = plan_channels(profile)

        if self.abort.is_set():
            return None
        view.status("starting capture…")
        # announce=view.status everywhere below: loader progress must go to the
        # view, never through click — a GUI has no stdio at all, and on Windows
        # click.echo dies probing its proxy (loaders module docstring). on_log
        # likewise, unless the front-end owns a real stderr and says so.
        provider = loaders.make_provider(
            options.replay,
            plans,
            # Pace file replay to wall-clock only when it feeds the live pass,
            # so a replay demonstrates captions at meeting cadence; batch just
            # dumps it.
            paced=options.live,
            aec=options.aec,
            aec_dump=options.aec_dump,
            announce=view.status,
            on_log=None if options.transport_stderr else loaders.CaptureLog(view=view),
        )
        if self.abort.is_set():
            # The cancel arrived while the provider was being built. Release
            # the devices and walk away — no audio exists, so there is nothing
            # to finalize (and a teardown error must not outrank the cancel).
            with contextlib.suppress(Exception):
                provider.stop()
            return None
        view.set_stop(provider.stop)  # Stop/Ctrl-C crosses to capture from here on
        # Capture starts NOW, before the models load: the provider's frame queue
        # is unbounded, so the meeting's first seconds buffer through a slow
        # (cold) model load instead of being lost. The run below consumes the
        # buffer (``provider_started=True``).
        provider.start({p.channel for p in plans})
        started = time.monotonic()
        tee = None
        try:
            if self.request.record_audio:
                # The tee is this run's first write, so it creates the folder.
                self.audio_path.parent.mkdir(parents=True, exist_ok=True)
                tee = WavTee(self.audio_path, {p.channel for p in plans})
            view.status("recording · loading models…")
            asr, vad, diarizer = loaders.load_backends(
                need_diarizer=any(p.num_speakers != 1 for p in plans),
                asr_backend=settings.asr.backend,
                asr_ep=settings.asr.ep,
                glossary=profile.glossary,
                attendee_names=profile.attendee_names,
                boost=settings.asr.boost,
                announce=view.status,
            )
            reid = None
            if diarizer is not None:  # re-ID relabels diarized speakers only
                reid = loaders.load_reid(
                    enabled=options.use_reid,
                    threshold=(
                        settings.speakers.reid_threshold
                        if options.reid_threshold is None
                        else options.reid_threshold
                    ),
                    store_path=options.reid_store or settings.speakers.profile_store,
                )
                if reid is not None:
                    view.status(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
            recorder = MeetingRecorder(
                profile,
                asr=asr,
                vad=vad,
                diarizer=diarizer,
                reid=reid,
                language=profile.language,
                glossary_threshold=(
                    settings.vocab.glossary_threshold
                    if options.glossary_threshold is None
                    else options.glossary_threshold
                ),
                dedup_echo=options.aec,
            )
            recorder.reuse_live_finalize = not options.full_finalize
        except BaseException:
            # Capture is already live but the run will never start (a load
            # failure) — release the devices on the way out; the error itself
            # still reaches the caller.
            provider.stop()
            if tee is not None:
                tee.close()
            raise
        # Loading is done; clear the status or the loading line would sit in the
        # header for the whole meeting (the recorder emits no status event
        # between capture start and finalize). REC/elapsed carry it from here.
        view.status("")
        try:
            result = recorder.run(
                provider,
                live=options.live,
                view=view,
                on_frame=tee.add if tee else None,
                checkpoint=(
                    CheckpointConfig(
                        output.checkpoint_writer(self.out_dir, self.basename),
                        options.resolved_flush_interval(),
                    )
                    if options.live
                    else None
                ),
                max_seconds=options.max_seconds,
                provider_started=True,
            )
        finally:
            if tee is not None:
                tee.close()  # flush + finalize the WAV header even on a dying run
                view.status(f"recorded audio: {tee.path}")
        self.result = result
        _report_lost_reference(result, view)
        transcript = result.transcript
        if transcript is not None:
            # Usually persisted already, at the finalized event (PersistOnce
            # replays); a view that skipped the event writes here.
            paths = self.persist(transcript)
            self.elapsed = time.monotonic() - started
            view.status(
                f"wrote {', '.join(p.name for p in paths)} → {self.out_dir} "
                f"({self.elapsed:.1f}s)"
            )
            notes_ok = True
            if self.request.notes and not self.abandon_notes.is_set():
                self.notes_running = True
                try:
                    notes_ok = notes_run.run_notes(
                        view,
                        transcript,
                        self.out_dir,
                        self.basename,
                        created_at=self.created_at,
                        notes_settings=settings.notes,
                        backend_name=options.notes_backend,
                        model=options.notes_model,
                        instructions_file=options.notes_instructions,
                    )
                finally:
                    self.notes_running = False
            if notes_ok:  # a notes failure keeps its own message visible
                # No "press q" hint: two views render this line and only one of
                # them is a terminal. Each says how to leave in its own footer.
                view.status("saved")
        return transcript


def _report_lost_reference(result: MeetingResult, view: LiveView) -> None:
    """Say how long echo cancellation ran unprotected, if it did.

    The canceller counts every 10 ms mic tick that ran without a usable system
    reference — frames that never arrived, or a dead tap delivering bit-exact
    zeros; the run reports that as ``result.reference_gap_s``. A lost
    reference degrades to "no cancellation" by design — but silently, so say
    how much of the meeting ran unprotected, and whether the armed text
    backstop had to clean up after it."""
    if not result.reference_gap_s:  # None = no canceller observed; 0 = healthy
        return
    if result.dropped_echo_lines:
        backstop = (
            f"; the text backstop removed {result.dropped_echo_lines} mic "
            "line(s) that duplicated remote speech"
        )
    else:
        backstop = "; review Local lines in those spans for leaked remote speech"
    view.error(
        f"echo cancellation ran without its reference for "
        f"{result.reference_gap_s:.1f}s — the system-audio tap "
        f"stalled or went silent{backstop}"
    )


@dataclass(frozen=True)
class TranscribeResult:
    """What a finished :func:`transcribe_recording` produced, and how fast."""

    paths: list[Path]
    out_dir: Path
    duration: float
    """Audio seconds transcribed."""
    elapsed: float
    """Wall-clock seconds the pipeline took."""

    def summary(self) -> str:
        """The one-line result both UIs show (identical wording by construction)."""
        speed = self.duration / self.elapsed if self.elapsed else 0.0
        return (
            f"wrote {', '.join(p.name for p in self.paths)} → {self.out_dir} "
            f"({self.elapsed:.1f}s, {speed:.1f}x realtime)"
        )


def transcribe_recording(
    audio_file: Path,
    *,
    on_status: Callable[[str], None],
    on_windows: Callable[[int, int], None] | None = None,
) -> TranscribeResult:
    """``steno transcribe`` minus the flags: a file in, a written transcript out.

    Settings.toml supplies formats, vocabulary and the ASR backend; channels are
    auto-detected, speaker counts estimated, re-ID on, and the output lands in a
    fresh date-named folder under the output home (rerun with the CLI to
    override any of it). ``on_windows(done, total)`` drives a progress bar on the
    single-channel path only — a split-channel recording finalizes per channel,
    not per window, and reports through ``on_status`` instead.

    Blocking; meant for a worker thread. Every failure raises."""
    settings = load_settings()
    glossary_terms, attendee_names = collect_terms((), None, (), vocab=settings.vocab)
    out_dir = output.allocate_meeting_dir(output.output_home(settings), datetime.now())
    write_formats = list(settings.transcript.formats or DEFAULT_FORMATS)

    split_pcms, _correlation = resolve_split_channels(audio_file, "auto")
    # Diarization is off unless [speakers] diarization = true — the button UIs'
    # only switch for it (or rerun with the CLI's --diarization): counts collapse
    # to one speaker per channel and the diarizer is never loaded.
    diarize = settings.speakers.diarization is True
    profile = MeetingProfile(
        glossary=glossary_terms, attendee_names=attendee_names, title=audio_file.stem
    )
    if split_pcms is not None:
        if not diarize:
            profile = dataclasses.replace(profile, local_speakers=1, remote_speakers=1)
        duration = len(split_pcms[0]) / SAMPLE_RATE
        on_status("2 voice channels — transcribing per channel…")

        status_view = CallbackView(
            on_status=on_status, on_error=lambda m: on_status(f"warning: {m}")
        )
        result, elapsed = transcribe_split_channels(
            *split_pcms,
            profile=profile,
            view=status_view,
            use_reid=True,
            reid_threshold=settings.speakers.reid_threshold,
            glossary_threshold=settings.vocab.glossary_threshold,
            asr_backend=settings.asr.backend,
            asr_ep=settings.asr.ep,
            asr_boost=settings.asr.boost,
            profile_store=settings.speakers.profile_store,
        )
        transcript = result.transcript
    else:
        samples = load_audio(audio_file)
        duration = len(samples) / SAMPLE_RATE
        on_status("loading models…")
        asr, vad, diarizer = loaders.load_backends(
            need_diarizer=diarize,
            asr_backend=settings.asr.backend,
            asr_ep=settings.asr.ep,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            boost=settings.asr.boost,
            announce=on_status,  # not click: the UI owns stdio (loaders docstring)
        )
        started = time.monotonic()
        reid = None
        if diarizer is not None:  # re-ID relabels diarized speakers only
            reid = loaders.load_reid(
                enabled=True,
                threshold=settings.speakers.reid_threshold,
                store_path=settings.speakers.profile_store,
            )

        def progress(stage: str, done: int, total: int) -> None:
            if stage == STAGE_ASR and on_windows is not None:
                on_windows(done, total)
            elif stage == STAGE_DIARIZATION:
                on_status("diarizing…")

        transcript = finalize_file(
            samples,
            profile=profile,
            asr=asr,
            vad=vad,
            diarizer=diarizer,
            num_speakers=None if diarize else 1,
            reid=reid,
            glossary_threshold=settings.vocab.glossary_threshold,
            on_progress=progress,
        )
        elapsed = time.monotonic() - started

    paths = output.write_transcript(transcript, out_dir, output.TRANSCRIPT_STEM, write_formats)
    return TranscribeResult(paths=paths, out_dir=out_dir, duration=duration, elapsed=elapsed)


def settings_report(preset: str | None = None) -> tuple[list[str], bool]:
    """``steno settings show`` as plain lines, plus whether the file loaded.

    Every key with its value and where it came from (env override, a meeting
    preset, settings.toml, built-in default), rendered through the same
    :func:`~stenograf.settings.settings_rows` helper the CLI prints from, so no
    two entries can disagree about the effective configuration. A broken file
    renders its error instead and returns ``False`` — what to *do* about it is
    the calling UI's line to write, since one has a keybinding and the other a
    button.

    ``preset`` answers "what does this meeting type actually change?" the way
    ``steno settings show --preset NAME`` does: the ``[meetings.<name>]``
    overlay applied, its keys attributed to it. An unknown name is a failure
    like an unreadable file — same error rendering, same ``False``, because a
    picker offering a name settings.toml no longer defines is exactly as
    broken."""
    path = settings_path()
    suffix = "" if path.exists() else " (not present — all defaults)"
    lines = [f"settings: {path}{suffix}"]
    try:
        settings = load_settings()
        preset_obj = None
        if preset is not None:
            settings, preset_obj = apply_meeting_preset(settings, preset)
    except SettingsError as exc:
        return [*lines, "", str(exc)], False
    if preset_obj is not None:
        summary = preset_obj.summary()
        lines.append(
            f"preset:   [meetings.{preset_obj.name}]" + (f" — {summary}" if summary else "")
        )
        if preset_obj.vocab.attendees or preset_obj.vocab.glossary_file:
            # Vocabulary merges rather than overlays, so no row can show it.
            lines.append("          its [vocab] merges into the standing vocabulary")
    for table, rows in settings_rows(settings, preset_obj):
        lines.append("")
        lines.append(f"[{table}]")
        width = max(len(key) for key, _, _ in rows)
        for key, value, source in rows:
            lines.append(f"  {key:<{width}} = {value}  ({source})")
    return lines, True


def notes_home() -> Path:
    """Where a notes picker starts browsing: the configured output home.

    Tolerates a broken settings.toml (falls back to the default home) — a picker
    that refuses to open is worse than one pointing at the standard folder."""
    with contextlib.suppress(Exception):  # a broken settings.toml
        return output.output_home(load_settings())
    return output.default_output_home()


def generate_notes_for(
    target: Path, *, on_progress: Callable[[str], None] | None = None
) -> tuple[list[Path], tuple[str, ...]]:
    """Notes for a meeting folder or a ``transcript.json``: ``(written paths,
    validation warnings)``.

    The picker-shaped ``steno notes``: no ``--last`` (a UI resolves that itself,
    via :func:`~stenograf.output.latest_meeting_dir`) and no backend overrides.
    Generation goes through the shared entry point, which owns the MLX
    thread-affinity guard — a worker thread must never reimplement it.
    Warnings are returned, not printed, because both notes screens render a
    bare status line — dropping them there would silence exactly the evidence
    the validation produced (they are also in the note's own footer).

    Blocking; meant for a worker thread. Every failure raises."""
    transcript, path, created_at = output.load_transcript(target)
    written, notes = notes_run.generate_and_write_notes(
        transcript, path.parent, path.stem, created_at=created_at, on_progress=on_progress
    )
    warnings = notes.provenance.warnings if notes.provenance is not None else ()
    return written, warnings


__all__ = [
    "MeetingRequest",
    "MeetingRequestError",
    "MeetingRun",
    "RunOptions",
    "TranscribeResult",
    "generate_notes_for",
    "notes_home",
    "resolve_meeting_request",
    "settings_report",
    "standing_settings",
    "transcribe_recording",
]
