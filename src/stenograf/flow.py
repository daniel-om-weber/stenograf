"""The workflows a *UI* runs: meeting, transcribe, notes — without any UI in them.

``steno start``'s command body is the CLI's; this module is the equivalent
for the button-driven entry, the Qt desktop app (:mod:`stenograf.gui`) — and
the reason it must stay out of the screens: the app and the CLI must run the
*same* meeting — same folder allocation, same load order, same notes tail —
or they drift into two products. So the screens keep only what a screen is:
gathering inputs, showing progress, handling the answer. Everything between
those lives here, expressed against :class:`~stenograf.view.LiveView` and
plain callbacks, so it is drivable from a Qt worker thread or a test with no
UI at all.

Differences from the CLI are deliberate scope, not drift: no ``--out``/``--force``
(a fresh date-named folder can't collide), no replay/AEC-dump/full-finalize
(developer flags), and progress reports through the view instead of
``click.echo``. Everything the CLI resolves from flags — formats, vocabulary,
re-ID, AEC, checkpoint cadence — comes from settings.toml through the very same
helpers ``cli/run.py`` uses, so a flagless ``steno start`` and a clicked Start
button can never disagree about defaults.

Ordering matters twice in :meth:`MeetingRun.run`:

- the *slow* assembly (model loading) runs after the capture provider is
  created **and started**, so the Stop control is wired almost immediately
  (:meth:`~stenograf.view.LiveView.set_stop`) and the meeting's first seconds
  buffer in the provider's queue through a slow (cold) model load instead of
  being lost;
- the transcript is persisted at the ``finalized`` event (the ``_PersistOnce``
  contract the CLI TUI path uses), so a force-quit on the "done" screen — or
  even mid-finalize — never loses the meeting.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from stenograf.config import Language, MeetingProfile
from stenograf.transcript import DEFAULT_FORMATS, Transcript

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from stenograf.settings import Settings
    from stenograf.view import LiveView


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
    ``record_audio`` is the CLI's bare ``--record-audio``: keep the raw capture
    as the meeting folder's ``audio.wav``."""

    profile: MeetingProfile
    settings: Settings
    notes: bool
    record_audio: bool


def standing_settings() -> Settings:
    """The settings a setup form's controls start from; a broken file reads as defaults.

    A form must open even when settings.toml is unusable — the error belongs on
    the Start attempt (:func:`resolve_meeting_request` reports it), not on a
    screen the user cannot yet act on."""
    from stenograf.settings import Settings, SettingsError, load_settings

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
    # The CLI's own resolution seams, reused so both entries share one source of
    # defaults (the thin-client rule): load_settings for the tables,
    # _collect_terms for the [vocab] glossary/attendee baseline.
    from click import ClickException

    from stenograf.cli.run import _collect_terms
    from stenograf.settings import SettingsError, apply_meeting_preset, load_settings

    try:
        settings = load_settings()
        preset_obj = None
        if preset is not None:
            settings, preset_obj = apply_meeting_preset(settings, preset)
        glossary_terms, attendee_names = _collect_terms(
            (),
            None,
            (),
            vocab=settings.vocab,
            extra_vocab=preset_obj.vocab if preset_obj is not None else None,
        )
    except (SettingsError, ClickException) as exc:  # e.g. a stale [vocab] glossary_file
        raise MeetingRequestError(getattr(exc, "message", str(exc))) from exc
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

    def __init__(self, request: MeetingRequest, *, abort: threading.Event | None = None) -> None:
        from stenograf.cli.start import _PersistOnce
        from stenograf.output import (
            TRANSCRIPT_STEM,
            allocate_meeting_dir,
            cleanup_checkpoints,
            default_output_home,
            write_transcript,
        )

        self.request = request
        self.created_at = datetime.now()
        self.basename = TRANSCRIPT_STEM
        # A fresh date-named folder under the visible output home — a button has
        # no --out equivalent, so allocation can never collide with an existing
        # meeting. Nothing is created until the first write.
        self.out_dir = allocate_meeting_dir(
            request.settings.output.dir or default_output_home(), self.created_at
        )
        formats = list(request.settings.transcript.formats or DEFAULT_FORMATS)

        def write(transcript: Transcript) -> list[Path]:
            paths = write_transcript(transcript, self.out_dir, self.basename, formats)
            cleanup_checkpoints(self.out_dir, self.basename)
            return paths

        self.persist = _PersistOnce(write)
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
        """Capture, finalize, and (if asked) write notes, reporting through ``view``.

        Blocking, and meant for a worker thread: it returns when the meeting is
        over. Ends when the view's stop callback — installed here, on the first
        line that has something to stop — is invoked."""
        from stenograf import loaders
        from stenograf.cli.start import _LIVE_FLUSH_INTERVAL_S
        from stenograf.output import AUDIO_NAME, checkpoint_writer
        from stenograf.session import CheckpointConfig, MeetingRecorder, plan_channels

        settings, profile = self.request.settings, self.request.profile
        plans = plan_channels(profile)

        if self.abort.is_set():
            return None
        view.status("starting capture…")
        # announce=view.status everywhere below: loader progress must go to the
        # view, never through click — a TUI owns stdio, a GUI has no stdio at
        # all, and on Windows click.echo dies probing its proxy (loaders module
        # docstring). on_log likewise: the capture transports' stderr chatter
        # must not be written over the running app; problems reach the view.
        provider = loaders.make_provider(
            None,
            plans,
            paced=True,
            aec=True,
            announce=view.status,
            on_log=loaders.CaptureLog(view=view),
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
        tee = None
        try:
            if self.request.record_audio:
                from stenograf.recording import WavTee

                # The tee is this run's first write, so it creates the folder.
                self.out_dir.mkdir(parents=True, exist_ok=True)
                tee = WavTee(self.out_dir / AUDIO_NAME, {p.channel for p in plans})
            view.status("recording · loading models…")
            asr, vad, diarizer = loaders.load_backends(
                need_diarizer=any(p.num_speakers != 1 for p in plans),
                asr_backend=settings.asr.backend,
                asr_provider=settings.asr.provider,
                glossary=profile.glossary,
                attendee_names=profile.attendee_names,
                boost=settings.asr.boost,
                announce=view.status,
            )
            reid = None
            if diarizer is not None:
                reid = loaders.load_reid(
                    enabled=True,
                    threshold=settings.speakers.reid_threshold,
                    store_path=settings.speakers.profile_store,
                )
            recorder = MeetingRecorder(
                profile,
                asr=asr,
                vad=vad,
                diarizer=diarizer,
                reid=reid,
                language=profile.language,
                glossary_threshold=settings.vocab.glossary_threshold,
                dedup_echo=True,
            )
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
                live=True,
                view=view,
                on_frame=tee.add if tee else None,
                checkpoint=CheckpointConfig(
                    checkpoint_writer(self.out_dir, self.basename), _LIVE_FLUSH_INTERVAL_S
                ),
                provider_started=True,
            )
        finally:
            if tee is not None:
                tee.close()  # flush + finalize the WAV header even on a dying run
        transcript = result.transcript
        if transcript is not None:
            # Persisted already, at the finalized event — this is display only.
            notes_ok = True
            if self.request.notes and not self.abandon_notes.is_set():
                from stenograf.cli.notes import _generate_notes

                self.notes_running = True
                try:
                    notes_ok = _generate_notes(
                        view,
                        transcript,
                        self.out_dir,
                        self.basename,
                        created_at=self.created_at,
                        notes_settings=settings.notes,
                    )
                finally:
                    self.notes_running = False
            if notes_ok:  # a notes failure keeps its own message visible
                # No "press q" hint: two views render this line and only one of
                # them is a terminal. Each says how to leave in its own footer.
                view.status("saved")
        return transcript


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
    import dataclasses
    import time

    from stenograf import loaders
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.cli.run import _collect_terms
    from stenograf.cli.transcribe import _resolve_split_channels, _transcribe_split_channels
    from stenograf.output import (
        TRANSCRIPT_STEM,
        allocate_meeting_dir,
        default_output_home,
        write_transcript,
    )
    from stenograf.settings import load_settings
    from stenograf.view import LiveView

    settings = load_settings()
    glossary_terms, attendee_names = _collect_terms((), None, (), vocab=settings.vocab)
    out_dir = allocate_meeting_dir(settings.output.dir or default_output_home(), datetime.now())
    write_formats = list(settings.transcript.formats or DEFAULT_FORMATS)

    split_pcms, _correlation = _resolve_split_channels(audio_file, "auto")
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

        class _StatusView(LiveView):
            """Routes the split-channel finalize's status lines to ``on_status``."""

            def status(self, message: str) -> None:
                on_status(message)

            def error(self, message: str) -> None:
                on_status(f"warning: {message}")

        result, elapsed = _transcribe_split_channels(
            *split_pcms,
            profile=profile,
            use_reid=True,
            reid_threshold=settings.speakers.reid_threshold,
            glossary_threshold=settings.vocab.glossary_threshold,
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
            asr_boost=settings.asr.boost,
            profile_store=settings.speakers.profile_store,
            view=_StatusView(),
        )
        transcript = result.transcript
    else:
        from stenograf.pipeline import STAGE_ASR, STAGE_DIARIZATION, finalize_file

        samples = load_audio(audio_file)
        duration = len(samples) / SAMPLE_RATE
        on_status("loading models…")
        asr, vad, diarizer = loaders.load_backends(
            need_diarizer=diarize,
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
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

    paths = write_transcript(transcript, out_dir, TRANSCRIPT_STEM, write_formats)
    return TranscribeResult(paths=paths, out_dir=out_dir, duration=duration, elapsed=elapsed)


def settings_report() -> tuple[list[str], bool]:
    """``steno settings show`` as plain lines, plus whether the file loaded.

    Every key with its value and where it came from (env override,
    settings.toml, built-in default), rendered through the same
    ``_settings_rows`` helper the CLI prints from, so no two entries can
    disagree about the effective configuration. A broken file renders its error
    instead and returns ``False`` — what to *do* about it is the calling UI's
    line to write, since one has a keybinding and the other a button."""
    from stenograf.cli.settings_cmd import _settings_rows
    from stenograf.settings import SettingsError, load_settings, settings_path

    path = settings_path()
    suffix = "" if path.exists() else " (not present — all defaults)"
    lines = [f"settings: {path}{suffix}"]
    try:
        settings = load_settings()
    except SettingsError as exc:
        return [*lines, "", str(exc)], False
    for table, rows in _settings_rows(settings):
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
    import contextlib

    from stenograf.output import default_output_home

    home = None
    with contextlib.suppress(Exception):  # a broken settings.toml
        from stenograf.settings import load_settings

        home = load_settings().output.dir
    return home or default_output_home()


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
    from stenograf.cli.notes import _generate_and_write_notes
    from stenograf.output import TRANSCRIPT_STEM, created_at_from_dir_name

    path = target / f"{TRANSCRIPT_STEM}.json" if target.is_dir() else target
    if not path.is_file():
        raise ValueError(f"{target} holds no {TRANSCRIPT_STEM}.json")
    try:
        transcript = Transcript.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path} is not a readable transcript JSON: {exc}") from exc
    out_dir = path.parent
    created_at = created_at_from_dir_name(out_dir.name) or datetime.fromtimestamp(
        path.stat().st_mtime
    )
    written, notes = _generate_and_write_notes(
        transcript, out_dir, path.stem, created_at=created_at, on_progress=on_progress
    )
    warnings = notes.provenance.warnings if notes.provenance is not None else ()
    return written, warnings


__all__ = [
    "MeetingRequest",
    "MeetingRequestError",
    "MeetingRun",
    "TranscribeResult",
    "generate_notes_for",
    "notes_home",
    "resolve_meeting_request",
    "settings_report",
    "standing_settings",
    "transcribe_recording",
]
