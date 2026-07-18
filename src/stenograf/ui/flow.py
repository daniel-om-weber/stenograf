"""Launcher meeting flow: assemble the pipeline and run it behind a pushed screen.

Phase 7, Task 3 (PLAN.md §5). The launcher-side equivalent of ``steno start``'s
command body, built from the same library seams (``loaders``,
``MeetingRecorder``, the output writers) with settings.toml supplying
everything the setup form doesn't ask about. Differences from the CLI are
deliberate scope, not drift: no ``--out``/``--force`` (a fresh date-named
folder can't collide), no replay/AEC-dump/full-finalize (developer flags),
and progress reports through the meeting screen's header instead of
``click.echo``. The audio tee is the form's "keep the audio" switch — the
CLI's bare ``--record-audio``, always the meeting folder's ``audio.wav``.

Ordering matters twice here:

- the *slow* assembly (model loading) runs on the meeting thread after the
  screen is up — the user watches "loading models…" in the header instead of
  a frozen launcher; the capture provider is created first so the Stop
  binding is wired almost immediately (``view.set_stop``);
- the transcript is persisted at the ``finalized`` event (the same
  ``_PersistOnce`` contract as the CLI TUI path), so a force-quit on the
  "done" screen — or even mid-finalize — never loses the meeting.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from stenograf import loaders
from stenograf.transcript import DEFAULT_FORMATS, Transcript
from stenograf.ui.meeting import TextualLiveView

if TYPE_CHECKING:
    from pathlib import Path

    from stenograf.settings import Settings
    from stenograf.ui.app import StenografApp
    from stenograf.ui.setup import MeetingRequest
    from stenograf.view import LiveView


def start_meeting(app: StenografApp, request: MeetingRequest) -> TextualLiveView:
    """Push a meeting screen onto ``app`` and run the meeting behind it.

    Arms the meeting on a background thread (started once the screen mounts),
    pushes the screen, and installs a dismiss callback that reports the outcome
    on whatever screen the user lands back on. Returns the view (tests drive
    it; interactive callers ignore it).
    """
    from stenograf.cli.start import _LIVE_FLUSH_INTERVAL_S, _PersistOnce
    from stenograf.output import (
        AUDIO_NAME,
        TRANSCRIPT_STEM,
        allocate_meeting_dir,
        checkpoint_writer,
        cleanup_checkpoints,
        default_output_home,
        write_transcript,
    )
    from stenograf.session import CheckpointConfig, MeetingRecorder, plan_channels

    settings, profile = request.settings, request.profile
    plans = plan_channels(profile)
    created_at = datetime.now()
    # A fresh date-named folder under the visible output home — the launcher
    # has no --out equivalent, so allocation can never collide with an
    # existing meeting (PLAN.md §5 Stage C).
    out_dir = allocate_meeting_dir(settings.output.dir or default_output_home(), created_at)
    basename = TRANSCRIPT_STEM
    write_formats = list(settings.transcript.formats or DEFAULT_FORMATS)

    def _persist_files(transcript: Transcript) -> list[Path]:
        paths = write_transcript(transcript, out_dir, basename, write_formats)
        cleanup_checkpoints(out_dir, basename)
        return paths

    persist = _PersistOnce(_persist_files)
    view = TextualLiveView(profile, language=profile.language, persist=persist, app=app)

    def meeting() -> Transcript | None:
        view.status("starting capture…")
        # announce=view.status everywhere below: loader progress must go to
        # the header, never through click — Textual owns stdio here, and on
        # Windows click.echo dies probing its proxy (loaders module docstring).
        # on_log likewise: the capture transports' stderr chatter must not be
        # written over the running app; problems reach the header instead.
        provider = loaders.make_provider(
            None,
            plans,
            paced=True,
            aec=True,
            announce=view.status,
            on_log=loaders.CaptureLog(view=view),
        )
        view.set_stop(provider.stop)  # Stop/Ctrl-C crosses to capture from here on
        tee = None
        if request.record_audio:
            from stenograf.recording import WavTee

            out_dir.mkdir(parents=True, exist_ok=True)  # the tee is this run's first write
            tee = WavTee(out_dir / AUDIO_NAME, {p.channel for p in plans})
        view.status("loading models…")
        asr, vad, diarizer = loaders.load_backends(
            need_diarizer=any(p.num_speakers != 1 for p in plans),
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
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
        # Loading is done; clear the status or "loading models…" would sit in
        # the header for the whole meeting (the recorder emits no status event
        # between capture start and finalize). REC/elapsed carry it from here.
        view.status("")
        try:
            result = recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=tee.add if tee else None,
                checkpoint=CheckpointConfig(
                    checkpoint_writer(out_dir, basename), _LIVE_FLUSH_INTERVAL_S
                ),
            )
        finally:
            if tee is not None:
                tee.close()  # flush + finalize the WAV header even on a dying run
        transcript = result.transcript
        if transcript is not None:
            # Persisted already, at the finalized event — this is display only.
            # No folder name here: the header is one line and even the date-named
            # folder pushes the quit hint off an 80-column screen; the dismiss
            # toast on Home carries the full path.
            if request.notes:
                _generate_notes(view, transcript, out_dir, basename, created_at, settings)
            view.status("saved · q to close")
        return transcript

    result: dict[str, object] = {}
    view.arm_meeting(meeting, result)

    def finished(transcript: Transcript | None) -> None:
        """Back on the previous screen: say how the meeting ended."""
        if isinstance(transcript, Transcript):
            app.notify(f"Files in {out_dir}", title="Meeting saved", timeout=10)
        elif "error" in result:
            app.notify(str(result["error"]), title="Meeting failed", severity="error", timeout=10)
        elif "transcript" in result:  # ended, but produced nothing to write
            app.notify(
                f"The meeting ended before a transcript was produced; any .partial "
                f"checkpoint is kept in {out_dir}",
                severity="warning",
                timeout=10,
            )
        else:  # force-quit while the finalize still runs on the meeting thread
            app.notify(
                f"Still finalizing in the background — files will appear in {out_dir}",
                severity="warning",
                timeout=10,
            )

    app.push_screen(view.screen, finished)
    return view


def _generate_notes(
    view: LiveView,
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    created_at: datetime,
    settings: Settings,
) -> None:
    """The ``--notes`` tail, launcher-shaped: non-fatal, progress via the header.

    Same contract as the CLI's ``_notes_after_run`` (PLAN.md §5 D6): the
    transcript is already on disk, so a notes failure warns and returns —
    rerun later with ``steno notes``. Generation goes through the shared notes
    entry point, which owns the MLX thread-affinity guard.
    """
    from stenograf.cli.notes import _generate_and_write_notes

    view.status("generating notes…")
    try:
        _generate_and_write_notes(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            notes_settings=settings.notes,
            on_progress=lambda message: view.status(f"notes: {message}"),
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal by contract
        view.error(f"notes failed: {exc} — the transcript is safe; retry with `steno notes`")
