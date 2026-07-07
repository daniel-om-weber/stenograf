"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import click

from stenograf import __version__
from stenograf.config import Language, MeetingProfile
from stenograf.doctor import run_checks
from stenograf.transcript import Transcript

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"

# The transcript formats stenograf can emit, and how each Transcript renders it.
# SRT/VTT re-flow the retained word timestamps into short subtitle cues.
_FORMATS: dict[str, str] = {
    "md": "to_markdown",
    "json": "to_json",
    "srt": "to_srt",
    "vtt": "to_vtt",
}
_DEFAULT_FORMATS = ("md", "json")


def _parse_formats(spec: str) -> list[str]:
    """Parse a ``--format`` value (comma-separated) into an ordered, de-duped list."""
    formats: list[str] = []
    for name in spec.split(","):
        name = name.strip().lower()
        if not name or name in formats:
            continue
        if name not in _FORMATS:
            raise click.BadParameter(
                f"unknown format {name!r}; choose from {', '.join(_FORMATS)}",
                param_hint="--format",
            )
        formats.append(name)
    if not formats:
        raise click.BadParameter("no formats given", param_hint="--format")
    return formats


@click.group()
@click.version_option(__version__, prog_name="stenograf")
def main() -> None:
    """Accuracy-first local meeting transcription. Audio never touches disk."""


@main.command()
@click.option(
    "--lang",
    type=click.Choice([lang.value for lang in Language]),
    default=None,
    help="Meeting language; omit to auto-detect from the first speech.",
)
@click.option(
    "--local",
    "local_speakers",
    type=click.IntRange(0, 8),
    default=None,
    help="Number of speakers in the room; omit to auto-detect.",
)
@click.option(
    "--remote",
    "remote_speakers",
    type=click.IntRange(0, 8),
    default=None,
    help="Number of remote speakers; 0 = in-room meeting without system audio.",
)
@click.option(
    "--replay",
    "replay",
    default=None,
    metavar="MIC[,SYSTEM]",
    help="Dev: replay audio file(s) as the mic (and optional system) channel "
    "instead of live capture. Exercises the full finalize pipeline without the "
    "native capture helper.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for the transcript [default: current directory].",
)
@click.option(
    "--record-audio",
    "record_audio",
    is_flag=False,
    flag_value=_RECORD_DEFAULT,
    default=None,
    metavar="[PATH]",
    help="Also save the raw captured audio to a WAV (mic left, system right). "
    "Off by default — audio otherwise never touches disk. Give a PATH or omit it "
    "to write <transcript>.wav.",
)
@click.option(
    "--flush-interval",
    "--checkpoint-interval",
    "flush_interval",
    type=click.FloatRange(0),
    default=180.0,
    metavar="SECONDS",
    help="Flush a <transcript>.partial crash checkpoint every N seconds of capture "
    "(live: the already-committed captions, zero extra inference; batch: only the "
    "newest tail is finalized, off the capture thread); 0 disables it.",
)
@click.option(
    "--max-seconds",
    type=click.FloatRange(0, min_open=True),
    default=None,
    metavar="SECONDS",
    help="Stop capture automatically after this many seconds [default: until Ctrl-C].",
)
@click.option(
    "--live/--no-live",
    default=True,
    help="Stream live captions while the meeting runs (the on-stop finalize still "
    "replaces them). --no-live captures silently and only finalizes on stop.",
)
@click.option(
    "--plain",
    is_flag=True,
    help="Force the plain line-by-line caption stream instead of the full-screen "
    "TUI (also the automatic choice when stdout is not a terminal).",
)
@click.option(
    "--reid/--no-reid",
    "use_reid",
    default=True,
    help="Relabel diarized speakers to saved profile names when their voice matches "
    "(cross-meeting re-identification). No effect without enrolled profiles.",
)
@click.option(
    "--reid-threshold",
    type=click.FloatRange(0, 1),
    default=None,
    help="Cosine similarity required to match a saved profile [default: 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=",".join(_DEFAULT_FORMATS),
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, srt, vtt "
    "[default: md,json]. srt/vtt re-flow speaker turns into subtitle cues.",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def start(
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    replay: str | None,
    out: Path | None,
    record_audio: str | None,
    flush_interval: float,
    max_seconds: float | None,
    live: bool,
    plain: bool,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str,
    print_markdown: bool,
) -> None:
    """Start transcribing a meeting (capture → finalize on stop)."""
    from stenograf.session import MeetingRecorder, plan_channels

    write_formats = _parse_formats(formats)

    profile = MeetingProfile(
        language=Language(lang) if lang else None,
        local_speakers=local_speakers,
        remote_speakers=remote_speakers,
    )
    mode = profile.mode.value if profile.mode else "auto"
    click.echo(f"profile: language={profile.language or 'auto'} mode={mode}")

    plans = plan_channels(profile)
    # Pace file replay to wall-clock only when it feeds the live pass, so
    # `--replay` demonstrates captions at meeting cadence; batch just dumps it.
    provider = _make_provider(replay, plans, paced=live)
    out_dir = out or Path.cwd()
    stem = f"meeting-{time.strftime('%Y%m%d-%H%M%S')}"

    started = time.monotonic()
    asr, vad, diarizer = _load_backends(need_diarizer=any(p.num_speakers != 1 for p in plans))
    reid = _load_reid(enabled=use_reid, threshold=reid_threshold) if diarizer is not None else None
    if reid is not None:
        click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
    recorder = MeetingRecorder(
        profile, asr=asr, vad=vad, diarizer=diarizer, reid=reid, language=profile.language
    )

    tee = _make_tee(record_audio, out_dir, stem, plans)

    # The full-screen TUI owns the terminal, so it can only run on a real TTY and
    # unless the user forced the plain stream (or turned live off entirely).
    use_tui = live and not plain and _stdout_is_tty()
    channels = ", ".join(p.channel.value for p in plans)
    if not use_tui:  # the TUI header shows REC / elapsed instead of this hint
        stop_hint = f"stops after {max_seconds:g}s" if max_seconds else "press Ctrl-C to stop"
        click.echo(f"capturing: {channels} ({stop_hint} and transcribe)")
    try:
        transcript = _run_meeting(
            recorder,
            provider,
            live=live,
            use_tui=use_tui,
            profile=profile,
            on_frame=tee.add if tee else None,
            out_dir=out_dir,
            stem=stem,
            flush_interval=flush_interval,
            max_seconds=max_seconds,
        )
    finally:
        if tee is not None:
            tee.close()
            click.echo(f"recorded audio: {tee.path}")

    if transcript is None:
        # Defensive: a live view exited without producing a transcript. There is
        # nothing authoritative to write; leave any .partial checkpoint in place
        # for recovery rather than deleting it or crashing on None.
        raise click.ClickException(
            "meeting ended before a transcript was produced; any .partial checkpoint is kept"
        )

    paths = _write_transcript(transcript, out_dir, stem, write_formats)
    _cleanup_checkpoints(out_dir, stem)  # the final transcript supersedes them
    elapsed = time.monotonic() - started
    found = len({e.speaker for e in transcript.entries})
    click.echo(f"speakers: {found} found")
    click.echo(f"wrote {', '.join(p.name for p in paths)} ({elapsed:.1f}s)")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _run_meeting(
    recorder,
    provider,
    *,
    live: bool,
    use_tui: bool,
    profile: MeetingProfile,
    on_frame,
    out_dir: Path,
    stem: str,
    flush_interval: float,
    max_seconds: float | None,
) -> Transcript:
    """Run the capture session through the right live view and return the transcript.

    Three shapes behind one call:

    - **TUI** (live, on a TTY, not ``--plain``): the Textual view runs the app on
      this thread while the meeting runs on a background thread; its quit binding
      crosses to ``provider.stop`` to end capture. Checkpoints are written silently
      (the TUI owns the screen).
    - **Plain live** (live, no TTY or ``--plain``): the meeting runs on this thread
      and streams committed captions to stdout; checkpoints written silently.
    - **Batch** (``--no-live``): no live pass; status and checkpoint notices echo
      as before.
    """
    if use_tui:
        from stenograf.tui import TextualLiveView

        view = TextualLiveView(profile, language=profile.language, stop=provider.stop)
        return view.serve(
            lambda: recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                on_checkpoint=_checkpoint_writer(out_dir, stem),
                checkpoint_interval=flush_interval,
                max_seconds=max_seconds,
            )
        )
    if live:
        from stenograf.view import PlainLiveView

        with PlainLiveView() as view:
            return recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                on_checkpoint=_checkpoint_writer(out_dir, stem),
                checkpoint_interval=flush_interval,
                max_seconds=max_seconds,
            )
    return recorder.run(
        provider,
        on_frame=on_frame,
        on_status=lambda msg: click.echo(f"  {msg}"),
        on_checkpoint=_checkpoint_writer(out_dir, stem, announce=lambda m: click.echo(f"  {m}")),
        checkpoint_interval=flush_interval,
        max_seconds=max_seconds,
    )


def _stdout_is_tty() -> bool:
    """Whether stdout is an interactive terminal (a seam so the view choice is testable)."""
    return sys.stdout.isatty()


def _checkpoint_writer(
    out_dir: Path, stem: str, announce: Callable[[str], None] | None = None
) -> Callable[[Transcript], None]:
    """Build the ``on_checkpoint`` sink that writes the ``.partial`` crash file.

    Live views keep the caption stream clean (``announce=None`` → write silently);
    the batch path narrates each write, as it always has. The final transcript
    supersedes these files, which ``_cleanup_checkpoints`` then removes.
    """

    def on_checkpoint(transcript: Transcript) -> None:
        # Checkpoints are crash recovery, always plain md+json — subtitles of a
        # partial transcript are pointless and the final write supersedes these.
        md = _write_transcript(transcript, out_dir, f"{stem}.partial")[0]
        if announce is not None:
            announce(f"checkpoint: {md.name} ({len(transcript.entries)} entries)")

    return on_checkpoint


def _make_tee(record_audio: str | None, out_dir: Path, stem: str, plans):
    """Create the audio tee if --record-audio was given, with a loud banner."""
    if record_audio is None:
        return None
    from stenograf.recording import WavTee

    path = out_dir / f"{stem}.wav" if record_audio == _RECORD_DEFAULT else Path(record_audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    tee = WavTee(path, {p.channel for p in plans})
    click.secho(
        f"● RECORDING AUDIO to {path} — raw audio is being written to disk",
        fg="red",
        bold=True,
    )
    return tee


def _make_provider(replay: str | None, plans, *, paced: bool = False):
    """Build the capture provider: file replay if given, else the native helper."""
    from stenograf.capture.base import Channel

    if replay is not None:
        from stenograf.capture.file import FileCaptureProvider

        paths = [p.strip() for p in replay.split(",") if p.strip()]
        channel_order = [Channel.MIC, Channel.SYSTEM]
        sources = dict(zip(channel_order, paths, strict=False))
        planned = {p.channel for p in plans}
        ignored = [ch.value for ch in sources if ch not in planned]
        if ignored:
            click.echo(f"note: ignoring replay for un-recorded channel(s): {', '.join(ignored)}")
        return FileCaptureProvider(
            {ch: p for ch, p in sources.items() if ch in planned}, paced=paced
        )

    if sys.platform != "darwin":
        raise click.ClickException(
            "live capture is macOS-only for now; on other platforms transcribe a "
            "recorded file with `steno transcribe`, or use `steno start --replay`."
        )
    from stenograf.capture.macos import HelperNotFoundError, MacOSCaptureProvider

    try:
        return MacOSCaptureProvider()
    except HelperNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--lang",
    type=click.Choice([lang.value for lang in Language]),
    default=None,
    help="Audio language (transcript metadata; the default ASR model is "
    "multilingual). Omit to auto-detect from the transcript.",
)
@click.option(
    "--speakers",
    type=click.IntRange(1, 16),
    default=None,
    help="Known speaker count (the biggest diarization accuracy lever); "
    "1 skips diarization, omit to estimate.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory [default: next to the input file].",
)
@click.option(
    "--reid/--no-reid",
    "use_reid",
    default=True,
    help="Relabel diarized speakers to saved profile names when their voice matches "
    "(cross-meeting re-identification). No effect without enrolled profiles.",
)
@click.option(
    "--reid-threshold",
    type=click.FloatRange(0, 1),
    default=None,
    help="Cosine similarity required to match a saved profile [default: 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=",".join(_DEFAULT_FORMATS),
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, srt, vtt "
    "[default: md,json]. srt/vtt re-flow speaker turns into subtitle cues.",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def transcribe(
    audio_file: Path,
    lang: str | None,
    speakers: int | None,
    out: Path | None,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str,
    print_markdown: bool,
) -> None:
    """Transcribe an audio/video file (batch finalize pass).

    Writes <name>.transcript.md and <name>.transcript.json next to the
    input (or into --out); --format also emits srt/vtt subtitles. This is
    the same pipeline a live meeting runs on stop; use it for recorded
    meetings or re-transcription.
    """
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.pipeline import finalize_channel, relabel_speakers

    started = time.monotonic()
    write_formats = _parse_formats(formats)
    language = Language(lang) if lang else None

    samples = load_audio(audio_file)
    duration = len(samples) / SAMPLE_RATE
    click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)})")

    asr, vad, diarizer = _load_backends(need_diarizer=speakers != 1)
    reid = _load_reid(enabled=use_reid, threshold=reid_threshold) if diarizer is not None else None
    if reid is not None:
        click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")

    def progress(stage: str, done: int, total: int) -> None:
        if stage == "asr" and done == 0:
            click.echo(f"transcribing {total} windows")
        elif stage == "diarization":
            click.echo(f"diarizing ({speakers or 'estimating'} speakers)")

    entries = relabel_speakers(
        finalize_channel(
            samples,
            asr=asr,
            language=language,
            vad=vad,
            diarizer=diarizer,
            num_speakers=speakers,
            reid=reid,
            on_progress=progress,
        )
    )
    if language is None:
        from stenograf.lid import detect_language

        language = detect_language(" ".join(e.text for e in entries))
        if language is not None:
            click.echo(f"language: detected {language.value}")
    transcript = Transcript(
        language=language, profile=MeetingProfile(language=language), entries=entries
    )

    paths = _write_transcript(transcript, out or audio_file.parent, audio_file.stem, write_formats)
    elapsed = time.monotonic() - started
    speed = duration / elapsed if elapsed else 0.0
    found = len({e.speaker for e in entries})
    click.echo(f"speakers: {found} found" if speakers is None else f"speakers: {speakers} given")
    click.echo(f"wrote {', '.join(p.name for p in paths)} ({elapsed:.1f}s, {speed:.1f}x realtime)")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _load_backends(*, need_diarizer: bool):
    """Load the finalize backends (ASR, VAD, and optionally the diarizer).

    Shared by ``start`` and ``transcribe`` so both use the same committed
    defaults (parakeet-mlx, Silero VAD, sherpa-onnx diarization).
    """
    from stenograf import models
    from stenograf.asr.parakeet import ParakeetMLXBackend
    from stenograf.vad import SileroVAD

    asr = ParakeetMLXBackend()
    click.echo(f"asr: loading {asr.model_id}")
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD, _model_progress))
    diarizer = _load_diarizer(need=need_diarizer)
    return asr, vad, diarizer


def _load_diarizer(*, need: bool = True):
    """Build the sherpa-onnx diarizer (or ``None`` when a channel is single-speaker).

    A seam of its own so ``steno profiles enroll`` computes its voiceprints with
    the exact same embedding path the finalize pass uses at match time (the two
    must agree for the cosine match to mean anything), and so tests can inject a
    fake without a real ONNX model.
    """
    if not need:
        return None
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    return SherpaOnnxDiarizer(progress=_model_progress)


def _load_reid(*, enabled: bool, threshold: float | None):
    """Build the cross-meeting re-ID resolver from the saved profile store, or ``None``.

    Returns ``None`` when re-ID is turned off or the store holds no profiles for
    the active embedding model — so the finalize pass is byte-for-byte unchanged
    without enrolled profiles (match-only, zero behaviour change; PLAN.md Phase 3
    Task 1b/1c). ``threshold=None`` uses the store default (0.5).
    """
    if not enabled:
        return None
    from stenograf import models
    from stenograf.profiles import ProfileStore, SpeakerReID

    store = ProfileStore.load()
    model = models.SPEAKER_EMBEDDING.name
    if not store.for_model(model):
        return None
    return SpeakerReID(store, model, threshold=threshold)


def _write_transcript(
    transcript: Transcript,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...] | list[str] = _DEFAULT_FORMATS,
) -> list[Path]:
    """Write the transcript in each requested format; returns the written paths.

    Markdown + JSON are the default (the only files stenograf emits unless the
    user asks for subtitles); SRT/VTT are opt-in via ``--format``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.transcript.{fmt}"
        path.write_text(getattr(transcript, _FORMATS[fmt])())
        paths.append(path)
    return paths


def _cleanup_checkpoints(out_dir: Path, stem: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for suffix in (".partial.transcript.md", ".partial.transcript.json"):
        (out_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def _model_progress(name: str, done: int, total: int) -> None:
    if total and done == 0:
        click.echo(f"model: downloading {name} ({total >> 20} MB)")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@main.command()
def doctor() -> None:
    """Check this machine's readiness (permissions, OS version, models)."""
    checks = run_checks()
    for check in checks:
        symbol = click.style("✓", fg="green") if check.ok else click.style("✗", fg="red")
        click.echo(f" {symbol} {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        raise SystemExit(1)


@main.group()
def profiles() -> None:
    """Manage saved speaker voiceprints for cross-meeting re-identification.

    Enroll a voice once and every later meeting relabels that speaker
    automatically (``steno start``/``transcribe`` unless ``--no-reid``).
    """


@profiles.command("list")
def profiles_list() -> None:
    """List enrolled speaker profiles."""
    from stenograf import models
    from stenograf.profiles import ProfileStore, default_store_path

    store = ProfileStore.load()
    all_profiles = store.profiles()
    if not all_profiles:
        click.echo(f"no speaker profiles yet ({default_store_path()})")
        click.echo("enroll one with: steno profiles enroll NAME sample.wav")
        return
    active_model = models.SPEAKER_EMBEDDING.name
    click.echo(f"speaker profiles ({default_store_path()}):")
    for p in sorted(all_profiles, key=lambda p: (p.embedding_model, p.name.lower())):
        noun = "sample" if p.samples == 1 else "samples"
        # A profile made under a different embedding model can never match a
        # cluster from the current one — flag it so the count is not misleading.
        tag = "" if p.embedding_model == active_model else "  [inactive: other embedding model]"
        click.echo(f"  {p.name}  ({p.samples} {noun}){tag}")


@profiles.command("enroll")
@click.argument("name")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--speakers",
    type=click.IntRange(1, 16),
    default=1,
    show_default=True,
    help="How many speakers are in the clip; it is diarized into this many and one "
    "cluster is enrolled.",
)
@click.option(
    "--speaker",
    "cluster",
    default=None,
    metavar="S<n>",
    help="Which diarized cluster to enroll when the clip has several speakers "
    "(re-run without it to see the choices). Ignored for a single-speaker clip.",
)
@click.option(
    "--reinforce",
    is_flag=True,
    help="Fold this sample into an existing profile's voiceprint instead of creating a new one.",
)
def profiles_enroll(
    name: str, audio_file: Path, speakers: int, cluster: str | None, reinforce: bool
) -> None:
    """Enroll speaker NAME from a voice sample in AUDIO_FILE.

    Give a short clip in which NAME is the only speaker (the default), or a
    multi-speaker recording (e.g. a meeting saved with ``--record-audio``) plus
    ``--speakers N`` and ``--speaker S<n>`` to enroll one person from it. The
    voiceprint is computed exactly the way meetings embed their clusters, so
    future meetings relabel this speaker automatically.
    """
    from stenograf import models
    from stenograf.audio import load_audio
    from stenograf.profiles import ProfileStore

    samples = load_audio(audio_file)
    diarizer = _load_diarizer(need=True)
    result = diarizer.diarize_with_embeddings(samples, num_speakers=speakers)
    if not result.embeddings:
        raise click.ClickException(
            f"no embeddable speech found in {audio_file.name}; is it silent or too short?"
        )
    embedding = _choose_cluster(result.embeddings, result.turns, cluster)

    model = models.SPEAKER_EMBEDDING.name
    store = ProfileStore.load()
    existing = store.get(name, model)
    if reinforce:
        if existing is None:
            raise click.ClickException(
                f"no profile named {name!r} to reinforce; drop --reinforce to create it."
            )
        updated = store.reinforce(existing, embedding)
        store.save()
        click.echo(f"reinforced {name!r} ({updated.samples} samples)")
        return
    if existing is not None:
        raise click.ClickException(
            f"a profile named {name!r} already exists; use --reinforce to add this sample "
            "to it, or remove it first with `steno profiles remove`."
        )
    store.enroll(name, embedding, model)
    store.save()
    click.echo(f"enrolled {name!r} from {audio_file.name}")


def _choose_cluster(embeddings, turns, cluster: str | None):
    """Pick one cluster's embedding, or raise a helpful error when it is ambiguous."""
    if cluster is not None:
        if cluster not in embeddings:
            available = ", ".join(sorted(embeddings)) or "none"
            raise click.ClickException(
                f"no cluster {cluster!r} in the clip; available: {available}"
            )
        return embeddings[cluster]
    if len(embeddings) == 1:
        return next(iter(embeddings.values()))
    durations: dict[str, float] = {}
    for turn in turns:
        durations[turn.speaker] = durations.get(turn.speaker, 0.0) + (turn.end - turn.start)
    listing = "\n".join(f"  {c}  ({durations.get(c, 0.0):.1f}s speech)" for c in sorted(embeddings))
    raise click.ClickException(
        "the clip has several speakers; re-run with --speaker to pick one:\n" + listing
    )


@profiles.command("rename")
@click.argument("old")
@click.argument("new")
def profiles_rename(old: str, new: str) -> None:
    """Rename speaker profile OLD to NEW."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(old, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {old!r}")
    try:
        store.rename(profile, new)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.save()
    click.echo(f"renamed {old!r} → {new!r}")


@profiles.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def profiles_remove(name: str, yes: bool) -> None:
    """Delete speaker profile NAME."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(name, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {name!r}")
    if not yes:
        click.confirm(f"delete speaker profile {name!r}?", abort=True)
    store.remove(profile)
    store.save()
    click.echo(f"removed {name!r}")
