"""Backend and capture-provider factory: settings/env in, loaded objects out.

One place turns backend *names* into loaded backend objects with the
committed defaults (parakeet, Silero VAD, sherpa+speakrs diarization) and
builds the platform capture provider, so ``start``, ``transcribe``,
``setup``, ``profiles enroll`` and the app can never disagree about
selection, gating, or first-run downloads. User errors raise the library's
own types — :class:`BackendUnavailableError` here,
:class:`~stenograf.capture.base.CaptureUnavailableError` from the capture
stack — and the CLI maps them to click errors at its command boundary; the
pure selection seams live one layer down
(:func:`stenograf.diarization.build_diarizer`, the ASR registry).

Both front-ends call these factories. The Qt app runs them inside a GUI,
where progress must not go through click — the process may own no usable
stdio at all (a Windows ``pythonw`` launch, or the retired Textual TUI whose
stdio proxy click probed to death with EBADF). Every announcing entry point
therefore takes ``announce``: ``None`` keeps the click-echoed CLI behaviour,
a callable routes the same lines to the caller's sink (the meeting screen's
status line).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import click

from stenograf.asr.base import BackendUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from stenograf.asr.base import ASRBackend
    from stenograf.capture.base import CaptureProvider, Channel
    from stenograf.diarization.base import Diarizer
    from stenograf.models import ProgressHook
    from stenograf.profiles import SpeakerReID
    from stenograf.session import ChannelPlan
    from stenograf.vad import SileroVAD
    from stenograf.view import LiveView

    Announce = Callable[[str], None]


def _say(announce: Announce | None, message: str) -> None:
    """One progress line: the caller's sink, or click for the CLI (module docstring)."""
    if announce is not None:
        announce(message)
    else:
        click.echo(message)


_PROBLEM_MARKERS = ("fatal", "warning", "error", "denied", "failed", "died", "silence")
"""Substrings (case-insensitive) that mark a capture-transport line as a
problem the user must see, vs. routine chatter (formats, started/stopped)."""


class CaptureLog:
    """Capture-transport diagnostics for a run whose owner is not the terminal.

    The native transport (stenocap) normally inherits stderr, so its
    status lines land on the terminal — right for the CLI, but the Qt meeting
    screen (:mod:`stenograf.flow`) has no terminal to show them on, and its
    process may not even have a usable stderr. Passed as ``on_log`` to
    :func:`make_provider`, this sink buffers every line instead. Lines that
    look like problems (permission denied, a stream dying) are forwarded to
    the attached ``view``'s status surface the moment they arrive, so a
    mid-meeting capture failure stays as visible as it was on stderr; the
    ``lines`` buffer keeps the full transport chatter for debugging.

    Called from the transports' relay threads; the list append is atomic and
    ``view.error`` marshals itself onto the UI loop, so no lock is needed.
    """

    def __init__(self, view: LiveView | None = None) -> None:
        self.lines: list[str] = []
        self.view = view
        """A LiveView (or None until one is attached); gets problem lines."""

    def __call__(self, line: str) -> None:
        self.lines.append(line)
        if self.view is not None and self._is_problem(line):
            self.view.error(line)

    @staticmethod
    def _is_problem(line: str) -> bool:
        lowered = line.lower()
        return any(marker in lowered for marker in _PROBLEM_MARKERS)


def download_progress(announce: Announce | None) -> ProgressHook:
    """ProgressHook that announces a model download once, at its start."""

    def hook(name: str, done: int, total: int) -> None:
        if total and done == 0:
            _say(announce, f"model: downloading {name} ({total >> 20} MB)")

    return hook


def model_progress(name: str, done: int, total: int) -> None:
    """The CLI-flavored ProgressHook (kept for the non-TUI call sites)."""
    download_progress(None)(name, done, total)


def load_backends(
    *,
    need_diarizer: bool,
    asr_backend: str | None = None,
    asr_provider: str | None = None,
    glossary: Sequence[str] = (),
    attendee_names: Sequence[str] = (),
    boost: float | None = None,
    announce: Announce | None = None,
) -> tuple[ASRBackend, SileroVAD, Diarizer | None]:
    """Load the finalize backends (ASR, VAD, and optionally the diarizer).

    Shared by ``start`` and ``transcribe`` so both use the same committed
    defaults. ``asr_backend`` and ``asr_provider`` are the ``[asr]`` settings;
    ``STENOGRAF_ASR_BACKEND`` / ``STENOGRAF_ASR_PROVIDER`` still override them.

    The run's ``glossary`` and ``attendee_names`` are compiled into a boosting tree
    that steers the decoder *while* it transcribes — see ``stenograf.asr.biasing``.
    They are passed at *load* time, not per call, because the tree is spliced into
    the model's decode loop. The text post-correction in ``stenograf.glossary``
    still runs afterwards, and the two are complementary: biasing wins the terms
    the decoder can nearly hear, post-correction catches the ones it heard as some
    other word entirely.
    """
    from stenograf import models
    from stenograf.asr import create_backend
    from stenograf.asr.providers import default_provider_name, validate_provider_name
    from stenograf.asr.registry import default_backend_name, get_spec
    from stenograf.doctor import installed
    from stenograf.vad import SileroVAD

    # The selection seam; a Linux backend registers alongside. Gate on the
    # spec's requires (as doctor and the model prefetch already do) so an
    # unknown or uninstalled backend is a clean user error, not an import
    # traceback.
    name = default_backend_name(asr_backend)
    try:
        spec = get_spec(name)
        provider = validate_provider_name(default_provider_name(asr_provider))
    except ValueError as exc:
        raise BackendUnavailableError(str(exc)) from exc
    missing = [module for module in spec.requires if not installed(module)]
    if missing:
        raise BackendUnavailableError(
            f"ASR backend {spec.label} is not installed here (missing: "
            f"{', '.join(missing)}) — reinstall stenograf, or select another backend "
            "via [asr] backend in settings.toml or STENOGRAF_ASR_BACKEND"
        )
    from stenograf.asr.biasing import boost_terms

    kwargs: dict[str, object] = {"glossary": boost_terms(glossary, attendee_names)}
    if boost is not None:
        kwargs["boost"] = boost
    asr = create_backend(name, **kwargs)
    # Only an ORT-backed backend is provider-configurable (provider != None);
    # a configured provider on a backend with its own runtime (MLX) is noted,
    # not an error, so one settings file can serve a mac and a Windows box.
    if asr.provider is not None:
        asr.provider = provider
    elif provider != "cpu":
        _say(announce, f"asr: provider {provider!r} ignored — {spec.label} manages its own runtime")
    _say(announce, f"asr: loading {asr.model_id or asr.name}")
    asr.load()  # an explicit accelerator that can't deliver raises here, loudly
    vad = SileroVAD(models.fetch(models.SILERO_VAD, download_progress(announce)))
    diarizer = load_diarizer(announce=announce) if need_diarizer else None
    return asr, vad, diarizer


def load_diarizer(*, announce: Announce | None = None) -> Diarizer:
    """The committed diarization stack with download progress attached.

    A seam of its own (over calling :func:`~stenograf.diarization.build_diarizer`
    inline) so tests can inject a fake without a real ONNX model."""
    from stenograf.diarization import build_diarizer

    return build_diarizer(progress=download_progress(announce))


def load_reid(
    *, enabled: bool, threshold: float | None, store_path: Path | None = None
) -> SpeakerReID | None:
    """Build the cross-meeting re-ID resolver from the saved profile store, or ``None``.

    Returns ``None`` when re-ID is turned off or the store holds no profiles for
    the active embedding model — so the finalize pass is byte-for-byte unchanged
    without enrolled profiles (match-only, zero behaviour change).
    ``threshold=None`` uses the store default (0.5). ``store_path``
    (``--profile-store`` / ``MeetingProfile.speaker_profile_store``) overrides the
    default store location.
    """
    if not enabled:
        return None
    from stenograf import models
    from stenograf.profiles import ProfileStore, SpeakerReID

    store = ProfileStore.load(store_path)
    model = models.SPEAKER_EMBEDDING.name
    if not store.for_model(model):
        return None
    return SpeakerReID(store, model, threshold=threshold)


def make_provider(
    replay: str | None,
    plans: Sequence[ChannelPlan],
    *,
    paced: bool = False,
    aec: bool = True,
    aec_dump: Path | None = None,
    announce: Announce | None = None,
    on_log: Announce | None = None,
) -> CaptureProvider:
    """Build the capture provider: file replay if given, else the native helper.

    When both channels are captured, the mic is echo-cancelled against the system
    channel — without it, remote participants coming out of the speakers land on
    the mic channel and get transcribed as the local speaker. ``aec_dump`` wraps
    even with ``--no-aec`` so the eval rig can record the uncancelled baseline.

    ``on_log`` is the capture transports' diagnostic sink (a :class:`CaptureLog`
    for the Qt meeting screen). ``None`` keeps the transports' stderr on the
    terminal — right for the CLI, invisible in a GUI process.
    """
    from stenograf.capture.base import Channel

    provider = _base_provider(replay, plans, paced=paced, announce=announce, on_log=on_log)
    channels = {plan.channel for plan in plans}
    if (aec or aec_dump is not None) and {Channel.MIC, Channel.SYSTEM} <= channels:
        from stenograf.aec import EchoCancellingProvider

        return EchoCancellingProvider(provider, cancel=aec, dump_dir=aec_dump)
    return provider


def _base_provider(
    replay: str | None,
    plans: Sequence[ChannelPlan],
    *,
    paced: bool = False,
    announce: Announce | None = None,
    on_log: Announce | None = None,
) -> CaptureProvider:
    from stenograf.capture.base import Channel

    if replay is not None:
        from stenograf.capture.file import FileCaptureProvider

        paths = [p.strip() for p in replay.split(",") if p.strip()]
        channel_order = [Channel.MIC, Channel.SYSTEM]
        sources = dict(zip(channel_order, paths, strict=False))
        planned = {p.channel for p in plans}
        ignored = [ch.value for ch in sources if ch not in planned]
        if ignored:
            _say(
                announce,
                f"note: ignoring replay for un-recorded channel(s): {', '.join(ignored)}",
            )
        return FileCaptureProvider(
            {ch: p for ch, p in sources.items() if ch in planned}, paced=paced
        )

    from stenograf.capture.base import CaptureUnavailableError

    if sys.platform == "darwin":
        from stenograf.capture.macos import MacOSCaptureProvider

        # A missing helper raises HelperNotFoundError, a CaptureUnavailableError.
        return MacOSCaptureProvider(on_log=on_log)

    if sys.platform.startswith("linux"):
        from stenograf.capture.linux import LinuxCaptureProvider, default_devices

        return _native_provider(LinuxCaptureProvider, default_devices, plans, announce, on_log)

    if sys.platform == "win32":
        from stenograf.capture.windows import WindowsCaptureProvider, default_devices

        return _native_provider(WindowsCaptureProvider, default_devices, plans, announce, on_log)

    raise CaptureUnavailableError(
        "live capture is supported on macOS, Linux, and Windows; here, transcribe "
        "a recorded file with `steno transcribe`, or use `steno start --replay`."
    )


def _native_provider(
    provider_cls: Callable[..., CaptureProvider],
    default_devices: Callable[[set[Channel]], dict[Channel, str]],
    plans: Sequence[ChannelPlan],
    announce: Announce | None = None,
    on_log: Announce | None = None,
) -> CaptureProvider:
    """Construct a provider for a platform that offers a device *choice*.

    Resolves the default devices up front so a broken audio stack fails before
    capture (and models) start, and says what will be recorded — the
    monitor-of-default-sink (Linux) / loopback-of-default-output (Windows)
    choice is invisible otherwise. macOS has no equivalent: its helper owns
    device selection.

    A broken audio stack raises CaptureUnavailableError from either call.
    """
    provider = provider_cls(on_log=on_log)
    devices = default_devices({p.channel for p in plans})
    for channel, device in devices.items():
        _say(announce, f"capture: {channel.value} ← {device}")
    return provider


def prefetch_models() -> None:
    """Download the VAD/diarization assets and the ASR weights now, not mid-meeting."""
    from stenograf import models
    from stenograf.asr import backend_model_id, create_backend, get_spec
    from stenograf.doctor import installed

    for asset in (models.SILERO_VAD, models.PYANNOTE_SEGMENTATION, models.SPEAKER_EMBEDDING):
        if models.cached_path(asset) is not None:
            click.echo(f"model: {asset.name} already cached")
        else:
            models.fetch(asset, model_progress)

    # Gate on the backend's runtime deps the way doctor does: the backend
    # *module* imports fine everywhere (its heavy imports live inside load()),
    # so a try/except around create_backend() would not catch a missing MLX.
    spec = get_spec()
    if not all(installed(module) for module in spec.requires):
        click.echo(f"ASR backend {spec.label} is not installed here; skipping its weights")
        return
    click.echo(f"model: fetching + loading ASR weights ({backend_model_id(spec)})")
    backend = create_backend()
    backend.load()  # pulls from HuggingFace on first run, then verifies it loads
    backend.unload()
