"""Backend factory for the CLI: settings/env in, loaded backends out.

One place turns backend *names* into loaded backend objects with the
committed defaults (parakeet, Silero VAD, sherpa+speakrs diarization), so
``start``, ``transcribe``, ``setup``, and ``profiles enroll`` can never
disagree about selection, gating, or first-run downloads. Like
:mod:`stenograf.view` this is CLI-support code — it reports progress via
click and raises ``click.ClickException`` for user errors; the pure
selection seams it drives live in the library
(:func:`stenograf.diarization.build_diarizer`, the ASR registry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from pathlib import Path


def model_progress(name: str, done: int, total: int) -> None:
    """ProgressHook that announces a model download once, at its start."""
    if total and done == 0:
        click.echo(f"model: downloading {name} ({total >> 20} MB)")


def load_backends(
    *, need_diarizer: bool, asr_backend: str | None = None, asr_provider: str | None = None
):
    """Load the finalize backends (ASR, VAD, and optionally the diarizer).

    Shared by ``start`` and ``transcribe`` so both use the same committed
    defaults. ``asr_backend`` and ``asr_provider`` are the ``[asr]`` settings;
    ``STENOGRAF_ASR_BACKEND`` / ``STENOGRAF_ASR_PROVIDER`` still override them.
    """
    from stenograf import models
    from stenograf.asr import create_backend
    from stenograf.asr.providers import (
        PROVIDER_LABELS,
        default_provider_name,
        validate_provider_name,
    )
    from stenograf.asr.registry import default_backend_name, get_spec
    from stenograf.doctor import installed
    from stenograf.vad import SileroVAD

    # The selection seam; a Linux backend registers alongside. Gate on the
    # spec's requires (as doctor and the model prefetch already do) so an
    # unknown or uninstalled backend is a CLI error, not an import traceback.
    name = default_backend_name(asr_backend)
    try:
        spec = get_spec(name)
        provider = validate_provider_name(default_provider_name(asr_provider))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    missing = [module for module in spec.requires if not installed(module)]
    if missing:
        raise click.ClickException(
            f"ASR backend {spec.label} is not installed here (missing: "
            f"{', '.join(missing)}) — reinstall stenograf, or select another backend "
            "via [asr] backend in settings.toml or STENOGRAF_ASR_BACKEND"
        )
    asr = create_backend(name)
    # Duck-typed: only the ORT-backed backend carries a provider; a configured
    # provider on a backend with its own runtime (MLX) is noted, not an error,
    # so one settings file can serve a mac and a Windows box.
    if hasattr(asr, "provider"):
        asr.provider = provider  # pyright: ignore[reportAttributeAccessIssue] — PLAN-CLEANUP.md C4
    elif provider != "cpu":
        click.echo(f"asr: provider {provider!r} ignored — {spec.label} manages its own runtime")
    click.echo(f"asr: loading {getattr(asr, 'model_id', None) or asr.name}")
    asr.load()
    if fallback := getattr(asr, "provider_fallback", None):
        click.secho(
            f"asr: acceleration unavailable ({fallback}) — using CPU", fg="yellow", err=True
        )
    elif (active := getattr(asr, "active_provider", None)) not in (None, "cpu"):
        click.echo(f"asr: accelerated ({PROVIDER_LABELS[active]})")
    vad = SileroVAD(models.fetch(models.SILERO_VAD, model_progress))
    diarizer = load_diarizer() if need_diarizer else None
    return asr, vad, diarizer


def load_diarizer():
    """The committed diarization stack with CLI download progress attached.

    A seam of its own (over calling :func:`~stenograf.diarization.build_diarizer`
    inline) so tests can inject a fake without a real ONNX model."""
    from stenograf.diarization import build_diarizer

    return build_diarizer(progress=model_progress)


def load_reid(*, enabled: bool, threshold: float | None, store_path: Path | None = None):
    """Build the cross-meeting re-ID resolver from the saved profile store, or ``None``.

    Returns ``None`` when re-ID is turned off or the store holds no profiles for
    the active embedding model — so the finalize pass is byte-for-byte unchanged
    without enrolled profiles (match-only, zero behaviour change; PLAN.md Phase 3
    Task 1b/1c). ``threshold=None`` uses the store default (0.5). ``store_path``
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
