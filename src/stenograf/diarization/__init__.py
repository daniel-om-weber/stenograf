from __future__ import annotations

from typing import TYPE_CHECKING

from stenograf.diarization.base import Diarizer, SpeakerTurn

if TYPE_CHECKING:
    from stenograf.assets import ProgressHook

__all__ = ["Diarizer", "SpeakerTurn", "build_diarizer"]


def build_diarizer(progress: ProgressHook | None = None) -> Diarizer:
    """Build the committed diarizer — the selection seam.

    NVIDIA Nemotron 3 Diarization counts the speakers itself; a stated count is
    applied afterwards as a cap (:class:`~stenograf.diarization.nemotron.NemotronDiarizer`).
    One function, so ``steno profiles enroll`` computes its voiceprints with
    the exact same embedding path the finalize pass uses at match time — the
    two must agree for the cosine match to mean anything. ``progress`` is a
    :data:`stenograf.assets.ProgressHook` for first-run model downloads."""
    from stenograf.diarization.nemotron import NemotronDiarizer

    return NemotronDiarizer(progress=progress)
