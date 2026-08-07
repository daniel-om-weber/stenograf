from __future__ import annotations

from typing import TYPE_CHECKING

from stenograf.diarization.base import Diarizer, SpeakerTurn

if TYPE_CHECKING:
    from stenograf.assets import ProgressHook

__all__ = ["Diarizer", "SpeakerTurn", "build_diarizer"]


def build_diarizer(progress: ProgressHook | None = None) -> Diarizer:
    """Build the committed diarization stack — the selection seam.

    When the stenodiar helper is present, unknown speaker counts go through
    speakrs' VBx estimation and explicit counts through the owned loop
    (``OwnDiarizer``: ward + the gated fold, 2026-08-07 — ``eval/README.md``);
    without it, the owned loop handles both (its threshold estimate mode
    over-splits badly — the helper is what makes "don't specify a count"
    usable). One function, so ``steno profiles enroll`` computes its
    voiceprints with the exact same embedding path the finalize pass uses at
    match time — the two must agree for the cosine match to mean anything.
    ``progress`` is a :data:`stenograf.assets.ProgressHook` for first-run
    model downloads."""
    from stenograf.diarization.loop import OwnDiarizer
    from stenograf.diarization.speakrs import (
        DiarizerHelperNotFoundError,
        SpeakrsCliDiarizer,
        find_stenodiar,
    )

    own = OwnDiarizer(progress=progress)
    try:
        find_stenodiar()
    except DiarizerHelperNotFoundError:
        return own
    return SpeakrsCliDiarizer(own)
