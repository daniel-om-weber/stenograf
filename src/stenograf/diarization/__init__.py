from stenograf.diarization.base import Diarizer, SpeakerTurn

__all__ = ["Diarizer", "SpeakerTurn", "build_diarizer"]


def build_diarizer(progress=None) -> Diarizer:
    """Build the committed diarization stack — the selection seam.

    When the stenodiar helper is present, unknown speaker counts go through
    speakrs' VBx estimation and explicit counts through sherpa; without it,
    sherpa handles both (its estimate mode over-splits badly — the helper is
    what makes "don't specify a count" usable). One function, so ``steno
    profiles enroll`` computes its voiceprints with the exact same embedding
    path the finalize pass uses at match time — the two must agree for the
    cosine match to mean anything. ``progress`` is a
    :data:`stenograf.models.ProgressHook` for first-run model downloads."""
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.diarization.speakrs import (
        DiarizerHelperNotFoundError,
        SpeakrsCliDiarizer,
        find_stenodiar,
    )

    sherpa = SherpaOnnxDiarizer(progress=progress)
    try:
        find_stenodiar()
    except DiarizerHelperNotFoundError:
        return sherpa
    return SpeakrsCliDiarizer(sherpa)
