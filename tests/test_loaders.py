"""``load_backends``: which model files a run fetches before it starts."""

from __future__ import annotations

import pytest
from conftest import FakeASR

from stenograf import assets, loaders


@pytest.fixture
def fetched(monkeypatch):
    """Model-free backends; returns the assets ``load_backends`` fetched."""
    import stenograf.asr as asr
    import stenograf.vad as vad
    from stenograf import doctor

    names: list[str] = []

    def fetch(asset, progress=None):
        names.append(asset.name)
        return f"/cache/{asset.name}"

    monkeypatch.setattr(doctor, "installed", lambda module: True)
    monkeypatch.setattr(asr, "create_backend", lambda name, **kwargs: FakeASR())
    monkeypatch.setattr(vad, "SileroVAD", lambda path: object())
    monkeypatch.setattr(assets, "fetch", fetch)
    monkeypatch.setattr(loaders, "load_diarizer", lambda announce=None: object())
    return names


def test_a_diarized_run_fetches_its_models_before_it_starts(fetched):
    _, _, diarizer = loaders.load_backends(need_diarizer=True, announce=lambda line: None)
    assert diarizer is not None
    wanted = [a.name for a in (assets.SILERO_VAD, *assets.NEMOTRON, assets.SPEAKER_EMBEDDING)]
    assert fetched == wanted


def test_an_undiarized_run_fetches_only_the_speech_detector(fetched):
    _, _, diarizer = loaders.load_backends(need_diarizer=False, announce=lambda line: None)
    assert diarizer is None
    assert fetched == [assets.SILERO_VAD.name]


def test_a_failed_diarizer_download_warns_and_the_meeting_goes_on(fetched, monkeypatch):
    def offline(asset, progress=None):
        if asset in assets.NEMOTRON:
            raise OSError("network unreachable")
        return f"/cache/{asset.name}"

    monkeypatch.setattr(assets, "fetch", offline)
    said: list[str] = []
    _, _, diarizer = loaders.load_backends(need_diarizer=True, announce=said.append)
    assert diarizer is not None, "the stop still diarizes if the files arrive by then"
    assert any("network unreachable" in line for line in said)
