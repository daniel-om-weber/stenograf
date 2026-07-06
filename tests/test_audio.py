import wave

import numpy as np

from stenograf.audio import SAMPLE_RATE, load_audio, to_float32


def write_wav(path, samples: np.ndarray, rate: int = SAMPLE_RATE, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def test_to_float32_scales_int16():
    out = to_float32(np.array([0, 16384, -32768], dtype=np.int16))
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.5, -1.0])


def test_load_mono_16k_wav(tmp_path):
    samples = (np.sin(np.linspace(0, 100, SAMPLE_RATE)) * 10000).astype(np.int16)
    path = tmp_path / "tone.wav"
    write_wav(path, samples)
    loaded = load_audio(path)
    assert loaded.dtype == np.float32
    assert len(loaded) == SAMPLE_RATE
    assert np.allclose(loaded, to_float32(samples))


def test_load_stereo_wav_downmixes(tmp_path):
    left = np.full(1000, 1000, dtype=np.int16)
    right = np.full(1000, 3000, dtype=np.int16)
    interleaved = np.column_stack([left, right]).ravel()
    path = tmp_path / "stereo.wav"
    write_wav(path, interleaved, channels=2)
    loaded = load_audio(path)
    assert len(loaded) == 1000
    assert np.allclose(loaded, 2000 / 32768.0)
