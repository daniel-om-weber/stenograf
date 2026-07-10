import wave

import numpy as np

from stenograf.audio import (
    SAMPLE_RATE,
    audio_channel_count,
    channels_look_independent,
    load_audio,
    load_audio_channels,
    to_float32,
    to_int16,
)


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


def test_to_int16_inverts_to_float32():
    ints = np.array([-32768, -1000, 0, 1000, 32767], dtype=np.int16)
    assert to_int16(ints) is ints  # int16 passes through
    roundtripped = to_int16(to_float32(ints))
    assert roundtripped.dtype == np.int16
    assert list(roundtripped) == list(ints)
    clipped = to_int16(np.array([-2.0, 2.0], dtype=np.float32))
    assert list(clipped) == [-32768, 32767]


def test_load_audio_channels_preserves_each_side(tmp_path):
    left = np.full(1000, 1000, dtype=np.int16)
    right = np.full(1000, 3000, dtype=np.int16)
    path = tmp_path / "stereo.wav"
    write_wav(path, np.column_stack([left, right]).ravel(), channels=2)
    channels = load_audio_channels(path)
    assert len(channels) == 2
    assert all(ch.dtype == np.float32 for ch in channels)
    assert np.allclose(channels[0], to_float32(left))
    assert np.allclose(channels[1], to_float32(right))


def test_load_audio_channels_mono_is_a_single_channel(tmp_path):
    samples = np.full(500, 1234, dtype=np.int16)
    path = tmp_path / "mono.wav"
    write_wav(path, samples)
    channels = load_audio_channels(path)
    assert len(channels) == 1
    assert np.allclose(channels[0], to_float32(samples))


def test_audio_channel_count_reads_the_wav_header(tmp_path):
    mono, stereo = tmp_path / "mono.wav", tmp_path / "stereo.wav"
    write_wav(mono, np.zeros(100, dtype=np.int16))
    write_wav(stereo, np.zeros(200, dtype=np.int16), channels=2)
    assert audio_channel_count(mono) == 1
    assert audio_channel_count(stereo) == 2


def _bursts(spans, seconds=6, amplitude=0.3):
    pcm = np.zeros(seconds * SAMPLE_RATE, dtype=np.float32)
    for start, end in spans:
        pcm[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = amplitude
    return pcm


def test_disjoint_voice_channels_read_as_independent():
    # Turn-taking on separate feeds (a tee, a call recording): envelopes decorrelate.
    left = _bursts([(0.0, 1.5), (3.0, 4.0)])
    right = _bursts([(1.5, 3.0), (4.0, 5.5)])
    independent, correlation = channels_look_independent(left, right)
    assert independent
    assert correlation is not None and correlation < 0.0


def test_stereo_image_reads_as_one_stream():
    # The same voices in both channels (panned/attenuated): envelopes track.
    signal = _bursts([(0.5, 2.0), (3.0, 5.0)])
    independent, correlation = channels_look_independent(signal, signal * 0.4)
    assert not independent
    assert correlation is not None and correlation > 0.9


def test_dead_channel_is_not_independent():
    # A constant channel has nothing to split; the mixdown is the status quo.
    signal = _bursts([(0.5, 2.0)])
    silent = np.zeros_like(signal)
    assert channels_look_independent(signal, silent) == (False, None)
    assert channels_look_independent(np.zeros(0, np.float32), silent) == (False, None)
