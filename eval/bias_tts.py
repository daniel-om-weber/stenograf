"""Tier 5 — synthetic fixtures for biasing. A diagnostic, explicitly NOT a metric.

TTS is the only way to get audio containing *Daniel's own* vocabulary with perfect
ground truth. It is also a liar. Synthetic speech is clean, prosodically flat, and —
the killer — a TTS engine mispronounces technical terms, inventing errors that never
occur in real speech, which biasing then "fixes". **No claim about how much biasing
helps may come from this file.** Measured, not feared: same script, same ASR, same
glossary, two TTS engines → a 25-point spread in the *unbiased* baseline, with every
apparent gain being biasing repairing an error the TTS itself manufactured. (Zhao et
al., Interspeech 2019, found the same inflation on real vs. synthetic contact names —
80.3 % vs 52.5 %, with the sign flipping in one condition.)

What it legitimately answers, and the only question asked of it: *can the mechanism
reach this class of error at all?* One fixture per class we have actually seen the
decoder fail on — compound-tail terms, glued words, casing, multi-token names — plus
false-insertion probes, where the term is **not in the audio** and recovering it is
the failure.

**Both pronunciations, on purpose.** Real German speakers say tech jargon anywhere
from native English to fully Germanized, so "correct" is a distribution, not a point.
Each sentence is therefore built twice — once with the term under German
pronunciation rules, once under English — and the term must be recovered under
*both*. A term recovered under only one is a coin flip, not a capability. This turns
the engine's biggest confound into a deliberate robustness axis.

The two pronunciations are spliced at the *phoneme* level: every word is phonemized
separately (identically in both conditions, so the only difference is the term), and
the term's phonemes come from espeak's German or English rules as the condition
demands. Piper bundles its own espeak-ng, so this needs no system espeak install and
no compiled ``de_extra`` dictionary — and unlike a dictionary it is exact about what
changed, which for a controlled comparison is the whole point.

Acoustic augmentation is deliberately skipped: Parakeet was almost certainly trained
with MUSAN/RIR-style augmentation, so augmenting with the same distributions barely
moves its WER while doing nothing about the prosody and disfluency gap — it buys
false confidence.

**Piper is GPL-3.0** — eval group only. It must never reach the shipped wheel.
Voices are thorsten/kerstin (CC0). Rejected: Chatterbox (embeds inaudible
watermarking into the audio — a hazard in an ASR eval set), XTTS-v2 and
F5-TTS-German (non-commercial), Kokoro (no German).

Usage:
    uv run --group eval eval/bias_tts.py            # generate fixtures + manifest
    uv run --group eval eval/bias.py --tier tts     # then score them
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bias_data import BIAS_DIR  # noqa: E402

TTS_DIR = BIAS_DIR / "tts"
MANIFEST = TTS_DIR / "manifest.json"

VOICES = ("de_DE-thorsten-medium",)
"""Only thorsten by default. ``de_DE-kerstin-low`` ships a 130-phoneme id map with
no combining cedilla, i.e. it cannot render the German ich-laut — Piper drops the
phoneme with a log line and synthesizes "Ich", "Küche" and "nächste" subtly wrong.
A fixture set whose *German* is mangled cannot be evidence about anything, and a
corrupted variable in a diagnostic is worse than one fewer voice. :func:`_coverage`
now refuses any voice that cannot say a fixture, so this fails loudly rather than
quietly. Pass ``--voice`` to add others once they pass that check."""

GERMAN = "de"
ENGLISH = "en-us"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Fixture:
    """One sentence, one error class, and the terms the decoder should recover."""

    case: str
    klass: str
    text: str
    terms: tuple[str, ...]
    foreign: tuple[str, ...] = ()
    """Words rendered with English pronunciation rules in the ``en`` condition.
    Usually the term itself, or the English half of a German compound."""
    absent: bool = False
    """The terms are NOT in the audio: a false-insertion probe, where recovering
    them is the failure and silence is the pass."""


FIXTURES: tuple[Fixture, ...] = (
    # The failure that motivated the compound-tail tokenization: the decoder wrote
    # "Grafana-Dashboot", and a word-start-only tree could not touch it, because
    # inside a compound the term carries no word-boundary marker at all.
    Fixture(
        case="compound-tail",
        klass="compound",
        text="Ich habe das Grafana-Dashboard heute morgen neu gebaut.",
        terms=("Grafana-Dashboard",),
        foreign=("Grafana-Dashboard",),
    ),
    # The decoder glues two words into one ("Prometheusalord"). Boosting "Alert"
    # alone provably cannot fix this — a completed phrase returns the tree to its
    # root and "Alert" never starts a word there — so the compound must be listed.
    Fixture(
        case="glued-words",
        klass="glued",
        text="Der Prometheus-Alert ist heute nacht zweimal ausgelöst worden.",
        terms=("Prometheus-Alert",),
        foreign=("Prometheus-Alert",),
    ),
    # Casing is load-bearing: the tree matches token ids, and "kubernetes" and
    # "Kubernetes" are different token paths.
    Fixture(
        case="casing",
        klass="casing",
        text="Wir migrieren den Cluster nächste Woche auf Kubernetes.",
        terms=("Kubernetes",),
        foreign=("Kubernetes",),
    ),
    # A name is usually misheard one part at a time, so parts are boosted too — but
    # the full name pulls hardest, its reward growing with phrase depth.
    Fixture(
        case="multi-token-name",
        klass="name",
        text="Ada Lovelace hat den ersten Algorithmus geschrieben.",
        terms=("Ada Lovelace",),
        foreign=("Ada", "Lovelace"),
    ),
    Fixture(
        case="rare-product-name",
        klass="compound",
        text="Der Kafka-Consumer hängt seit dem Deployment am Mittwoch.",
        terms=("Kafka-Consumer",),
        foreign=("Kafka-Consumer",),
    ),
    # False-insertion probes: the audio is ordinary German with none of the boosted
    # terms in it. Any of them appearing in the transcript is the decoder being
    # talked into a word that was never spoken — the failure mode NVIDIA warns
    # about, and the one that ruins a meeting transcript.
    Fixture(
        case="absent-terms",
        klass="false-insertion",
        text="Wir treffen uns am Montag um zehn Uhr im großen Besprechungsraum.",
        terms=("Grafana-Dashboard", "Kubernetes", "Prometheus-Alert", "Ada Lovelace"),
        absent=True,
    ),
    Fixture(
        case="absent-near-miss",
        klass="false-insertion",
        # "Kaffee" is acoustically close to the boosted "Kafka": if boosting can
        # overwrite a correctly-heard everyday word, it happens here first.
        text="Ich hole mir noch schnell einen Kaffee aus der Küche.",
        terms=("Kafka-Consumer", "Kafka"),
        absent=True,
    ),
)


def _phonemes(phonemizer, fixture: Fixture, pronunciation: str) -> list[str]:
    """The sentence's phonemes, with the term under German or English rules.

    Every word is phonemized on its own in *both* conditions, so the two differ only
    in the term's phonemes and nothing else — the comparison is controlled by
    construction rather than by hope. The cost is a little sentence-level
    co-articulation, which no claim here depends on.
    """
    foreign = {word.strip(string.punctuation) for word in fixture.foreign}
    phonemes: list[str] = []
    for word in fixture.text.split():
        bare = word.strip(string.punctuation)
        language = ENGLISH if (pronunciation == "en" and bare in foreign) else GERMAN
        for sentence in phonemizer.phonemize(language, word):
            phonemes.extend(sentence)
        phonemes.append(" ")
    return phonemes


def _coverage(voice, phonemizer) -> dict[str, list[str]]:
    """Phonemes each fixture needs that this voice's id map does not have.

    Piper drops an unmapped phoneme with nothing but a log line, so the WAV is
    quietly wrong — the worst possible failure for a fixture whose entire job is to
    be ground truth. Checked up front, once, so a bad voice cannot reach the report.
    """
    id_map = voice.config.phoneme_id_map
    missing: dict[str, list[str]] = {}
    for fixture in FIXTURES:
        for pronunciation in ("de", "en") if fixture.foreign else ("de",):
            absent = {
                phoneme
                for phoneme in _phonemes(phonemizer, fixture, pronunciation)
                if phoneme not in id_map
            }
            if absent:
                missing[f"{fixture.case}/{pronunciation}"] = sorted(absent)
    return missing


def _synthesis_config():
    """Deterministic synthesis — the fixtures must be a *fixed* set.

    Piper's VITS decoder samples noise (and a stochastic duration predictor), so the
    same sentence synthesized twice is different audio. Caught the hard way:
    regenerating the fixtures moved "Kafka-Konsumer" to "Kafka Konsumer" and flipped
    a verdict. A benchmark whose stimulus drifts under it is not a benchmark, and it
    would quietly poison every comparison, so the noise is zeroed. The speech comes
    out a little flatter, which no claim here depends on — these are reachability
    probes, not naturalness ones.
    """
    from piper import SynthesisConfig

    return SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)


def _synthesize(voice, phonemizer, fixture: Fixture, pronunciation: str) -> np.ndarray:
    """The sentence's audio, with the term under German or English phoneme rules."""
    phonemes = _phonemes(phonemizer, fixture, pronunciation)
    ids = voice.phonemes_to_ids(phonemes)
    audio = voice.phoneme_ids_to_audio(ids, _synthesis_config())
    return np.asarray(audio, dtype=np.float32)


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write mono int16 at 16 kHz — the one wire format the whole harness uses."""
    if rate != SAMPLE_RATE:
        # Linear resample. The fixtures are a reachability probe, not an acoustic
        # benchmark, and Parakeet's own front end is far less fussy than the
        # difference between this and a windowed-sinc kernel.
        target = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    samples = np.clip(audio, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((samples * 32767).astype(np.int16).tobytes())


def generate(voices: tuple[str, ...] = VOICES) -> list[dict]:
    from piper import PiperVoice
    from piper.download_voices import download_voice
    from piper.phonemize_espeak import EspeakPhonemizer

    voice_dir = TTS_DIR / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    phonemizer = EspeakPhonemizer()

    manifest: list[dict] = []
    for name in voices:
        model = voice_dir / f"{name}.onnx"
        if not model.exists():
            print(f"↓ {name}", flush=True)
            download_voice(name, voice_dir)
        voice = PiperVoice.load(model)
        rate = voice.config.sample_rate

        gaps = _coverage(voice, phonemizer)
        if gaps:
            raise SystemExit(
                f"{name} cannot render these fixtures — its phoneme id map is missing "
                f"{sorted({p for ps in gaps.values() for p in ps})}, and Piper would "
                f"drop them silently: {sorted(gaps)}"
            )

        for fixture in FIXTURES:
            # A probe with no foreign words has one pronunciation, not two: there is
            # no term in the audio to say either way.
            conditions = ("de", "en") if fixture.foreign else ("de",)
            for pronunciation in conditions:
                wav = TTS_DIR / "wav" / f"{fixture.case}.{pronunciation}.{name}.wav"
                _write_wav(wav, _synthesize(voice, phonemizer, fixture, pronunciation), rate)
                manifest.append(
                    {
                        "wav": str(wav),
                        "case": fixture.case,
                        "class": fixture.klass,
                        "text": fixture.text,
                        "terms": list(fixture.terms),
                        "pronunciation": pronunciation,
                        "voice": name,
                        "absent": fixture.absent,
                    }
                )
                print(f"  {wav.name}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", nargs="+", default=list(VOICES))
    args = parser.parse_args()

    manifest = generate(tuple(args.voice))
    print(f"\nwrote {len(manifest)} fixtures → {MANIFEST}")
    print("Reachability probes only — no quality claim may be made from synthetic audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
