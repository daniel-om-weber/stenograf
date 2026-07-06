# Phase 0 — model evaluation harness

Goal: decide the default finalize-pass ASR model on *real* meeting audio, not
read-speech leaderboards. Candidates:

| Model | Runtime | Role |
|---|---|---|
| Canary-1B-v2 | MLX | expected finalize default |
| Voxtral Small 24B (4-bit) | mlx-voxtral | max-accuracy challenger |
| Parakeet-TDT-0.6B-v3 | parakeet-mlx | live-pass model (baseline here) |
| Whisper large-v3 | mlx-whisper | mature fallback |

Planned procedure:

1. Extract audio from meeting recordings in `../examples/` (local only, never
   committed) with ffmpeg → mono 16 kHz WAV segments.
2. Hand-correct reference transcripts for ~10 minutes per language
   (German + English), including one in-room far-field sample.
3. Run every candidate over the segments; report WER (via `jiwer`),
   realtime factor, and peak memory on the M4 Max.
4. Side quests: verify the Canary MLX port's maturity (biggest stack unknown)
   and sanity-check speakrs diarization quality on the same audio.

The harness lives here as scripts with their own dependencies
(`uv run --group eval ...`), separate from the shipped package.
