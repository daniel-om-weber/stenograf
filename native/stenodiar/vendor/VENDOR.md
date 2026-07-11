# Vendored: speakrs 0.5.0 (Apache-2.0)

Pristine crates.io source (`speakrs-0.5.0.crate`) plus two Linux/CPU
throughput patches, wired in via `[patch.crates-io]` in stenodiar's
Cargo.toml. Upstream pins pure-CPU diarization to ~1x realtime; with both
patches the same pipeline measured 7.8x realtime on a Ryzen AI 9 HX 370
(307 s three-speaker fixture: 215 s → 39.5 s, identical turns and speaker
count). Neither patch touches CoreML/CUDA behaviour.

1. `src/inference/embedding/session.rs` — embedding ORT sessions were built
   with one intra-op thread (upstream's parallelism lives in the CoreML/CUDA
   chunk workers, which don't exist for plain CPU). For
   `ExecutionMode::Cpu`, use `available_parallelism().min(8)` instead;
   measured 215 s → 76 s alone. The cap: past ~8 threads ORT burns >50 %
   more CPU for ~5 % wall-clock.

2. `src/models.rs` — the CPU download list fetched only the single-item
   ONNX models, silently disabling the multi-mask/batched code paths the
   crate already has (the HF repo carries the exports). Fetch them for CPU
   too; the pipeline then picks batched segmentation and multi-mask
   embedding (fbank + ResNet trunk once per chunk instead of per speaker
   mask); measured 215 s → 107 s alone, 39.5 s combined with patch 1.

Both are upstream-worthy fixes, not workarounds — offer them to
https://github.com/avencera/speakrs and delete this vendor copy once a
release contains them. To rebase onto a newer speakrs: re-extract the
crate, reapply the two patches (each is a single hunk), rebuild, and rerun
the throughput measurement in PLAN.md §5 Phase 5.
