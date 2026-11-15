# Nemotron 3 as the one diarizer — migration plan (2026-11-15)

Evidence: `eval/README.md`, "The estimate path, and Nemotron 3 Diarization as
the one diarizer" and the three sections under it (held-out AMI on macOS,
re-ID, two speakers off AMI). Nemotron beats the owned loop *given the true
count* everywhere it was measured and stenodiar everywhere; re-ID holds at
0.56 with a stated count applied afterwards.

Daniel's decisions (2026-11-15): **the owned loop is deleted in this migration,
no fallback release**; profiles are protected by a **guard on assign**, the
speaker count stays optional everywhere; **CPU only**, WebGPU later.

Settled by measurement before the plan (2026-11-15):

- **Licence:** OpenMDW-1.1 — use, modify, commercial use without field-of-use
  limits, nothing on outputs. Redistributing the weights requires shipping the
  licence and origin notices; stenograf downloads them at first run and never
  redistributes, so it owes nothing beyond naming the licence in its docs.
- **Runtime:** the onnxruntime the wheel already locks (1.27.0) runs the export
  (fp32 7.4e-6 against the PyTorch reference, fp16 1.4e-3, no flips).
- **Precision:** fp16 (199 MB, half of fp32's 398 MB) — held-out attribution
  within 0.1 pt of fp32, identical on the two-remote-voices set, 17 s per
  audio-hour on M4 Max CPU (fp32: 15).

## Steps — each one commit, CI triple green, behaviour change only where named

1. **The voice embedder stands alone.** ERes2Net embedding, the shared worker
   pool, `cluster_embeddings` and the overlap helpers move out of
   `diarization/sherpa.py` into `diarization/embedding.py`, so re-ID, enrollment
   and the eval harness no longer reach the embedder through a diarizer class
   (`OwnDiarizer().embed` is used across `eval/`). Pure move — embeddings
   bit-identical.
2. **`diarization/nemotron.py`, not yet wired in.** Numpy 128-band log-mel
   front end (preemphasis 0.97, n_fft 512, win 400, hop 160; parity fixture
   generated once from the reference processor so tests need no torch), the
   chunk loop and speaker cache ported from `eval/nemotron_arm.py` (its three
   traps are in its docstring), the stated count applied afterwards as a cap
   (`eval/heldout.py`'s `count_capped` rule), embeddings from step 1. New asset:
   the fp16 export + `config.json` from `onnx-community/Nemotron-3-Diarization-ONNX`,
   pinned by revision and sha256. Gate: library turns identical to
   `nemotron_arm.py`'s on the held-out set.
3. **The switch.** `build_diarizer()` returns Nemotron; `pipeline._attribute`
   and `cli/profiles.py` pass the stated count straight through as the cap
   and stop diarizing at k+1 / folding. Decide `collapse_single_voice` by
   measurement on Nemotron's single-speaker channels (it over-counts 2 of 12
   held-out mic channels): keep it only if it fixes those without merging a
   real pair. `steno setup` / `doctor` fetch and check the new model; settings
   and help text stop calling an unstated count "less accurate".
4. **Enrollment guard.** `steno profiles assign` refuses (clear message,
   `--force` to override) an entry whose cosine to the profile's existing
   entries is below the naming threshold. The held-out phantom (0.046 against
   FIO084's real cluster) must be refused and genuine second entries pass;
   check the threshold on the harness's session-a clusters before shipping.
5. **The old diarizers go.** `diarization/loop.py` (owned loop), the sherpa
   diarizer class, `diarization/speakrs.py`, `fold_excess_clusters` and
   `FOLD_PAIR_SIMILARITY`, the pyannote segmentation asset, `native/stenodiar/`
   with the vendored speakrs, its doctor check, CI job, pre-commit hook mention
   and `hatch_build.py` branches; the manylinux_2_39 wheel goes and
   manylinux_2_28 becomes the only Linux wheel. `PLAN-DIARIZATION-SPEED.md`'s
   open item (stenodiar's estimate speed) closes with it. Eval scripts whose
   arms run the owned loop or stenodiar lose those arms — their numbers stay in
   `eval/README.md` and git history; scripts that only embed switch to step 1's
   module.
6. **Docs and observation.** README, `CLAUDE.md`, `native/README.md`, the
   `verify` skill, `PLAN.md`'s Nemotron section (the gates it lists close; the
   shared-room-mic check becomes "watch the first real hybrid meetings"), the
   licence named. Then a real finalize replayed through the `verify` skill with
   peak memory measured, and CI green on all three platforms. Release = version
   bump + tag, Daniel's call.

## Out of scope for now

- **WebGPU.** CPU is already 17 s per audio-hour; the plugin adds a second
  onnxruntime copy on macOS and segfaults on a second session. Revisit as an
  optional extra once the CPU path has shipped.
- **Streaming diarization in live captions.** Nemotron has an online mode;
  live captions carry no speaker labels today.
