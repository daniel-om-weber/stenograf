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
   **Done 2026-11-15:** `VoiceEmbedder` + `cluster_embeddings` in
   `diarization/embedding.py`; the sherpa class delegates to it, embed-only
   eval scripts construct it directly. Gate: on 4 min of Bmr021's mic channel,
   the loop's pair vectors, the three cluster voices, the channel voice and 20
   raw slice embeddings were exactly equal (`np.array_equal`) before and after,
   through both the diarizer and the standalone embedder.
2. **`diarization/nemotron.py`, not yet wired in.** Numpy 128-band log-mel
   front end (preemphasis 0.97, n_fft 512, win 400, hop 160; parity fixture
   generated once from the reference processor so tests need no torch), the
   chunk loop and speaker cache ported from `eval/nemotron_arm.py` (its three
   traps are in its docstring), the stated count applied afterwards as a cap
   (`eval/heldout.py`'s `count_capped` rule), embeddings from step 1. New asset:
   the fp16 export + `config.json` from `onnx-community/Nemotron-3-Diarization-ONNX`,
   pinned by revision and sha256. Gate: library turns identical to
   `nemotron_arm.py`'s on the held-out set.
   **Done 2026-11-15**, gate as amended by Daniel the same day: the turns are
   identical only given the reference's features (36/36), because the numpy
   front end is not bit-equal to torch's and the speaker cache amplifies the
   difference (1/36 end to end); the held-out scores are unchanged.
   Numbers: `eval/README.md`, "The library port".
3. **The switch.** `build_diarizer()` returns Nemotron; `pipeline._attribute`
   and `cli/profiles.py` pass the stated count straight through as the cap
   and stop diarizing at k+1 / folding. Decide `collapse_single_voice` by
   measurement on Nemotron's single-speaker channels (it over-counts 2 of 12
   held-out mic channels): keep it only if it fixes those without merging a
   real pair. `steno setup` / `doctor` fetch and check the new model; settings
   and help text stop calling an unstated count "less accurate".
   **Done 2026-11-15.** The collapse is dropped: on Nemotron's uncapped turns
   it fixed one count and no attribution, and merged one real two-voice pair
   (81.7 → 50.2 %); the "2 of 12" over-counts were a scoring quirk (fixed in `heldout.py
   score`) — all five
   single-speaker extras are crosstalk (`eval/README.md`, "No single-voice
   collapse behind Nemotron"). `steno setup` fetches and `doctor` checks the
   Nemotron files in place of the pyannote segmentation model; the doctor's
   stenodiar check went now, since it would have reported an unused helper.
   End to end, `steno transcribe` on 3 min of ES2004b.duo found the same two
   speakers with `--speakers 2` and with no count.
4. **Enrollment guard.** `steno profiles assign` refuses (clear message,
   `--force` to override) an entry whose cosine to the profile's existing
   entries is below the naming threshold. The held-out phantom (0.046 against
   FIO084's real cluster) must be refused and genuine second entries pass;
   check the threshold on the harness's session-a clusters before shipping.
   **Done 2026-11-15**, at the naming threshold. The phantom scores 0.053 and
   every full-length genuine entry passes (lowest 0.72); a weak genuine
   sample (2–3 s) is refused about half the time, and measured right to be:
   admitted, entries below 0.56 cost that person their later short turns
   whoever's voice they are (`eval/README.md`, "The assign guard"). Not
   covered: `steno profiles enroll --reinforce`, and a profile whose *first*
   entry is the bad one (nothing to compare against). Measured next the same
   day: a median, or the mean without the lowest score, keeps a 4-entry
   profile with one bad entry as good as clean on 3 s turns, where the mean
   loses a third to two thirds (`eval/README.md`, "A robust profile score").
   Daniel chose the median; the store matches by it since that day.
5. **The old diarizers go.** `diarization/loop.py` (owned loop), the sherpa
   diarizer class, `diarization/speakrs.py`, `fold_excess_clusters`,
   `FOLD_PAIR_SIMILARITY`, `collapse_single_voice` and `COLLAPSE_SIMILARITY`
   (production stopped calling all four in step 3; only eval arms still do), the pyannote segmentation asset, `native/stenodiar/`
   with the vendored speakrs (its doctor check already went in step 3), CI job, pre-commit hook mention
   and `hatch_build.py` branches; the manylinux_2_39 wheel goes and
   manylinux_2_28 becomes the only Linux wheel. `PLAN-DIARIZATION-SPEED.md`'s
   open item (stenodiar's estimate speed) closes with it. Eval scripts whose
   arms run the owned loop or stenodiar lose those arms — their numbers stay in
   `eval/README.md` and git history; scripts that only embed switch to step 1's
   module.
   **Done 2026-11-15.** Also gone: `sherpa.py`'s base class (nothing
   subclassed it any more) and the tar-archive extraction in `assets.py`,
   whose only user was the segmentation asset. Platform wheels now carry
   stenocap alone: macOS arm64, win_amd64, and one manylinux_2_28 Linux wheel,
   smoke-tested on a current runner and on Ubuntu 22.04. Thirteen eval scripts
   that measured only the deleted pieces were removed (listed in
   `eval/README.md`'s corpus-harness section); `diarize.py`, `solo_arms.py`,
   `split_recovery.py`, `heldout.py`, `owncalls.py` and `naming_gate.py`
   lost their old arms and run Nemotron where they run a diarizer at all.
   `PLAN-DIARIZATION-SPEED.md` closed and was deleted; its two facts that
   outlive it are in `PLAN.md`. Not verified: the release workflow, which
   only a tag or a `workflow_dispatch` dry run exercises.
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
