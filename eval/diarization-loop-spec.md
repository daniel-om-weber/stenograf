# Owning the diarization loop — reference spec

Citations: sherpa-onnx at commit `f737aebe7ca86b1fb029b2b62ce655080994c47d` (master, 2026-08-06),
clone in scratchpad. Line numbers are from that commit. **Installed package is
`sherpa_onnx` 1.13.4** — one behavioural difference from master is flagged in §1.1.
Every number below is `measured` (read from source / run) unless labelled otherwise.

---

## 1. sherpa's pyannote offline diarization loop

Everything lives in one header:
`sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h`
(class `OfflineSpeakerDiarizationPyannoteImpl`). Top-level order in `Process()`
(lines 96–221):

```
RunSpeakerSegmentationModel  -> ToMultiLabel (powerset argmax)
  -> [labels.size()==1 ? HandleOneChunkSpecialCase and return]
  -> ComputeSpeakersPerFrame          (how many speakers each global frame has)
  -> GetChunkSpeakerSampleIndexes     (ExcludeOverlap, >=10-frame gate, sample spans)
  -> ComputeEmbeddings                (one embedding per (chunk, local speaker))
  -> FastClustering::Cluster          (complete linkage on cosine distance)
  -> ConvertChunkSpeakerToCluster / ReLabel
  -> ComputeSpeakerCount              (vote accumulation on the global frame grid)
  -> FinalizeLabels                   (top-k per frame, k = speakers_per_frame)
  -> ComputeResult                    (frames -> times, MergeSegments, min_duration_on)
```

### 1.1 Windowing

| Quantity | Value | Source |
|---|---|---|
| `window_size` | **160000 samples = 10.0 s** | ONNX metadata key `window_size`; written by `scripts/pyannote/segmentation/export-onnx.py:97` as `int(model.specifications.duration)*16000` |
| `window_shift` | **16000 samples = 1.0 s** | *not* in metadata — computed. master: `offline-speaker-segmentation-pyannote-model.cc:96-113`, `window_shift = window_shift_ratio * window_size`, default `window_shift_ratio = 0.1f` (`offline-speaker-segmentation-pyannote-model-config.h:15`). **v1.13.4 (installed) hardcodes it**: `meta_data_.window_shift = static_cast<int32_t>(0.1 * meta_data_.window_size)` — no config knob. So parity target = 16000 either way. |
| `receptive_field_shift` | **270 samples = 0.016875 s** | ONNX metadata; export script line 99 |
| `receptive_field_size` | **991 samples = 0.0619375 s** | ONNX metadata; export script line 98 |
| frames per 10 s chunk | **589** | export script line 70 asserts `[1, 589, 7]` |
| `sample_rate` | 16000 | ONNX metadata |

Note 589 × 270 = 159030 ≠ 160000. sherpa uses **two different frame→sample scales**
depending on the call site — see the gotcha in §1.5.

**Chunking** (`RunSpeakerSegmentationModel`, lines 278–337):

```cpp
if (n <= window_size) {                       // 301
  std::vector<float> buf(window_size);        // zero-initialised
  std::copy(audio, audio + n, buf.data());    // right-pad with zeros
  return { ProcessChunk(buf.data()) };        // ONE chunk, always
}
int32_t num_chunks   = (n - window_size) / window_shift + 1;   // 314
bool has_last_chunk  = ((n - window_size) % window_shift) > 0; // 315
for (i = 0; i != num_chunks; ++i, p += window_shift) ...       // 321
if (has_last_chunk) {                                          // 327
  std::vector<float> buf(window_size);        // zero-init again
  std::copy(p, audio + n, buf.data());        // remainder, right-padded
}
```

So: **the last partial window is zero-padded to a full 10 s and run**, and its
frames are later truncated (§1.7). `p` after the loop points at
`audio + num_chunks*window_shift`, so the tail chunk is a full-length window
starting at the normal stride position, not a shortened one.

`ProcessChunk` (339–358) feeds a fixed `{1, 1, window_size}` float tensor.

### 1.2 Powerset mapping (7 classes, exact order)

`InitPowersetMapping()`, lines 243–276. `num_classes=7`, `num_speakers=3`,
`powerset_max_classes=2` (all from ONNX metadata). The loop emits, in order:

| idx | speaker set | row |
|---|---|---|
| 0 | {} | 0 0 0 |
| 1 | {s0} | 1 0 0 |
| 2 | {s1} | 0 1 0 |
| 3 | {s2} | 0 0 1 |
| 4 | {s0,s1} | 1 1 0 |
| 5 | {s0,s2} | 1 0 1 |
| 6 | {s1,s2} | 0 1 1 |

Identical to `pyannote.audio.utils.powerset.Powerset.build_mapping` (by cardinality,
then lexicographic) — sherpa's own comment at line 242 cites it. `powerset_max_classes > 2`
is a hard exit (line 273).

`ToMultiLabel` (360–372) is a plain **argmax over the 7 logits per frame**, then a
mapping-row lookup. No thresholding, no softmax. (The ONNX output is already
log-softmax — pyannote's `PyanNet.forward` applies `nn.LogSoftmax(dim=-1)` for
`MONO_LABEL_CLASSIFICATION`; argmax is invariant to it, so raw logits would be
equivalent for the hard decode.)

### 1.3 Global frame grid

`ComputeSpeakersPerFrame` (376–406) and `ComputeSpeakerCount` (641–677) share it:

```cpp
num_frames = (window_size + (num_chunks - 1) * window_shift) / receptive_field_shift + 1;
// chunk i's 589 frames are pasted starting at:
int32_t start = static_cast<float>(i) * window_shift / receptive_field_shift + 0.5; // = round(i * 59.259...)
```

`num_chunks` here is `labels.size()` (i.e. **including** the padded tail chunk).
Integer division, and `+0.5` then narrowing to int32 == round-half-up.

`ComputeSpeakersPerFrame` averages the per-frame speaker *count* across all chunks
covering that frame and rounds:
`round(sum(active speakers) / number of covering chunks)` (line 405,
`(count/(weight+1e-12) + 0.5).cast<int32_t>()`). This is the k that
`FinalizeLabels` will use. It is computed on the **overlap-inclusive** labels.

### 1.4 What gets embedded

`GetChunkSpeakerSampleIndexes` (412–484). One embedding per **(chunk, local speaker
index 0..2)** pair — but measured on the ES2003c.loop reference the mean is
**1.05–1.08 pairs per chunk** (2026-08-09), and steps 1–3 below crop each
pair to its sole-speaker active runs, so our redundancy is ~10× (window/shift
overlap),
not the ~21× §1.5 of the research record ascribes to pyannote (which embeds
the full window per speaker slot).

1. `ExcludeOverlap` first (487–508): any frame whose row-sum ≥ 2 is zeroed
   entirely. So overlap frames are decoded into the *output turns* but never
   contribute to *clustering embeddings*.
2. Minimum-duration gate (436–439):
   ```cpp
   auto d = tmp.row(speaker_index);
   if (d.sum() < 10) continue;   // "skip segments less than 10 frames"
   ```
   **10 non-overlapped active frames per (chunk, speaker)**, summed, not contiguous.
   ≈ 0.169 s. This is the only duration gate before embedding.
3. Contiguous runs of active frames become sample spans (447–475):
   ```cpp
   start_samples = (float)start_index / num_frames * window_size + sample_offset;
   end_samples   = (float)k           / num_frames * window_size + sample_offset;
   // sample_offset = chunk_index * window_shift
   ```
   A trailing active run ends at `(num_frames - 1)/num_frames * window_size`
   (line 472) — off-by-one vs the closed case, harmless.
4. `ComputeEmbeddings` (517–572) feeds **all spans of that (chunk, speaker) into one
   extractor stream** (`AcceptWaveform` per span, then `InputFinished`) — i.e. the
   spans are concatenated, one embedding out. Spans are clipped to `n` (line 534).
   NaN embeddings are dropped and the (chunk, speaker) list is filtered to match
   (`valid_indexes`, lines 162–185).

### 1.5 GOTCHA: two frame→sample scales

- Embedding spans use `window_size / num_frames = 160000/589 = 271.646` samples per frame
  (line 457).
- Output turn times use `receptive_field_shift = 270` samples per frame plus a
  `0.5 * receptive_field_size` offset (§1.7).

The embedding path therefore drifts up to 589×1.646 ≈ 970 samples (≈ 61 ms) at the end
of a chunk relative to the output-time path. Inferred consequence: a reimplementation
that uses 270 everywhere will not be bit-parity with sherpa on embeddings. Decide
deliberately which one to keep.

### 1.6 FastClustering

`sherpa-onnx/csrc/fast-clustering.cc:18-68`.

```cpp
m.rowwise().normalize();                       // 31  L2-normalise in place
double cosine_similarity   = v.dot(m.row(j));  // 39
double consine_dissimilarity = 1 - cosine_similarity;  // 40, clamped >= 0 at 42-44
fastclustercpp::hclust_fast(num_rows, distance.data(),
                            fastclustercpp::HCLUST_METHOD_COMPLETE,  // 55
                            merge.data(), height.data());
if (config_.num_clusters > 0)
  cutree_k(num_rows, merge.data(), config_.num_clusters, labels.data());   // 60
else
  cutree_cdist(num_rows, merge.data(), height.data(), config_.threshold, labels.data()); // 63
```

- **Linkage: COMPLETE** (`HCLUST_METHOD_COMPLETE = 1`, `fastcluster.h:69`).
- **Metric: cosine DISTANCE `1 - cos`, clamped to ≥ 0.** So sherpa's `threshold` is a
  cosine *distance*: `threshold = 0.5` ⇔ merge while cosine similarity ≥ 0.5.
  (Contrast pyannote 3.1, whose 0.7046 is a *Euclidean* distance on unit vectors
  ⇔ cosine similarity 0.752 — §4.)
- Condensed upper-triangle distance array, `n(n-1)/2` (`fastcluster.h` doc).
- Cut semantics (`fastcluster.cpp:95-105`, BSD-style license):
  ```cpp
  for (k=0; k<(n-1); k++) if (height[k] >= cdist) break;
  cutree_k(n, merge, n-k, labels);
  ```
  i.e. **first merge whose height ≥ threshold stops the tree**; `>=`, not `>`.
- `num_clusters > 0` bypasses the threshold entirely (`cutree_k`). `n<=1` short-circuits
  (lines 20–26). Labels are 0-based, arbitrary order.
- **No `min_cluster_size` equivalent.** pyannote's small-cluster reassignment
  (§4) has no counterpart here — this is a real behavioural gap.

`FastClusteringConfig` defaults (`fast-clustering-config.h:19,25`):
`num_clusters = -1`, `threshold = 0.5f`. Confirmed live on the installed 1.13.4:
`FastClusteringConfig(num_clusters=-1, threshold=0.5)`.

### 1.7 Stitching, frames → times, post-processing

`ReLabel` (593–639) rewrites each chunk's (frames × 3 local speakers) matrix into
(frames × num_clusters), dropping local speakers with no cluster assignment
(dropped by the 10-frame gate or NaN).

`ComputeSpeakerCount` (641–677) accumulates those onto the global grid (same `start`
formula as §1.3), then **truncates the tail** when `has_last_chunk`:

```cpp
bool has_last_chunk = ((num_samples - window_size) % window_shift) > 0;  // 666
int32_t last_frame = num_samples / receptive_field_shift;                // 672
return count(Eigen::seqN(0, last_frame + 1), all);                       // 676
```

`FinalizeLabels` (679–702): per global frame, take the **top-k clusters by vote count**,
k = `speakers_per_frame[i]` from §1.3; k == 0 → silence. `TopkIndex`
(`sherpa-onnx/csrc/math.h:112-133`) is a `partial_sort` on descending value; ties
resolve by index order. This is where overlap re-enters the output: k can be 2.

`ComputeResult` (704–767) — the time mapping:

```cpp
float scale        = receptive_field_shift / sample_rate;        // 270/16000 = 0.016875
float scale_offset = 0.5 * receptive_field_size / sample_rate;   // 991/32000 = 0.03096875
start_time = start_index * scale + scale_offset;
end_time   = frame_index * scale + scale_offset;
```

**MEASURED / falsification test passed**: ran the installed 1.13.4 on 120 s of
`eval/audio/de-inroom.wav` (26 segments); all 52 endpoints satisfy
`(t - 0.03096875) / 0.016875 ∈ ℤ` to within 1e-3 frames. Chunk arithmetic also
confirmed (n = 1 920 000 → `num_chunks` 111, `has_last_chunk` false,
`num_frames` 7112). A reimplementation that gets this grid wrong is detectable
in one assertion.

Then, per speaker:

1. `MergeSegments` (791–808) — repeat-until-fixpoint pairwise merge using
   `OfflineSpeakerDiarizationSegment::Merge`
   (`offline-speaker-diarization-result.cc:33-51`):
   ```cpp
   if (end_ < other.start_ && end_ + gap >= other.start_)  return {start_, other.end_};
   ```
   **Gap merged when `0 < gap_length <= min_duration_off`.** Touching/overlapping
   segments of the same speaker are *not* merged by this (strict `<`) — they cannot
   occur anyway, since runs come from a per-speaker frame scan. O(n²) but n is small.
2. Drop rule (line 760): `if (seg.Duration() > config_.min_duration_on)` — **strict >**,
   applied *after* merging.

Defaults (`offline-speaker-diarization.h:25,30`, confirmed live on 1.13.4):
`min_duration_on = 0.3 s`, `min_duration_off = 0.5 s`.

Result ordering: `SortByStartTime` sorts by (start, speaker)
(`offline-speaker-diarization-result.cc:88-97`).

### 1.8 Single-chunk special case

`labels.size() == 1` (lines 120–133) — happens exactly when `n <= 160000` (10 s), since
`160000 < n < 176000` produces `num_chunks=1 + has_last_chunk=true` = 2 matrices.
In that path there is **no clustering and no embedding at all**: the three local
powerset speakers become the three output speakers directly.
`HandleOneChunkSpecialCase` (769–789) truncates to `num_samples / receptive_field_shift`
frames when the input was padded, then calls `ComputeResult`. Reimplementations must
keep this or short clips will diverge.

### 1.9 Python wrapper

`sherpa_onnx.OfflineSpeakerDiarization` is a thin pybind of the above; `set_config`
only replaces `config.clustering` (`offline-speaker-diarization.h:69-71`, impl at
`offline-speaker-diarization-pyannote-impl.h:87-94`) — this is what
`src/stenograf/diarization/sherpa.py:81` relies on to avoid reloading ONNX sessions.
The result object carries **no embeddings** (confirmed in-repo).

---

## 2. pyannote segmentation-3.0 ONNX I/O

All MEASURED on `~/Library/Caches/stenograf/pyannote-segmentation-3-0.onnx` by the
research subagent, cross-checked against `scripts/pyannote/segmentation/export-onnx.py`
(read directly, lines 61–133).

- **Input**: name `x`, shape `['N', 1, 'T']`, dtype `tensor(float)`. Raw waveform,
  16 kHz mono, **not** normalised by the model. Batch and length both dynamic
  (`dynamic_axes={"x": {0:"N", 2:"T"}, "y": {0:"N", 1:"T"}}`, export script 89–92).
  Verified working at T=80000 → 293 frames, T=160016 → 589 frames. sherpa always
  feeds exactly `window_size`.
- **Output**: name `y`, shape `['N', <frames>, 7]`. Frames symbolic:
  `floor(floor(floor(floor(T/10 - 251/10)/3 - 2/3)/3)/3 - 8/3) + 1`.
  **T = 160000 → 589.** (The "587" figure is UNVERIFIED and has no traceable source;
  inverting the formula, 587 would mean 9.951–9.968 s of input.)
- **Values are log-probabilities (log-softmax over the 7 classes)**, not raw logits.
  `PyanNet.forward` returns `self.activation(...)` and `Model.default_activation()`
  returns `nn.LogSoftmax(dim=-1)` for `Problem.MONO_LABEL_CLASSIFICATION` (the export
  script asserts that problem type, line 49). MEASURED: all outputs negative,
  `exp(y).sum(-1) ∈ [0.9999997, 1.0000002]`. Argmax is unaffected; anything
  probabilistic (soft powerset → multilabel) must `exp()` first.
- **Frame geometry**: step 270 samples = 0.016875 s; receptive field 991 samples =
  0.0619375 s. Derived from SincNet conv geometry
  (`kernel_size=[251,3,5,3,5,3]`, `stride=[10,3,1,3,1,3]`, `PyanNet.SINCNET_DEFAULTS = {"stride": 10}`)
  and asserted in the export script, lines 72–77.
- **Embedded ONNX metadata (values, MEASURED)**:
  `model_type=pyannote-segmentation-3.0`, `sample_rate=16000`, `window_size=160000`,
  `receptive_field_size=991`, `receptive_field_shift=270`, `num_speakers=3`,
  `num_classes=7`, `powerset_max_classes=2`, `version=1`, `model_author=pyannote`,
  `maintainer=k2-fsa`, plus `license`/`url_1`/`url_2`.
  Careful: `num_speakers=3` is `len(specifications.classes)` and `num_classes=7` is the
  powerset dimension (export script 118–120) — the names read backwards.
  **`window_shift` is NOT in the metadata** (§1.1).
- **Difference from the HF original: none of substance.** The log-softmax is in the
  PyTorch model, not added by the export; `torch.onnx.export` is called plain, opset 13;
  the only additions are the metadata block and a separate `model.int8.onnx`
  (`quantize_dynamic`, QUInt8). `window_size=160000` is advisory metadata, not a graph
  constraint.

---

## 3. VBx in shippable form

### 3.1 The crux: the PLDA is in the *preprocessing*, not the VB loop

`VBx/VBx.py:27` (BUT VBx):

```python
def VBx(X, Phi, loopProb=0.9, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10,
        epsilon=1e-4, alphaQInit=1.0, ...)
```

Docstring: *"Phi — D array with across-class covariance matrix diagonal. The model
assumes zero mean, diagonal across-class and identity within-class covariance
matrix."* **`Phi` is a D-vector, not a PLDA object.** The whole loop is:

```python
G     = -0.5*(np.sum(X**2, axis=1, keepdims=True) + D*np.log(2*np.pi))   # (23)
V     = np.sqrt(Phi); rho = X * V                                        # (18)
invL  = 1.0 / (1 + Fa/Fb * gamma.sum(0, keepdims=True).T * Phi)          # (17)
alpha = Fa/Fb * invL * gamma.T.dot(rho)                                  # (16)
log_p_= Fa * (rho.dot(alpha.T) - 0.5*(invL+alpha**2).dot(Phi) + G)       # (23)
tr    = np.eye(len(pi))*loopProb + (1-loopProb)*pi                       # (1)
gamma, log_pX_, ... = forward_backward(log_p_, tr, pi)                   # (19)-(22)
```

Paper: Landini/Profant/Diez/Burget, *Bayesian HMM clustering of x-vector sequences*,
Computer Speech & Language 71:101254 (2022), arXiv:2012.14952. §2.3 states the
assumption explicitly: *"we assume that the x-vectors are linearly transformed into a
space where Σ_b is diagonal and Σ_w is identity"* (transform Eq. 2–3, with truncation
doubling as LDA). §2.5 on the scales: *"The theoretically correct values … are
F_A = F_B = 1. However, choosing different values gives us finer control."*
`loopProb` is Eq. (1).

**License, BUT VBx (github.com/BUTSpeechFIT/VBx): there is NO LICENSE file.** GitHub API
reports `"license": null`. Apache-2.0 appears only as per-file headers in `VBx/VBx.py`,
`VBx/diarization_lib.py`, `VBx/vbhmm.py`. **`VBx/models/` (containing `plda` and
`transform.h5`) carries no license at all** — UNVERIFIED whether the file headers were
meant to extend to the weights. **Do not vendor `VBx/models/`.**

### 3.2 BUT's preprocessing chain and published hyperparameters

`VBx/vbhmm.py:107-158`. Two corrections to common belief: there is no `AHC()` function
(AHC is inline `fastcluster.linkage` + `fcluster`), and there is no `mean.vec`/
`transform.txt` (that is the Kaldi convention) — VBx uses `transform.h5` with HDF5 keys
`mean1`, `mean2`, `lda`, plus a Kaldi-format `plda`.

```
x-vector → −mean1 → L2-norm → lda projection → −mean2 → L2-norm      # embedding space
        → −plda_mu → ·plda_tr.T → [:lda_dim]                         # VB input space
```
then `VBx(fea, plda_psi[:lda_dim], pi=…, gamma=qinit, maxIters=40, epsilon=1e-6,
loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)`. `maxIters=40, epsilon=1e-6` are
hardcoded, not CLI args.

**AHC init is already cosine-only** (`vbhmm.py:132-151`): `cos_similarity(x)` →
per-recording threshold from `twoGMMcalib_lin` (a 2-Gaussian calibration of the score
histogram, `diarization_lib.py:13`) offset by `--threshold` → `fastcluster.linkage(...,
method='average')` → `fcluster(criterion='distance')` → one-hot → `softmax(q *
init_smoothing)`. No PLDA needed for the init. `kaldi_ivector_plda_scoring_dense` is
dead code on master.

Published recipes:

| Recipe | Fa | Fb | loopP | thr | lda_dim | smoothing |
|---|---|---|---|---|---|---|
| `AMI_run.sh:44-49` (16 kHz ResNet101) | **0.4** | **64** (68 for Mix-Headset, line 6) | **0.65** | −0.015 | 128 | 7.0 |
| `DIHARD2_run.sh:42-47` (16 kHz) | 0.2 | 6 | 0.35 | −0.015 | 128 | 7.0 |
| `CALLHOME_run.sh:42-47` (8 kHz) | 0.4 | 17 | 0.40 | −0.015 | 128 | 7.0 |
| `run_example.sh` (16 kHz demo) | 0.3 | 17 | 0.99 | −0.015 | 128 | — |

The AMI row is the 16 kHz meeting-domain recipe. Note the spread (Fb 6→64,
loopP 0.35→0.99): these are heavily per-corpus tuned, which is itself evidence they
will not transfer to a different embedding.

### 3.3 VBx without a PLDA — what exists

**Spherical `Phi` is a legal special case of the existing API** (`Phi = φ·ones(D)` is
diagonal), consistent with §2.3 of arXiv:2012.14952. *Epistemic status: inferred from
the API contract + the paper, not measured.* What it does not fix is the **scale**:
`Phi` is between-class variance relative to a unit within-class variance, and raw
L2-normalised ERes2Net vectors have within-class covariance ≪ I. Worse, `Fa/Fb·Phi`
enters Eqs. (16)–(17) only as a product — φ is degenerate with `Fa/Fb`, so you gain a
free parameter without gaining a degree of freedom.

**The one published PLDA-free parameterization: SphereVBx-PF**, arXiv:2606.24528
(Pálka, Han, Singh, Delcroix, Tawara, Burget; submitted 2026-06-23). Replaces the
Gaussian PLDA backend with T-PSDA → VB over a mixture of von Mises–Fisher
distributions. The parameter-free variant is the special case **κ_b = 0, κ_w = 1,
d = D**, for which *"the resulting log-likelihood ratio scores … become a monotonic
function of cosine similarity"* — requires unit-norm embeddings, which we already have.
Result: MS-SphereVBx-PF 12.48 avg DER vs VBx/DiariZen 12.65 over 8 benchmarks — the
parameter-free variant **matched or beat** the PLDA one.

§6.2 gives the only guidance on the scales: *"To compensate for the absence of a
trained T-PSDA backend, we set FA>10 and FB<1, unlike typical settings (FA<1, FA<FB)."*
**Exact FA/FB values are NOT published.** The paper claims code at
github.com/BUTSpeechFIT/DiariZen; the subagent enumerated that tree and found **zero**
paths matching `sphere|psda|vmf|vonmises` — **UNRELEASED as of 2026-08-07.**

**No shipped implementation runs VBx without a downloaded PLDA:**

| Repo | VBx? | Needs PLDA? | Path | License |
|---|---|---|---|---|
| BUT VBx | yes | yes (`plda` + `transform.h5`) | `VBx/VBx.py`, `VBx/vbhmm.py` | Apache-2.0 headers only, **no LICENSE file** |
| DiariZen | yes | yes (`xvec_transform.npz` + `plda.npz`) | `diarizen/clustering/VBx.py:158-194` | repo MIT; that file Apache-2.0 header |
| pyannote-audio 4.x | yes | yes (`plda.npz` + `xvec_transform.npz`) | `src/pyannote/audio/utils/vbx.py`, `core/plda.py` | MIT |
| NeMo | no | — | `offline_clustering.py` (NME-SC instead) | Apache-2.0 |
| 3D-Speaker | no | — | `speakerlab/process/cluster.py` (all cosine) | Apache-2.0 |
| WeSpeaker | no (ships a PLDA *trainer*) | — | `wespeaker/utils/plda/two_cov_plda.py` | Apache-2.0 |

Nobody estimates the PLDA from the meeting's own embeddings — unsurprising, 2–6
speakers cannot support a 192×192 between-class covariance.

**DVBx** (github.com/BUTSpeechFIT/DVBx, ICASSP 2024, arXiv:2310.02732) — **MIT with a
real LICENSE file.** Backpropagates through the VB loop to discriminatively train
`Fa`, `Fb`, `loop_prob`, `init_smoothing` *and* the PLDA covariances. It does not
remove the PLDA; it trains it. Needs labeled diarization training data.

### 3.4 pyannote DOES use VBx (correction to a common belief)

`src/pyannote/audio/pipelines/speaker_diarization.py:206-210`:
```python
plda: PipelinePLDA = {"checkpoint": "pyannote/speaker-diarization-community-1",
                      "subfolder": "plda"},
clustering: str = "VBxClustering",
```
`Clustering` enum (`clustering.py:759`): `AgglomerativeClustering, KMeansClustering,
VBxClustering, OracleClustering`.

**Model files, `pyannote/speaker-diarization-community-1`** (HF API, public metadata):
`config.yaml`, `embedding/pytorch_model.bin`, `segmentation/pytorch_model.bin`,
**`plda/README.md`, `plda/plda.npz`, `plda/xvec_transform.npz`**.
**License `cc-by-4.0`** (commercial use OK with attribution), `gated: auto`.
PLDA "trained by BUT Speech@FIT group" per the bundled README.
`pyannote/speaker-diarization-3.1` is MIT and ships **no** plda files — 3.1 is AHC-only.

**You cannot reuse this PLDA**: it is fit to pyannote's WeSpeaker-derived embedding;
the `lda` matrix will not accept 192-dim ERes2Net input. (Its exact input dim is
UNVERIFIED — the file is behind the gate.)

Two implementation details worth stealing:

- **pyannote deleted the HMM.** `utils/vbx.py:27` drops `loopProb` from the signature;
  lines 117–128 replace forward-backward with a GMM update (`logsumexp(log_p_ + lpi,
  axis=-1)`). Signature: `cluster_vbx(ahc_init, fea, Phi, Fa, Fb, maxIters=20,
  init_smoothing=7.0)`. So community-1's "VBx" is VB-**GMM**, no loop probability.
- **pyannote rescales where BUT does not** (`vbx.py:211`):
  `sqrt(lda.shape[1]) * l2_norm(lda.T.dot(sqrt(lda.shape[0]) * l2_norm(x - mean1).T).T - mean2)`.
  Those `sqrt(D)` factors are absent from BUT's `vbhmm.py`. This is exactly the scale
  calibration §3.3 warns about, and it is why `Fa=0.07` (pyannote) and `Fa=0.4` (BUT)
  are not comparable numbers.
- pyannote's AHC init is `linkage(normed, method="centroid", metric="euclidean")` —
  centroid/Euclidean, not BUT's average/cosine.

### 3.5 Recommendation

**Nobody has published `Fa`/`Fb`/`loopP` for 192-dim ERes2Net cosine embeddings.**
Stating that plainly rather than handing over numbers that would look authoritative.
The two nearest anchors are mutually inconsistent because they are different models:
pyannote `Fa=0.07, Fb=0.8` (Gaussian VB-GMM, *with* PLDA, unit-norm) and SphereVBx-PF
`FA>10, FB<1` (vMF, no PLDA).

Ranked:

- **A — train a two-covariance PLDA for ERes2Net, then run stock VBx.** Turns an
  unsolved tuning problem into a solved one. `wespeaker/utils/plda/two_cov_plda.py`
  (Apache-2.0, `class TwoCovPLDA` with `train`/`em_one_iter`/`transform_embedding`/
  `save_model`, CLI `wespeaker/bin/train_plda.py`) produces exactly the `mu`/`tr`/`psi`
  triple BUT's `read_plda` consumes. Needs a speaker-labeled corpus (VoxCeleb2 dev),
  not diarization labels. Then start from `AMI_run.sh:44-49`:
  `Fa=0.4, Fb=64, loopP=0.65, thr=-0.015, lda_dim=128, smoothing=7.0`; sweep `Fb`
  first (the speaker-count knob), then `loopP`. With 192-dim input, consider
  `lda_dim` 96–128.
- **B — SphereVBx-PF from the paper.** Genuinely no PLDA and it beat the PLDA
  baseline. κ_b=0, κ_w=1, d=D on the already-L2-normalised 192-dim vectors; start
  `FA > 10`, `FB < 1`. Cost: implement vMF VB from Eqs. (3)–(8) yourself — the code
  is not released despite the paper's claim.
- **C — spherical `Phi` in stock VBx.** Cheapest, least supported, and φ is degenerate
  with `Fa/Fb`. Only after A or B.
- **D — skip VBx.** NME-SC (auto-tuning spectral clustering, Park et al., IEEE SPL 27,
  arXiv:2003.02405; `nemo/.../offline_clustering.py:868 class NMESC`, `SpectralClustering:755`,
  `getCosAffinityMatrix:423`, Apache-2.0) gives auto speaker-count and no tuned
  threshold on the cosine matrix we already have.

**Worth adopting regardless of which route:** BUT's `twoGMMcalib_lin` per-recording
adaptive AHC threshold (`vbhmm.py:135-145`, `diarization_lib.py:13`). It needs no
PLDA and removes a global threshold constant.

Other permissive alternatives, one line each:
`getMultiScaleCosAffinityMatrix` (NeMo, Apache-2.0) — multi-scale affinity fusion buys
short-segment resolution; `SpectralCluster` (3D-Speaker `cluster.py:23`, Apache-2.0) —
drop-in for AHC with eigengap count estimation; `UmapHdbscan` (`cluster.py:117`) — no k,
no threshold. **No permissively-licensed overlap-aware resegmenter was verified** —
pyannote 4.x `main` has no `resegmentation.py` (only an unmerged `feat/resegmentation`
branch); DiariZen multi-domain weights are CC-BY-NC-4.0, only
`BUT-FIT/diarizen-meeting-base` is MIT.

---

## 4. Numbers to diagnose parity failures against

**sherpa (our current baseline)** — `min_duration_on` and `min_duration_off` are
sherpa defaults we never override; `clustering_threshold=0.5` is our own default
(`src/stenograf/diarization/sherpa.py:45`) and happens to equal sherpa's.

| Knob | Value | Semantics |
|---|---|---|
| `window_size` | 160000 (10.0 s) | |
| `window_shift` | 16000 (1.0 s) | 0.1 ratio; not configurable in installed 1.13.4 |
| frames/chunk | 589 | |
| `receptive_field_shift` | 270 (0.016875 s) | output time grid |
| `receptive_field_size` | 991 (0.0619375 s) | half of it is the time offset |
| clustering `threshold` | **0.5** | **cosine DISTANCE**, complete linkage, cut at first height ≥ threshold |
| clustering `num_clusters` | −1 | > 0 ⇒ `cutree_k`, threshold ignored |
| min frames to embed | 10 (≈0.169 s) | per (chunk, local speaker), post-ExcludeOverlap, summed |
| `min_duration_on` | 0.3 s | strict `>`, applied after merging |
| `min_duration_off` | 0.5 s | gap merged when `0 < gap <= 0.5` |
| overlap in embeddings | excluded | `ExcludeOverlap`, row-sum ≥ 2 → frame zeroed |
| `min_cluster_size` | **none** | no equivalent exists |

**pyannote speaker-diarization-3.1** — config.yaml. Canonical HF URL is gated (401);
values recovered from three mirrors that agree byte-for-byte, primary
`https://modelscope.cn/models/pyannote/speaker-diarization-3.1/resolve/master/config.yaml`.

```yaml
pipeline.params:
  clustering: AgglomerativeClustering
  embedding: pyannote/wespeaker-voxceleb-resnet34-LM
  embedding_batch_size: 32
  embedding_exclude_overlap: true
  segmentation: pyannote/segmentation-3.0
  segmentation_batch_size: 32
params:
  clustering: {method: centroid, min_cluster_size: 12, threshold: 0.7045654963945799}
  segmentation: {min_duration_off: 0.0}
```
Sliding window: duration **10.0 s**, step **1.0 s** (`segmentation_step: float = 0.1`
as a ratio of duration, `speaker_diarization.py:118`; not overridden in config).
There is **no `segmentation.threshold`** and structurally cannot be — `__init__`
(v3.1.1, lines 152–161) branches on `specifications.powerset` and the powerset branch
exposes only `min_duration_off`. (It vanished at 3.0, not 3.1; 2.1 had
`threshold: 0.4442`, `min_duration_off: 0.5817`, `min_cluster_size: 15`.)

**Threshold-units trap.** pyannote's 0.7046 is a **Euclidean distance on L2-normalised
vectors**, not a cosine distance (`clustering.py:369-376`: for `centroid`/`median`/`ward`
it unit-normalises then calls `linkage(..., metric="euclidean")`, and line 322 comments
`Uniform(0.0, 2.0)  # assume unit-normalized embeddings`; applied via
`fcluster(criterion="distance")`, line 385). Convert with `cos = 1 − t²/2`:

> **0.7046 Euclidean ⇔ cosine similarity 0.752 ⇔ cosine distance 0.248 ⇔ 41.25°.**
> When comparing against sherpa's 0.5 cosine-distance threshold or this repo's
> `voiceprints.py` cosine similarities, **0.752 is the number, not 0.705.**

pyannote's `min_cluster_size` machinery (no sherpa equivalent, `clustering.py:359-363`
and `457-479`):
```python
min_cluster_size = min(self.min_cluster_size, max(1, round(0.1 * num_embeddings)))
...
small_clusters = cluster_unique[cluster_counts < min_cluster_size]
centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)  # cosine!
for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
    clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]
```
Only "large" clusters count toward the speaker count; small ones are absorbed into the
nearest large centroid. The floor saturates at n ≥ 119 embeddings and is effectively 1
below n = 15. Note the asymmetry: dendrogram in Euclidean space, reassignment in cosine.
If `num_clusters` is constrained, pyannote abandons the distance criterion entirely and
re-cuts by iteration index searching outward from the threshold (lines 405–451).
Edge case: `num_large_clusters == 0` collapses everything to cluster 0 (453–455).

**pyannote community-1** — config.yaml is gated; **SINGLE-SOURCED** from
`https://modelscope.cn/models/pyannote/speaker-diarization-community-1/resolve/master/config.yaml`,
but corroborated by `speaker_diarization.py:289-293 default_parameters()` in the MIT
source, which returns the identical numbers:

```yaml
pipeline.params: {clustering: VBxClustering, segmentation: $model/segmentation,
                  segmentation_batch_size: 32, embedding: $model/embedding,
                  embedding_batch_size: 32, embedding_exclude_overlap: true,
                  plda: $model/plda}
params: {clustering: {threshold: 0.6, Fa: 0.07, Fb: 0.8},
         segmentation: {min_duration_off: 0.0}}
```
Search ranges (`clustering.py:568-570`): `threshold=Uniform(0.5,0.8)`,
`Fa=Uniform(0.01,0.5)`, `Fb=Uniform(0.01,15.0)`; `maxIters=20` hardcoded.
No `min_cluster_size` — VBxClustering declares none. Its 0.6 is the **AHC-init**
threshold in the PLDA-transformed space, **not comparable to 3.1's 0.7046 nor to
sherpa's 0.5.** Embedding is a bundled copy of `pyannote/wespeaker-voxceleb-resnet34-LM`.
Segmentation window duration for the bundled checkpoint: **UNVERIFIED**
(specifications not published); the 0.1 step ratio still applies.

Licenses (HF API): community-1 **cc-by-4.0**; 3.1 **mit**; segmentation-3.0 **mit**;
wespeaker-voxceleb-resnet34-LM **cc-by-4.0** (ungated). The CC-BY attribution
obligation was already inherited by 3.1 through its embedding, so the MIT→CC-BY
headline overstates the delta.

---

## 5. UNVERIFIED / open

- SphereVBx-PF's actual `FA`/`FB` values — inequalities only; code unreleased despite
  the paper's claim.
- The embedding dimension community-1's `plda.npz` was fit for (file gated).
- community-1's segmentation window duration (bundled checkpoint, specs unpublished).
- Whether BUT VBx's per-file Apache-2.0 headers extend to `VBx/models/` binaries —
  no statement exists either way.
- "587 frames" — no traceable source found; 589 is measured three independent ways.
- Two second-hand hyperparameter claims the subagent could not extract from the source
  PDFs (`Fb=1.0, loop_prob=0.5`; VoxConverse `Fa=0.3, Fb=16, loopP=0.9` attributed to
  arXiv:2209.09635) — **not standing behind either.**
