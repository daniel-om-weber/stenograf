"""Assert that runner outputs honour the ``embedio`` .npz interface::

python3 eval/embedders/check.py --manifest M.json OUT.npz [OUT.npz ...]
"""

import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("npz", nargs="+")
args = p.parse_args()
with open(args.manifest) as f:
    items = json.load(f)["items"]
for path in args.npz:
    with np.load(path, allow_pickle=False) as z:
        keys = {"emb", "offsets", "item_ids", "model", "seconds_cpu", "audio_seconds"}
        assert set(z.files) == keys, z.files
        emb, off = z["emb"], z["offsets"]
        assert emb.dtype == np.float32 and emb.ndim == 2, (emb.dtype, emb.shape)
        counts = [0] + [len(i["spans"]) for i in items]
        assert off.dtype == np.int64 and off.tolist() == np.cumsum(counts).tolist()
        assert list(z["item_ids"]) == [i["id"] for i in items]
        ok = ~np.isnan(emb).any(axis=1)
        assert np.isnan(emb[~ok]).all(), "a row is partly NaN"
        assert np.allclose(np.linalg.norm(emb[ok], axis=1), 1, atol=1e-5)
        assert z["model"].dtype.kind == "U" and z["model"].ndim == 0
        assert float(z["seconds_cpu"]) > 0 and float(z["audio_seconds"]) > 0
        nan_rows = int((~ok).sum())
        print(
            f"OK {path}: model={z['model']} dim={emb.shape[1]} rows={len(emb)} nan_rows={nan_rows}"
        )
