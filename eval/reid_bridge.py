"""Can saved voice profiles survive an embedding-model change by translation?

A profile entry is a vector from the embedder that made it; a new embedder
reads none of them. The bridge: a map fitted on corpus audio both models
embedded, applied once to every stored old vector, so the profiles keep
naming people under the new model until its own vectors replace them.

Arms, on ``reid_models``' frozen items and trials (no re-embedding):

- ``base``: the old model end to end — what a user had before the change.
- ``redimnet2-b6-vb2-lm``: the new model end to end — the ceiling.
- ``bridge-*``: profile entries are translated ``base`` vectors, trials are
  native new-model vectors — the state right after an upgrade.

The map never sees the people it is tested on: AMI and Nemotron items fold by
meeting series (leave one series out, fit on the other five), and the user's
calls are tested with a map fitted on AMI alone — as a shipped map would be,
fitted on public audio. ``bridge-shuffled`` fits on mispaired spans and must
collapse; ``bridge-*-leaky`` fits on everything, test people included, and
bounds what a map of that family can do at all. The learning curve refits the
rotation on 1–5 training series to show whether more voices would help.

    uv run --group eval eval/reid_bridge.py   # → out/reid_models/bridge.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from reid_models import (
    EMB_DIR,
    OUT,
    PRIMARY_SETS,
    Arrays,
    _macro,
    load_vectors,
    summary,
    trial_scores,
)

OLD, NEW = "base", "redimnet2-b6-vb2-lm"
PROD_THRESHOLD = 0.621  # stenograf.voiceprints.DEFAULT_THRESHOLD, the new model's
RIDGE_ALPHA = 1.0
SEED = 20261119
CURVE_DRAWS = 10

Map = Callable[[np.ndarray], np.ndarray]


def series(item_id: str) -> str:
    kind, rest = item_id.split("/", 2)[:2]
    if kind == "own":
        return "own"
    return rest.split(".")[0][:-1]  # ES2004a / ES2004a.duo → ES2004


def spans(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(EMB_DIR / f"{name}.npz", allow_pickle=False)
    emb, off, ids = d["emb"], d["offsets"], [str(i) for i in d["item_ids"]]
    owner = np.repeat(np.arange(len(ids)), np.diff(off))
    return emb, np.array([series(ids[k]) for k in owner])


# numpy on macOS Accelerate raises spurious overflow/invalid flags in matmul on
# finite unit vectors (results checked finite, |v| ≤ 0.31); errstate silences them.


def fit_ridge(x: np.ndarray, y: np.ndarray) -> Map:
    xa = np.hstack([x, np.ones((len(x), 1))])
    reg = RIDGE_ALPHA * np.eye(xa.shape[1])
    reg[-1, -1] = 0.0
    with np.errstate(all="ignore"):
        w = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return lambda v: np.append(v, 1.0) @ w


def fit_procrustes(x: np.ndarray, y: np.ndarray) -> Map:
    """The rotation that best carries old vectors onto new ones — angles are
    preserved, so the old model's similarity structure survives intact."""
    with np.errstate(all="ignore"):
        u, _, vt = np.linalg.svd(x.T @ y)
        r = u @ vt
    return lambda v: v @ r


FITS = {"ridge": fit_ridge, "procrustes": fit_procrustes}


def main() -> None:
    items = {i["id"]: i for i in json.loads((OUT / "manifest.json").read_text())["items"]}
    sets = json.loads((OUT / "trials.json").read_text())
    old_vecs = load_vectors(EMB_DIR / f"{OLD}.npz", items)
    new_vecs = load_vectors(EMB_DIR / f"{NEW}.npz", items)
    profile_items = {e for s in sets.values() for es in s["profiles"].values() for e in es}

    x_all, ser = spans(OLD)
    y_all, ser_new = spans(NEW)
    assert (ser == ser_new).all()
    ok = np.isfinite(x_all).all(1) & np.isfinite(y_all).all(1)
    ami_series = sorted(set(ser) - {"own"})
    print(f"{ok.sum()} span pairs, {len(ami_series)} AMI series + own calls")
    rng = np.random.default_rng(SEED)

    def bridged(
        fit: Callable[[np.ndarray, np.ndarray], Map],
        train_for: Callable[[str], list[str]],
        shuffle: bool = False,
    ) -> dict[str, np.ndarray]:
        """Trials native; every profile entry translated by a map fitted on
        ``train_for(its series)``."""
        maps: dict[str, Map] = {}
        out = dict(new_vecs)
        for iid in profile_items:
            s = series(iid)
            if s not in maps:
                sel = ok & np.isin(ser, train_for(s))
                x, y = x_all[sel], y_all[sel]
                if shuffle:
                    y = y[rng.permutation(len(y))]
                maps[s] = fit(x, y)
            if iid in old_vecs:
                with np.errstate(all="ignore"):
                    v = maps[s](old_vecs[iid])
                out[iid] = v / np.linalg.norm(v)
        return out

    def held_out(s: str) -> list[str]:
        return [a for a in ami_series if a != s]

    def everything(_: str) -> list[str]:
        return [*ami_series, "own"]

    arms = {OLD: old_vecs, NEW: new_vecs}
    for kind, fit in FITS.items():
        arms[f"bridge-{kind}"] = bridged(fit, held_out)
        arms[f"bridge-{kind}-leaky"] = bridged(fit, everything)
    arms["bridge-shuffled"] = bridged(fit_ridge, held_out, shuffle=True)

    results = {m: {n: trial_scores(s, v) for n, s in sets.items()} for m, v in arms.items()}
    lines = ["# Profile bridge: old-model profiles translated, new-model trials", ""]
    lines += summary(results)

    lines += [
        "",
        f"## At the shipped threshold {PROD_THRESHOLD} (named right / stranger named)",
        "",
        "| model | set | 2s | 3s | full |",
        "|---|---|---|---|---|",
    ]
    for m, by_set in results.items():
        for n, rows in by_set.items():
            if n == "owncalls":
                rows = [r for r in rows if r["identity"] != "self"]
            cells = []
            for d in ("2s", "3s", "full"):
                rs = [r for r in rows if r["duration"] == d]
                named = np.mean([r["top_ok"] and r["target"] >= PROD_THRESHOLD for r in rs])
                fa = np.mean([r["stranger"] >= PROD_THRESHOLD for r in rs])
                cells.append(f"{100 * named:.1f} / {100 * fa:.1f}")
            lines.append(f"| {m} | {n} | " + " | ".join(cells) + " |")

    lines += ["", "## Fidelity: cos(translated old, native new) per profile entry", ""]
    for m in [a for a in arms if a.startswith("bridge")]:
        per = defaultdict(list)
        for n, s in sets.items():
            for es in s["profiles"].values():
                per[n] += [float(arms[m][e] @ new_vecs[e]) for e in es if e in new_vecs]
        lines.append(f"- {m}: " + ", ".join(f"{n} {np.median(v):.3f}" for n, v in per.items()))

    lines += [
        "",
        "## Learning curve: the rotation fitted on k held-out series",
        "",
        "Macro DIR at 1 % macro stranger accepts, as the summary. k < 5 averages "
        f"{CURVE_DRAWS} random draws of the k series per test series.",
        "",
        "| training series (people) | fidelity | 2s | 3s | full |",
        "|---|---|---|---|---|",
    ]
    for k in range(1, len(ami_series)):
        draws = CURVE_DRAWS if k < len(ami_series) - 1 else 1
        fid, dirs = [], defaultdict(list)
        for rep in range(draws):
            # Seeded per test series: maps are fitted in set order, which
            # string hashing changes between runs.
            def subset(s: str, k=k, rep=rep) -> list[str]:
                rng = np.random.default_rng([SEED, k, rep, [*ami_series, "own"].index(s)])
                return list(rng.choice(held_out(s), k, replace=False))

            vecs = bridged(fit_procrustes, subset)
            fid.append(np.median([vecs[e] @ new_vecs[e] for e in profile_items if e in vecs]))
            res = {n: trial_scores(s, vecs) for n, s in sets.items()}
            for d in ("2s", "3s", "full"):
                arrs = []
                for n in PRIMARY_SETS:
                    if n == "owncalls-remote":
                        rows = [r for r in res["owncalls"] if r["identity"] != "self"]
                    else:
                        rows = res[n]
                    arrs.append(Arrays([r for r in rows if r["duration"] == d]))
                dirs[d].append(_macro(arrs, None, 0.01)[0])
        lines.append(
            f"| {k} ({4 * k}) | {np.mean(fid):.3f} | "
            + " | ".join(f"{100 * np.mean(dirs[d]):.1f}" for d in ("2s", "3s", "full"))
            + " |"
        )

    out = OUT / "bridge.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
