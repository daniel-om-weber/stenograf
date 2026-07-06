"""Blind adjudication of model disagreements — WER without hand-written references.

Aligns all backend hypotheses per segment, finds the spots where they disagree,
cuts a short audio snippet around each spot, and writes a single self-contained
HTML page. The reviewer listens and clicks the correct variant; the variants are
shuffled and unlabeled, so no model gets anchor-bias favoritism. Where models
agree, no human time is spent.

Usage:
    uv run --group eval eval/adjudicate.py                  # → eval/out/adjudication.html
    uv run --group eval eval/adjudicate.py --max-sites 40   # cap sites per segment
    uv run --group eval eval/adjudicate.py --score ~/Downloads/adjudication-results.json

Open the HTML in a browser, judge each site (keys 1–9, 0 = unsure), then click
"Download results" and run --score on the downloaded file.
"""

# ruff: noqa: E501  (embedded HTML/JS template)
from __future__ import annotations

import argparse
import base64
import difflib
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import OUT_DIR, load_manifest
from score import normalize

PIVOT = "whisper"  # only backend with reliable word-level timestamps in its output
CONTEXT_WORDS = 3
SNIPPET_PAD_S = 0.4


@dataclass
class PivotWord:
    display: str
    norm: str
    start: float
    end: float


def load_pivot_words(segment_id: str) -> list[PivotWord]:
    record = json.loads((OUT_DIR / PIVOT / f"{segment_id}.json").read_text())
    words = []
    for seg in record["segments"]:
        for w in seg["words"]:
            norm = normalize(w["text"])
            if norm:
                words.append(PivotWord(w["text"], norm, w["start"], w["end"]))
    return words


def load_hyp_words(backend: str, segment_id: str) -> list[tuple[str, str]]:
    """(display, normalized) words of a backend's transcript."""
    record = json.loads((OUT_DIR / backend / f"{segment_id}.json").read_text())
    out = []
    for token in record["text"].split():
        norm = normalize(token)
        if norm:
            out.append((token, norm))
    return out


def align_to_pivot(
    pivot_norms: list[str], hyp: list[tuple[str, str]]
) -> list[tuple[str, int, int, int, int]]:
    """difflib opcodes aligning the hypothesis onto the pivot word sequence."""
    matcher = difflib.SequenceMatcher(a=pivot_norms, b=[n for _, n in hyp], autojunk=False)
    return matcher.get_opcodes()


def hyp_span(
    opcodes: list[tuple[str, int, int, int, int]],
    hyp: list[tuple[str, str]],
    s: int,
    e: int,
) -> list[str]:
    """The hypothesis' own display words aligned onto pivot range [s, e)."""
    words: list[str] = []
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            lo, hi = max(i1, s), min(i2, e)
            if lo < hi:
                words.extend(d for d, _ in hyp[j1 + (lo - i1) : j1 + (hi - i1)])
        elif (i1 < e and i2 > s) or (i1 == i2 and s <= i1 < e):
            words.extend(d for d, _ in hyp[j1:j2])
    return words


def build_sites(segment_id: str, backends: list[str]) -> list[dict]:
    pivot_words = load_pivot_words(segment_id)
    pivot_norms = [w.norm for w in pivot_words]

    # align every backend once; collect contested pivot spans
    aligned = {
        backend: (align_to_pivot(pivot_norms, hyp), hyp)
        for backend in backends
        if backend != PIVOT
        for hyp in [load_hyp_words(backend, segment_id)]
    }
    spans = sorted(
        (i1, i2)
        for opcodes, _ in aligned.values()
        for op, i1, i2, _, _ in opcodes
        if op != "equal"
    )

    # merge overlapping/nearby contested spans into sites
    sites: list[list[int]] = []
    for i1, i2 in spans:
        if sites and i1 <= sites[-1][1] + 2:
            sites[-1][1] = max(sites[-1][1], i2)
        else:
            sites.append([i1, i2])

    out = []
    for s, e in sites:
        variants: dict[str, list[str]] = {PIVOT: [w.display for w in pivot_words[s:e]]}
        for backend, (opcodes, hyp) in aligned.items():
            variants[backend] = hyp_span(opcodes, hyp, s, e)
        # collapse to unique variants; space-insensitive key so German compound
        # orthography ("Hauptbahnhof" vs "Haupt Bahnhof" vs "Haupt-Bahnhof")
        # doesn't masquerade as a recognition difference
        unique: dict[str, dict] = {}
        for backend, words in variants.items():
            key = normalize(" ".join(words)).replace(" ", "")
            unique.setdefault(key, {"text": " ".join(words) or "(nichts)", "models": []})
            unique[key]["models"].append(backend)
        if len(unique) < 2:
            continue
        ctx_lo, ctx_hi = max(0, s - CONTEXT_WORDS), min(len(pivot_words), e + CONTEXT_WORDS)
        t0 = pivot_words[ctx_lo].start if ctx_lo < len(pivot_words) else 0.0
        t1 = pivot_words[min(ctx_hi, len(pivot_words) - 1)].end
        out.append(
            {
                "segment": segment_id,
                "before": " ".join(w.display for w in pivot_words[ctx_lo:s]),
                "after": " ".join(w.display for w in pivot_words[e:ctx_hi]),
                "t0": max(0.0, t0 - SNIPPET_PAD_S),
                "t1": t1 + SNIPPET_PAD_S,
                "variants": list(unique.values()),
            }
        )
    return out


def snippet_b64(wav: Path, t0: float, t1: float) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(t0), "-to", str(t1), "-i", str(wav),
             "-c:a", "libmp3lame", "-b:a", "48k", tmp.name],
            check=True,
        )
        return base64.b64encode(Path(tmp.name).read_bytes()).decode()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>stenograf adjudication</title>
<style>
body{font-family:system-ui;max-width:44rem;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
.ctx{color:#888} .site{display:none} .site.active{display:block}
button.variant{display:block;width:100%;text-align:left;margin:.4rem 0;padding:.7rem;font-size:1rem;
 background:#222;color:#eee;border:1px solid #444;border-radius:.5rem;cursor:pointer}
button.variant:hover{background:#334} .picked{background:#254d25!important}
#bar{height:4px;background:#4a4;transition:width .2s} audio{width:100%;margin:.8rem 0}
.meta{color:#777;font-size:.85rem}
</style></head><body>
<div id="bar" style="width:0%"></div>
<p class="meta">Hör den Ausschnitt an und wähle die korrekte Variante (Tasten 1–9, 0 = unsicher, ␣ = nochmal abspielen).</p>
<div id="sites"></div>
<div id="done" style="display:none"><h2>Fertig ✓</h2></div>
<button class="variant" onclick="download()">Download results (auch zwischendurch)</button>
<script>
const SITES = __SITES__;
const STORE = 'stenograf-adjudication-' + SITES.length;
const picks = JSON.parse(localStorage.getItem(STORE) || '{}');
let cur = Object.keys(picks).length;
const root = document.getElementById('sites');
SITES.forEach((s, i) => {
  const d = document.createElement('div');
  d.className = 'site'; d.id = 'site' + i;
  d.innerHTML = `<p class="meta">${i + 1} / ${SITES.length} — ${s.segment}</p>
    <audio controls src="data:audio/mpeg;base64,${s.audio}"></audio>
    <p><span class="ctx">…${s.before} </span><b>???</b><span class="ctx"> ${s.after}…</span></p>` +
    s.variants.map((v, j) =>
      `<button class="variant" onclick="pick(${i},${j})">${j + 1}. ${v.text}</button>`).join('') +
    `<button class="variant" onclick="pick(${i},-1)">0. unsicher / keine</button>`;
  root.appendChild(d);
});
function show(i) {
  document.querySelectorAll('.site').forEach(e => e.classList.remove('active'));
  document.getElementById('bar').style.width = (100 * i / SITES.length) + '%';
  if (i >= SITES.length) { document.getElementById('done').style.display = 'block'; return; }
  const el = document.getElementById('site' + i);
  el.classList.add('active');
  el.querySelector('audio').play().catch(() => {});
}
function pick(i, j) {
  picks[i] = j; localStorage.setItem(STORE, JSON.stringify(picks));
  cur = i + 1; show(cur);
}
document.addEventListener('keydown', e => {
  if (cur >= SITES.length) return;
  if (e.key === ' ') { e.preventDefault(); const a = document.querySelector('.site.active audio'); a.currentTime = 0; a.play(); }
  const n = parseInt(e.key);
  if (e.key === '0') pick(cur, -1);
  else if (n >= 1 && n <= SITES[cur].variants.length) pick(cur, n - 1);
});
function download() {
  const results = SITES.map((s, i) => ({
    segment: s.segment,
    variants: s.variants.map(v => ({ models: v.models })),
    picked: picks[i] ?? null,
  }));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(results, null, 1)], { type: 'application/json' }));
  a.download = 'adjudication-results.json';
  a.click();
}
show(0);
</script></body></html>"""


def generate(max_sites: int | None, seed: int) -> int:
    backends = sorted(
        d.name for d in OUT_DIR.iterdir() if d.is_dir() and any(d.glob("*.json"))
    )
    if PIVOT not in backends:
        print(f"pivot backend '{PIVOT}' has no hypotheses", file=sys.stderr)
        return 1
    rng = random.Random(seed)
    all_sites = []
    for segment in load_manifest():
        if not segment.wav_path.exists():
            continue
        missing = [b for b in backends if not (OUT_DIR / b / f"{segment.id}.json").exists()]
        if missing:
            print(f"[skip] {segment.id}: missing hypotheses for {missing}", file=sys.stderr)
            continue
        sites = build_sites(segment.id, backends)
        total = len(sites)
        if max_sites and len(sites) > max_sites:
            sites = sorted(rng.sample(sites, max_sites), key=lambda s: s["t0"])
        for site in sites:
            site["audio"] = snippet_b64(segment.wav_path, site["t0"], site["t1"])
            rng.shuffle(site["variants"])
        all_sites.extend(sites)
        print(f"{segment.id}: {total} disagreement sites, {len(sites)} included")

    for site in all_sites:  # strip build-only fields
        for key in ("t0", "t1"):
            site.pop(key)
    page = PAGE.replace("__SITES__", json.dumps(all_sites, ensure_ascii=False))
    out = OUT_DIR / "adjudication.html"
    out.write_text(page)
    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out} ({len(all_sites)} sites, {size_mb:.1f} MB) — open it in a browser")
    return 0


def score(results_path: Path) -> int:
    results = json.loads(results_path.read_text())
    wins: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = {}
    judged = 0
    for site in results:
        if site["picked"] is None or site["picked"] < 0:
            continue
        judged += 1
        winners = site["variants"][site["picked"]]["models"]
        seg_lang = site["segment"].split("-")[0]
        for model in winners:
            wins[model] = wins.get(model, 0) + 1
            counts.setdefault(model, {}).setdefault(seg_lang, 0)
            counts[model][seg_lang] += 1
    if not judged:
        print("no judged sites in results file", file=sys.stderr)
        return 1
    print(f"{judged} sites judged (of {len(results)})\n")
    print("| Model | Correct on contested sites | de | en |")
    print("|---|---|---|---|")
    models = sorted({m for site in results for v in site["variants"] for m in v["models"]})
    for model in sorted(models, key=lambda m: -wins.get(m, 0)):
        by_lang = counts.get(model, {})
        print(
            f"| {model} | {wins.get(model, 0)}/{judged} ({wins.get(model, 0) / judged:.0%}) "
            f"| {by_lang.get('de', 0)} | {by_lang.get('en', 0)} |"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, help="score a downloaded results JSON")
    parser.add_argument("--max-sites", type=int, default=50, help="sites per segment (0 = all)")
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()
    if args.score:
        return score(args.score)
    return generate(args.max_sites or None, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
