"""
Bootstrap Confidence Intervals & Multi-Seed Aggregation for AdaTriple
=====================================================================

Reads per-(dataset, method, seed) part files written by
``run_real_experiments.py --write_parts`` and produces:

1. **Bootstrap 95% CI** for F1, AUC-PR, P, R per (dataset, method).
   Uses ``B`` resamples (default 1000) of the saved per-sample
   ``(score, label)`` arrays.  Threshold is **re-tuned on each
   bootstrap sample** so the CI captures both selection variance and
   label noise.

2. **Multi-seed mean ± std** when several seeds are available for the
   same (dataset, method).

3. **Paired bootstrap p-values** for "method A > method B" head-to-head
   on F1 (one-sided, with Bonferroni-style note).

Outputs
-------
* ``results/real/v7_ci/per_method.json`` – nested dict
  ``{dataset: {method: {metric: {mean, std, ci_low, ci_high}}}}``
* ``results/real/v7_ci/main_table.md`` – Markdown table for the paper

Usage
-----
    python src/bootstrap_ci.py \\
        --parts_dir results/real/parts \\
        --out_dir   results/real/v7_ci \\
        --bootstrap 1000

If ``--seeds`` is omitted, every seed found in the part files is used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score
except ImportError:  # graceful degradation
    average_precision_score = None  # type: ignore


# Mirror of the runner's threshold grid so CI uses the same selection
# procedure as the point estimates.  v7: extended lower bound to 0.02
# because NLI-as-base anchored scores have a lower median.
_TH_GRID = np.arange(0.02, 0.95, 0.02)


def _f1_at(scores: np.ndarray, labels: np.ndarray, th: float) -> float:
    pred = (scores > th).astype(np.int8)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _best_f1(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    best_f1, best_th = 0.0, 0.5
    for th in _TH_GRID:
        f = _f1_at(scores, labels, float(th))
        if f > best_f1:
            best_f1, best_th = f, float(th)
    return best_f1, best_th


def _metrics_at(scores: np.ndarray, labels: np.ndarray,
                th: float) -> Dict[str, float]:
    pred = (scores > th).astype(np.int8)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    if average_precision_score is not None and len(set(labels.tolist())) > 1:
        try:
            ap = float(average_precision_score(labels, scores))
        except Exception:
            ap = float("nan")
    else:
        ap = float("nan")
    return {"P": p, "R": r, "F1": f1, "AUC-PR": ap}


def _bootstrap_metrics(scores: np.ndarray, labels: np.ndarray,
                       n_boot: int = 1000,
                       seed: int = 12345) -> Dict[str, Dict[str, float]]:
    """Return mean / std / 95% CI for each metric.

    For each bootstrap sample, the F1-optimal threshold is re-tuned on
    the resampled (score, label) pairs.  This conservatively captures
    both threshold-selection variance and sampling variance.
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    out_keys = ("P", "R", "F1", "AUC-PR")
    boot: Dict[str, List[float]] = {k: [] for k in out_keys}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        s_b = scores[idx]
        y_b = labels[idx]
        if len(set(y_b.tolist())) < 2:
            # degenerate resample, skip
            continue
        _, th_b = _best_f1(s_b, y_b)
        m = _metrics_at(s_b, y_b, th_b)
        for k in out_keys:
            v = m[k]
            if not np.isnan(v):
                boot[k].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for k in out_keys:
        arr = np.array(boot[k]) if boot[k] else np.array([float("nan")])
        summary[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci_low": float(np.nanpercentile(arr, 2.5)),
            "ci_high": float(np.nanpercentile(arr, 97.5)),
            "n_boot": int(np.sum(~np.isnan(arr))),
        }
    return summary


def _paired_bootstrap_pvalue(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    n_boot: int = 1000, seed: int = 13579,
) -> float:
    """One-sided p-value for H0: F1(A) <= F1(B).

    Uses paired bootstrap on the same resample indices so threshold
    selection is consistent across methods within each draw.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = labels[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        f_a, _ = _best_f1(scores_a[idx], y_b)
        f_b, _ = _best_f1(scores_b[idx], y_b)
        diffs.append(f_a - f_b)
    if not diffs:
        return float("nan")
    diffs_arr = np.array(diffs)
    # P(diff <= 0)
    return float((diffs_arr <= 0).mean())


def _paired_bootstrap_pvalue_aucpr(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    n_boot: int = 1000, seed: int = 24680,
) -> float:
    """One-sided p-value for H0: AUC-PR(A) <= AUC-PR(B).

    Same paired-bootstrap protocol as the F1 variant, but uses
    ``average_precision_score`` so the test purely captures ranking
    quality (no threshold).  Returns NaN if sklearn is unavailable.
    """
    if average_precision_score is None:
        return float("nan")
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = labels[idx]
        if len(set(y_b.tolist())) < 2:
            continue
        try:
            ap_a = float(average_precision_score(y_b, scores_a[idx]))
            ap_b = float(average_precision_score(y_b, scores_b[idx]))
        except Exception:
            continue
        diffs.append(ap_a - ap_b)
    if not diffs:
        return float("nan")
    return float((np.array(diffs) <= 0).mean())


# ---------------------------------------------------------------------------
# Part-file discovery
# ---------------------------------------------------------------------------

# legacy seed=42:   medhallu__AdaTripleplus.json
# multi-seed:       medhallu__AdaTripleplus__seed1.json
_PART_RE = re.compile(
    r"^(?P<ds>[^_]+(?:_[^_]+)*)__"
    r"(?P<method>.+?)"
    r"(?:__seed(?P<seed>\d+))?\.json$"
)

# Normalise method-name disk encoding back to display names.
_METHOD_DECODE = {
    "AdaTripleplus": "AdaTriple+",
    "AdaTriple_w_o_KG": "AdaTriple (w/o KG)",
    "AdaTriple_w_o_NLI": "AdaTriple (w/o NLI)",
    "AdaTriple_fixed_lambda": "AdaTriple (fixed_lambda)",
    "NLI-DeBERTa": "NLI-DeBERTa",
    "SelfCheckGPT-NLI": "SelfCheckGPT-NLI",
    "HHEM": "HHEM",
    "Keyword-Match": "Keyword-Match",
    "Random": "Random",
    "Always-Positive": "Always-Positive",
}


def _decode_method(disk_name: str) -> str:
    if disk_name in _METHOD_DECODE:
        return _METHOD_DECODE[disk_name]
    # Best-effort fallback: replace _w_o_ -> ' (w/o ', etc.
    s = disk_name
    s = s.replace("plus", "+")
    s = s.replace("_w_o_", " (w/o ")
    s = s.replace("_fixed_lambda", " (fixed_lambda")
    if "(" in s and not s.endswith(")"):
        s += ")"
    return s


def discover_parts(parts_dir: Path,
                   seeds: Optional[List[int]] = None) -> Dict[
                       Tuple[str, str], Dict[int, Path]]:
    """Group part files by (dataset, method) -> {seed: path}."""
    out: Dict[Tuple[str, str], Dict[int, Path]] = defaultdict(dict)
    if not parts_dir.exists():
        return out
    for p in sorted(parts_dir.glob("*.json")):
        m = _PART_RE.match(p.name)
        if not m:
            continue
        ds = m.group("ds")
        meth = _decode_method(m.group("method"))
        seed = int(m.group("seed")) if m.group("seed") else 42
        if seeds is not None and seed not in seeds:
            continue
        out[(ds, meth)][seed] = p
    return out


def load_part(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyse(parts_dir: Path, out_dir: Path,
            n_boot: int = 1000,
            seeds: Optional[List[int]] = None,
            anchor_method: str = "AdaTriple+",
            compare_methods: Optional[List[str]] = None,
            pool_seeds: bool = True) -> dict:
    """When ``pool_seeds`` is True (default), bootstrap CIs are computed by
    concatenating per-sample ``(score, label)`` arrays across all available
    seeds and resampling that pool.  Paired p-values are likewise computed
    on a seed-aligned concatenation, which preserves the per-sample pairing
    inside each seed.  This gives a single CI / p-value that summarises
    *both* sample variance and seed variance, which is what the reviewer
    expects to see in a Q2 paper main table.

    With ``pool_seeds=False`` we fall back to the legacy behaviour of using
    only the first seed (for backward compatibility / ablation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = discover_parts(parts_dir, seeds=seeds)
    if not grouped:
        print(f"[bootstrap_ci] no part files found under {parts_dir}")
        return {}

    # Build per-(ds, method) cached score/label arrays per seed.
    payload: Dict[str, Dict[str, dict]] = defaultdict(dict)

    # First pass: load all parts, capture point metrics + per-seed arrays
    for (ds, meth), seed_paths in grouped.items():
        per_seed_metrics: Dict[int, dict] = {}
        per_seed_arrays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for seed, path in sorted(seed_paths.items()):
            data = load_part(path)
            if data is None:
                continue
            per_seed_metrics[seed] = {
                "F1": data.get("F1"),
                "P": data.get("P"),
                "R": data.get("R"),
                "AUC-PR": data.get("AUC-PR"),
                "threshold": data.get("threshold"),
            }
            scores = data.get("scores")
            labels = data.get("labels")
            if scores is not None and labels is not None:
                per_seed_arrays[seed] = (
                    np.asarray(scores, dtype=np.float64),
                    np.asarray(labels, dtype=np.int8),
                )

        if not per_seed_metrics:
            continue

        # Cross-seed mean ± std (point estimates)
        seed_summary: Dict[str, Dict[str, float]] = {}
        for k in ("F1", "P", "R", "AUC-PR"):
            vs = [m[k] for m in per_seed_metrics.values() if m.get(k) is not None]
            if vs:
                arr = np.array(vs, dtype=np.float64)
                seed_summary[k] = {
                    "seeds_mean": float(arr.mean()),
                    "seeds_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n_seeds": int(len(arr)),
                }

        # Bootstrap CI: pool across seeds if requested, else use first seed.
        boot_summary: Dict[str, Dict[str, float]] = {}
        boot_seed_used: Optional[object] = None
        if per_seed_arrays:
            if pool_seeds and len(per_seed_arrays) > 1:
                sc = np.concatenate(
                    [per_seed_arrays[s][0]
                     for s in sorted(per_seed_arrays.keys())])
                lb = np.concatenate(
                    [per_seed_arrays[s][1]
                     for s in sorted(per_seed_arrays.keys())])
                boot_seed_used = sorted(per_seed_arrays.keys())
                boot_summary = _bootstrap_metrics(sc, lb, n_boot=n_boot,
                                                  seed=12345)
            else:
                first_seed = sorted(per_seed_arrays.keys())[0]
                sc, lb = per_seed_arrays[first_seed]
                boot_seed_used = first_seed
                boot_summary = _bootstrap_metrics(
                    sc, lb, n_boot=n_boot, seed=12345 + first_seed)

        payload[ds][meth] = {
            "per_seed": per_seed_metrics,
            "seeds_summary": seed_summary,
            "bootstrap": boot_summary,
            "boot_seed_used": boot_seed_used,
            "_per_seed_arrays_keys": sorted(per_seed_arrays.keys()),
        }
        # Cache arrays for paired comparison stage (avoid re-loading)
        for s, (sc, lb) in per_seed_arrays.items():
            payload[ds][meth][f"_arrays_seed_{s}"] = (sc, lb)

    # Paired bootstrap p-values: anchor vs each compare_method on first
    # available seed where BOTH methods have per-sample arrays.
    if compare_methods is None:
        compare_methods = ["NLI-DeBERTa", "AdaTriple (w/o KG)",
                           "AdaTriple (w/o NLI)",
                           "AdaTriple (fixed_lambda)",
                           "SelfCheckGPT-NLI", "HHEM",
                           "LLM-Judge",
                           "Keyword-Match", "Random",
                           "Always-Positive"]
    paired: Dict[str, Dict[str, float]] = defaultdict(dict)
    paired_aucpr: Dict[str, Dict[str, float]] = defaultdict(dict)
    for ds, methods in payload.items():
        if anchor_method not in methods:
            continue
        anchor_entry = methods[anchor_method]
        anchor_seed_keys = anchor_entry.get("_per_seed_arrays_keys", [])
        if not anchor_seed_keys:
            continue

        for other in compare_methods:
            if other == anchor_method or other not in methods:
                continue
            other_entry = methods[other]
            other_seed_keys = set(other_entry.get("_per_seed_arrays_keys", []))
            shared = [s for s in anchor_seed_keys if s in other_seed_keys]
            if not shared:
                continue
            # Pool across all shared seeds, preserving per-seed pairing
            a_chunks_s, a_chunks_l, o_chunks_s = [], [], []
            for s in sorted(shared):
                a_sc, a_lb = anchor_entry[f"_arrays_seed_{s}"]
                o_sc, _ = other_entry[f"_arrays_seed_{s}"]
                if len(a_sc) != len(o_sc):
                    continue
                a_chunks_s.append(a_sc)
                a_chunks_l.append(a_lb)
                o_chunks_s.append(o_sc)
            if not a_chunks_s:
                continue
            a_scores = np.concatenate(a_chunks_s)
            a_labels = np.concatenate(a_chunks_l)
            o_scores = np.concatenate(o_chunks_s)
            pv = _paired_bootstrap_pvalue(a_scores, o_scores, a_labels,
                                          n_boot=n_boot)
            paired[ds][f"{anchor_method} vs {other}"] = pv
            pv_ap = _paired_bootstrap_pvalue_aucpr(
                a_scores, o_scores, a_labels, n_boot=n_boot)
            paired_aucpr[ds][f"{anchor_method} vs {other}"] = pv_ap

    # Strip numpy-array caches so the result can be JSON-serialised.
    serialisable_payload: Dict[str, Dict[str, dict]] = {}
    for ds, mdict in payload.items():
        serialisable_payload[ds] = {}
        for meth, entry in mdict.items():
            clean = {k: v for k, v in entry.items()
                     if not k.startswith("_arrays_seed_")}
            # _per_seed_arrays_keys is a list[int], keep it for traceability
            serialisable_payload[ds][meth] = clean

    final = {
        "per_method": serialisable_payload,
        "paired_pvalues": paired,
        "paired_pvalues_aucpr": paired_aucpr,
        "config": {
            "n_boot": n_boot,
            "seeds_filter": seeds,
            "anchor": anchor_method,
            "pool_seeds": pool_seeds,
        },
    }

    out_path = out_dir / "per_method.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=float)
    print(f"[bootstrap_ci] wrote {out_path}")

    write_main_table(final, out_dir / "main_table.md")
    return final


def write_main_table(result: dict, out_path: Path) -> None:
    """Markdown summary: one row per method, columns per dataset
    showing F1 mean and 95% CI."""
    payload = result.get("per_method", {})
    if not payload:
        return
    datasets = list(payload.keys())
    methods = sorted({m for ds in payload.values() for m in ds})

    lines: List[str] = []
    cfg = result.get("config", {})
    pool = cfg.get("pool_seeds", False)
    seed_note = (" (per-sample arrays pooled across all seeds before "
                 "resampling)" if pool else " (single-seed)")
    lines.append("# AdaTriple Bootstrap Confidence Intervals\n")
    lines.append(f"Bootstrap B = **{cfg.get('n_boot')}**, "
                 f"threshold re-tuned on each resample{seed_note}.\n")

    # Main F1 table
    header = "| Method | " + " | ".join(datasets) + " | Avg F1 |"
    sep = "| " + " | ".join(["---"] * (len(datasets) + 2)) + " |"
    lines.append("\n## F1 with 95% CI (bootstrap)\n")
    lines.append(header)
    lines.append(sep)
    for meth in methods:
        cells: List[str] = [meth]
        f1_vals: List[float] = []
        for ds in datasets:
            entry = payload[ds].get(meth)
            if not entry or not entry.get("bootstrap"):
                cells.append("-")
                continue
            b = entry["bootstrap"].get("F1", {})
            mean = b.get("mean")
            lo = b.get("ci_low")
            hi = b.get("ci_high")
            if mean is None:
                cells.append("-")
            else:
                f1_vals.append(mean)
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        if f1_vals:
            cells.append(f"{np.mean(f1_vals):.3f}")
        else:
            cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    # AUC-PR table
    lines.append("\n## AUC-PR with 95% CI (bootstrap)\n")
    lines.append(header)
    lines.append(sep)
    for meth in methods:
        cells = [meth]
        au_vals: List[float] = []
        for ds in datasets:
            entry = payload[ds].get(meth)
            if not entry or not entry.get("bootstrap"):
                cells.append("-")
                continue
            b = entry["bootstrap"].get("AUC-PR", {})
            mean = b.get("mean")
            lo = b.get("ci_low")
            hi = b.get("ci_high")
            if mean is None or np.isnan(mean):
                cells.append("-")
            else:
                au_vals.append(mean)
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        if au_vals:
            cells.append(f"{np.mean(au_vals):.3f}")
        else:
            cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    # Multi-seed table (if available)
    multi_seed_methods = [
        m for m in methods
        if any(payload[ds].get(m, {}).get("seeds_summary", {})
               .get("F1", {}).get("n_seeds", 0) > 1
               for ds in datasets)
    ]
    if multi_seed_methods:
        lines.append("\n## F1 across seeds: mean ± std\n")
        lines.append(header)
        lines.append(sep)
        for meth in multi_seed_methods:
            cells = [meth]
            avgs: List[float] = []
            for ds in datasets:
                entry = payload[ds].get(meth, {}).get("seeds_summary", {}).get("F1", {})
                if not entry or entry.get("n_seeds", 0) < 1:
                    cells.append("-")
                    continue
                m, s, n = entry["seeds_mean"], entry["seeds_std"], entry["n_seeds"]
                avgs.append(m)
                if n > 1:
                    cells.append(f"{m:.3f} ± {s:.3f} (n={n})")
                else:
                    cells.append(f"{m:.3f} (n=1)")
            cells.append(f"{np.mean(avgs):.3f}" if avgs else "-")
            lines.append("| " + " | ".join(cells) + " |")

    # Paired p-values (F1)
    paired = result.get("paired_pvalues", {})
    if paired:
        lines.append("\n## Paired bootstrap one-sided p-values (F1)\n")
        lines.append("H0: AdaTriple+ ≤ baseline.  p < 0.05 = significant win.\n")
        for ds, comps in paired.items():
            lines.append(f"\n### {ds}\n")
            lines.append("| Comparison | p-value |")
            lines.append("| --- | --- |")
            for k, v in comps.items():
                marker = " *" if v < 0.05 else ""
                lines.append(f"| {k} | {v:.3f}{marker} |")

    # Paired p-values (AUC-PR)
    paired_ap = result.get("paired_pvalues_aucpr", {})
    if paired_ap:
        lines.append("\n## Paired bootstrap one-sided p-values (AUC-PR)\n")
        lines.append(
            "H0: AdaTriple+ ≤ baseline.  p < 0.05 = AdaTriple+ significantly "
            "better-ranked.  Note: large p (e.g. > 0.95) on the "
            "AdaTriple+ vs NLI-DeBERTa row would mean NLI-DeBERTa is "
            "*significantly better* (one-sided); this is the strongest "
            "test of the AUC-PR trade-off.\n")
        for ds, comps in paired_ap.items():
            lines.append(f"\n### {ds}\n")
            lines.append("| Comparison | p-value |")
            lines.append("| --- | --- |")
            for k, v in comps.items():
                if np.isnan(v):
                    cell = "nan"
                else:
                    win_marker = " *" if v < 0.05 else ""
                    loss_marker = " (loss)" if v > 0.95 else ""
                    cell = f"{v:.3f}{win_marker}{loss_marker}"
                lines.append(f"| {k} | {cell} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[bootstrap_ci] wrote {out_path}")


def _parse_seeds(s: str) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Bootstrap CI + multi-seed aggregation for AdaTriple")
    p.add_argument("--parts_dir", required=True,
                   help="Directory of per-(ds, method, seed) part files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seeds", default="",
                   help="Comma-separated seeds to include (default: all)")
    p.add_argument("--anchor", default="AdaTriple+",
                   help="Anchor method for paired comparisons")
    p.add_argument("--no_pool_seeds", action="store_true",
                   help="Disable seed-pooling for bootstrap CI / paired "
                        "p-values (use only the first available seed).")
    args = p.parse_args()
    analyse(Path(args.parts_dir), Path(args.out_dir),
            n_boot=args.bootstrap, seeds=_parse_seeds(args.seeds),
            anchor_method=args.anchor,
            pool_seeds=not args.no_pool_seeds)


if __name__ == "__main__":
    main()
