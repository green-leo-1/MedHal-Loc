"""
Computational Efficiency Table for AdaTriple
============================================

Reads ``time_s`` from per-(dataset, method, seed) part files in
``results/real/parts/`` and aggregates mean inference latency per method
on 1{,}000-sample runs.  Combines with hand-curated model parameter
counts and approximate VRAM footprints to produce a LaTeX table.

Notes on numbers
----------------
* ``time_s`` in each part file is the wall-clock time for a single
  subprocess running 1{,}000 samples on the dataset.  It includes model
  load (one-time cost amortised over 1{,}000 samples) plus inference.
* For methods we time ourselves on the same RTX 4090 49 GB rig, the
  numbers are directly comparable.  ``Always-Positive`` etc. are
  effectively zero-cost.

Usage
-----
    python src/efficiency_table.py \
        --parts_dir results/real/parts \
        --out_dir   results/real/v8_ci_with_llmjudge
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# Hand-curated model footprints. Numbers are *approximate* and primarily
# meant for relative cost comparison, not benchmark-grade efficiency.
# Sources: HF model cards (parameter counts) and our run logs (VRAM peak
# measured during a representative run on the same RTX 4090 rig).
METHOD_INFO = {
    # display_name -> (param_count_str, vram_str, backbone_str, granularity)
    "AdaTriple+ (Full)": (
        "0.3 B",
        "~3.5 GB",
        "DeBERTa-v3-base + SapBERT + Hetionet",
        "Triple ($e_h, r, e_t$) + sentence + response",
    ),
    "NLI-DeBERTa": (
        "0.18 B",
        "~1.5 GB",
        "DeBERTa-v3-base-MNLI",
        "Sentence-pair entailment",
    ),
    "HHEM": (
        "0.25 B",
        "~2.0 GB",
        "HHEM-2.1-Open (Flan-T5-base)",
        "Sentence-pair consistency",
    ),
    "SelfCheckGPT-NLI": (
        "0.18 B",
        "~1.5 GB",
        "DeBERTa-v3-base-MNLI",
        "Self-consistency over $n{=}5$ samples",
    ),
    "LLM-Judge (Qwen2.5-7B)": (
        "7.6 B",
        "~14 GB",
        "Qwen2.5-7B-Instruct (logit-only)",
        "Response-level Yes/No",
    ),
    "Keyword-Match": (
        "0",
        "0 GB",
        "Token-overlap",
        "Response-level overlap",
    ),
    "Random": (
        "0",
        "0 GB",
        "Uniform $[0,1]$",
        "--",
    ),
    "Always-Positive": (
        "0",
        "0 GB",
        "Constant 1",
        "--",
    ),
}


# Map between part-file method names and display names in this table.
PART_TO_DISPLAY = {
    "AdaTriple+": "AdaTriple+ (Full)",
    "NLI-DeBERTa": "NLI-DeBERTa",
    "HHEM": "HHEM",
    "SelfCheckGPT-NLI": "SelfCheckGPT-NLI",
    "LLM-Judge": "LLM-Judge (Qwen2.5-7B)",
    "Keyword-Match": "Keyword-Match",
    "Random": "Random",
    "Always-Positive": "Always-Positive",
}


def _decode_method_name(stem: str) -> str:
    """Reverse the safe-name transforms applied in ExperimentRunner."""
    # stem like "medhallu__LLM-Judge__seed1" or "medhallu__LLM-Judge"
    parts = stem.split("__")
    if len(parts) < 2:
        return ""
    name = parts[1]
    return name.replace("plus", "+").replace("_", " ")


def collect_times(parts_dir: Path) -> Dict[str, List[float]]:
    """Return {method_display_name: [time_s, ...]}"""
    by_method: Dict[str, List[float]] = defaultdict(list)
    for f in parts_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("_failed"):
            continue
        # Prefer the persisted method name; fall back to filename decoding.
        method_raw = data.get("_method") or _decode_method_name(f.stem)
        display = PART_TO_DISPLAY.get(method_raw)
        if not display:
            # Try a looser match: method names containing the key
            for key, disp in PART_TO_DISPLAY.items():
                if method_raw.startswith(key):
                    display = disp
                    break
        if not display:
            continue
        t = data.get("time_s")
        if t is None or not isinstance(t, (int, float)):
            continue
        by_method[display].append(float(t))
    return dict(by_method)


def latex_table(times: Dict[str, List[float]]) -> str:
    # Order rows: proposed first, then strong baselines, then trivial.
    order = [
        "AdaTriple+ (Full)",
        "NLI-DeBERTa",
        "HHEM",
        "SelfCheckGPT-NLI",
        "LLM-Judge (Qwen2.5-7B)",
        "Keyword-Match",
        "Random",
        "Always-Positive",
    ]

    rows = []
    for m in order:
        info = METHOD_INFO.get(m, ("--", "--", "--", "--"))
        params, vram, backbone, gran = info
        ts = times.get(m, [])
        if ts:
            mean_s = sum(ts) / len(ts)
            n = len(ts)
            sps = 1000.0 / mean_s if mean_s > 0 else 0.0
            wall = f"{mean_s:.1f}\\,s"
            thr = f"{sps:.1f}"
        else:
            wall, thr = "--", "--"
            n = 0
        rows.append(
            f"{m} & {params} & {vram} & {wall} & {thr} & {gran} \\\\")

    lines = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\caption{Computational efficiency of all evaluated methods on a "
        "single RTX 4090 (49 GB VRAM).  \\textbf{Wall-clock} is the mean "
        "end-to-end time for a 1{,}000-sample subprocess (model load $+$ "
        "inference), averaged over the 5 datasets $\\times$ 3 seeds in our "
        "main experiment.  \\textbf{Throughput} is the corresponding mean "
        "samples/sec, derived from the per-job wall-clock.  Parameter and "
        "VRAM figures are approximate and reflect the inference-only "
        "configuration we benchmark.  AdaTriple+ delivers fine-grained "
        "(triple-level) decisions at $0.3$ B parameters, $25\\times$ smaller "
        "than the LLM-Judge baseline, while preserving sentence- and "
        "response-level scores as a by-product of the same forward pass.}",
        "\\label{tab:efficiency}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llllrl@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Params} & \\textbf{VRAM} & "
        "\\textbf{Wall-clock} & \\textbf{Thr.\\ (sps)} & "
        "\\textbf{Output granularity} \\\\",
        "\\midrule",
    ]
    lines.extend(rows)
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def md_table(times: Dict[str, List[float]]) -> str:
    order = [
        "AdaTriple+ (Full)",
        "NLI-DeBERTa",
        "HHEM",
        "SelfCheckGPT-NLI",
        "LLM-Judge (Qwen2.5-7B)",
        "Keyword-Match",
        "Random",
        "Always-Positive",
    ]
    lines = [
        "# Computational efficiency on RTX 4090 49 GB",
        "",
        "| Method | Params | VRAM | Wall-clock 1k samples | Thr. (sps) | Granularity |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for m in order:
        info = METHOD_INFO.get(m, ("--", "--", "--", "--"))
        params, vram, backbone, gran = info
        ts = times.get(m, [])
        if ts:
            mean_s = sum(ts) / len(ts)
            n = len(ts)
            sps = 1000.0 / mean_s if mean_s > 0 else 0.0
            wall = f"{mean_s:.1f}s (n={n})"
            thr = f"{sps:.1f}"
        else:
            wall, thr = "--", "--"
        lines.append(
            f"| {m} | {params} | {vram} | {wall} | {thr} | {gran} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = collect_times(parts_dir)
    if not times:
        raise SystemExit(f"No part files with time_s found in {parts_dir}")

    md = md_table(times)
    (out_dir / "efficiency.md").write_text(md, encoding="utf-8")

    tex = latex_table(times)
    (out_dir / "efficiency_table.tex").write_text(tex, encoding="utf-8")

    print("[efficiency] sampled methods:")
    for m, ts in sorted(times.items()):
        print(f"  {m:30s}  n={len(ts):2d}  mean={sum(ts)/len(ts):6.1f}s")
    print(f"[efficiency] wrote {out_dir / 'efficiency.md'}")
    print(f"[efficiency] wrote {out_dir / 'efficiency_table.tex'}")


if __name__ == "__main__":
    main()
