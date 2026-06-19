"""
Holm-Bonferroni Multiple-Testing Correction for AdaTriple Paired p-Values
=========================================================================

Reads ``results/real/v8_ci_with_llmjudge/per_method.json`` and applies the
Holm-Bonferroni step-down procedure separately to each metric family
(F1, AUC-PR), reporting which paired comparisons remain significant
after controlling family-wise error rate (FWER) at alpha=0.05.

Step-down rule
--------------
Sort the m p-values ascending: p_(1) <= p_(2) <= ... <= p_(m).
Reject H_(k) iff for every i <= k, p_(i) <= alpha / (m - i + 1).

Outputs
-------
* ``results/real/v8_ci_with_llmjudge/holm_corrected.md`` -- per-family
  ranked table with raw p, threshold, and reject/retain verdict.
* ``results/real/v8_ci_with_llmjudge/holm_table.tex`` -- LaTeX table
  ready to drop into the paper.

Usage
-----
    python src/holm_correction.py \
        --in_json results/real/v8_ci_with_llmjudge/per_method.json \
        --out_dir results/real/v8_ci_with_llmjudge \
        --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Datasets we want to display (canonical order in the paper)
DS_ORDER = ["medhallu", "pubmedqa", "medqa", "scifact", "mmlu_medical"]
DS_LABEL = {
    "medhallu": "MedHallu",
    "pubmedqa": "PubMedQA",
    "medqa": "MedQA",
    "scifact": "SciFact",
    "mmlu_medical": "MMLU-Med",
}

# Only correct over comparisons against external (non-ablation) baselines.
# Including ablations would inflate m without a clean FWER interpretation.
EXTERNAL_BASELINES = [
    "NLI-DeBERTa",
    "HHEM",
    "SelfCheckGPT-NLI",
    "LLM-Judge",
    "Always-Positive",
    "Random",
    "Keyword-Match",
]


def _collect(payload: Dict, key: str
             ) -> List[Tuple[str, str, float]]:
    """Return list of (dataset, comparison_label, p_value) tuples for the
    requested paired-pvalue family ('paired_pvalues' or
    'paired_pvalues_aucpr')."""
    out: List[Tuple[str, str, float]] = []
    for ds in DS_ORDER:
        ds_block = payload.get(ds, {})
        # Find any method that has the paired dict
        # (the bootstrap_ci writer emits identical paired blocks under
        #  every method-level dict; we just need one).
        # Actually paired_pvalues is at the TOP level of per_method.json
        # under the "paired_pvalues" key, indexed by ds.
        pass
    # The structure produced by bootstrap_ci.compute() is:
    #   payload["paired_pvalues"][ds]["AdaTriple+ vs <baseline>"] = p
    pp = payload.get(key, {})
    for ds, comps in pp.items():
        if ds not in DS_ORDER:
            continue
        for label, p in comps.items():
            # Strip the "AdaTriple+ vs " prefix
            base = label.replace("AdaTriple+ vs ", "").strip()
            if base not in EXTERNAL_BASELINES:
                continue
            try:
                out.append((ds, base, float(p)))
            except (TypeError, ValueError):
                continue
    return out


def _holm(pairs: List[Tuple[str, str, float]],
          alpha: float = 0.05
          ) -> List[Tuple[str, str, float, float, bool]]:
    """Apply Holm-Bonferroni step-down. Returns list ordered by ascending p,
    each entry is (ds, baseline, p, threshold, reject)."""
    sorted_pairs = sorted(pairs, key=lambda t: t[2])
    m = len(sorted_pairs)
    out = []
    rejected_so_far = True
    for i, (ds, base, p) in enumerate(sorted_pairs):
        thr = alpha / (m - i)  # m - i + 1 with i 1-indexed -> m - i with 0-indexed
        if rejected_so_far and p <= thr:
            reject = True
        else:
            reject = False
            rejected_so_far = False
        out.append((ds, base, p, thr, reject))
    return out


def _md_table(corrected: List[Tuple[str, str, float, float, bool]],
              title: str, alpha: float) -> str:
    m = len(corrected)
    n_reject = sum(1 for *_, r in corrected if r)
    lines = [f"## {title}", ""]
    lines.append(f"Family size m = {m}.  alpha = {alpha}.  "
                 f"**{n_reject}/{m}** comparisons survive Holm-Bonferroni.")
    lines.append("")
    lines.append("| Rank | Dataset | Baseline | Raw p | Holm threshold | Survive? |")
    lines.append("| ---:| --- | --- | ---:| ---:| :---:|")
    for i, (ds, base, p, thr, rej) in enumerate(corrected, 1):
        ds_lbl = DS_LABEL.get(ds, ds)
        marker = "✓" if rej else "—"
        lines.append(
            f"| {i} | {ds_lbl} | {base} | {p:.4f} | {thr:.5f} | {marker} |")
    lines.append("")
    return "\n".join(lines)


def _tex_table(f1_corrected: List, ap_corrected: List, alpha: float) -> str:
    """Compact LaTeX table summarising Holm survival per (metric, ds, baseline)."""
    # Build a lookup: (metric, ds, baseline) -> reject?
    survive = {}
    for ds, base, p, thr, rej in f1_corrected:
        survive[("F1", ds, base)] = rej
    for ds, base, p, thr, rej in ap_corrected:
        survive[("AP", ds, base)] = rej

    cols = "l" + "c" * len(EXTERNAL_BASELINES)
    head = ["\\textbf{Dataset}"] + [
        "\\textbf{" + b.replace("_", "\\_") + "}" for b in EXTERNAL_BASELINES]

    def row(metric_label: str, marker_dict, ds: str) -> str:
        cells = [DS_LABEL.get(ds, ds)]
        for b in EXTERNAL_BASELINES:
            v = marker_dict.get((metric_label, ds, b), None)
            if v is None:
                cells.append("--")
            elif v:
                cells.append("\\textbf{$\\checkmark$}")
            else:
                cells.append("$\\circ$")
        return " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\caption{{Holm--Bonferroni step-down survival of paired one-sided "
        f"bootstrap $p$-values at $\\alpha={alpha}$, family-wise error rate "
        f"controlled separately within F1 (top, $m={len(f1_corrected)}$ "
        f"hypotheses) and AUC-PR (bottom, $m={len(ap_corrected)}$ "
        f"hypotheses). \\textbf{{$\\checkmark$}}: AdaTriple+ remains "
        f"significantly better than the baseline after correction; "
        f"$\\circ$: not significant after correction. Comparisons against "
        f"AdaTriple ablations are excluded from the family count to keep "
        f"the FWER interpretation clean.}}",
        "\\label{tab:holm}",
        f"\\resizebox{{\\textwidth}}{{!}}{{%",
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        "\\toprule",
        " & ".join(head) + " \\\\",
        "\\midrule",
        "\\multicolumn{8}{l}{\\textit{F1 paired bootstrap}} \\\\",
    ]
    for ds in DS_ORDER:
        lines.append(row("F1", survive, ds))
    lines.append("\\midrule")
    lines.append("\\multicolumn{8}{l}{\\textit{AUC-PR paired bootstrap}} \\\\")
    for ds in DS_ORDER:
        lines.append(row("AP", survive, ds))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Holm-Bonferroni correction over paired bootstrap p-values")
    ap.add_argument("--in_json", required=True,
                    help="per_method.json from bootstrap_ci.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    payload = json.loads(Path(args.in_json).read_text(encoding="utf-8"))

    # Defensively detect the paired-pvalue dicts.
    f1 = _collect(payload, "paired_pvalues")
    ap_ = _collect(payload, "paired_pvalues_aucpr")
    if not f1:
        raise SystemExit(
            "No 'paired_pvalues' found in input json - "
            "did you run bootstrap_ci.py with the LLM-Judge baseline?")

    f1_corr = _holm(f1, alpha=args.alpha)
    ap_corr = _holm(ap_, alpha=args.alpha) if ap_ else []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = "# Holm-Bonferroni multiple-testing correction\n\n"
    md += (f"Family-wise error rate controlled at "
           f"alpha = {args.alpha} separately within each metric.  "
           f"Step-down rule: reject H_(i) iff "
           f"p_(i) <= alpha / (m - i + 1) for all earlier i in sorted order.  "
           f"Hypotheses are 'AdaTriple+ > baseline' for one-sided paired "
           f"bootstrap; ablation comparisons are excluded from the family.\n\n")
    md += _md_table(f1_corr, "F1 family", args.alpha) + "\n\n"
    if ap_corr:
        md += _md_table(ap_corr, "AUC-PR family", args.alpha) + "\n\n"
    (out_dir / "holm_corrected.md").write_text(md, encoding="utf-8")

    tex = _tex_table(f1_corr, ap_corr, args.alpha)
    (out_dir / "holm_table.tex").write_text(tex, encoding="utf-8")

    n_f1_rej = sum(1 for *_, r in f1_corr if r)
    n_ap_rej = sum(1 for *_, r in ap_corr if r)
    print(f"[holm] F1: {n_f1_rej}/{len(f1_corr)} survive at alpha={args.alpha}")
    print(f"[holm] AUC-PR: {n_ap_rej}/{len(ap_corr)} survive at alpha={args.alpha}")
    print(f"[holm] wrote {out_dir / 'holm_corrected.md'}")
    print(f"[holm] wrote {out_dir / 'holm_table.tex'}")


if __name__ == "__main__":
    main()
