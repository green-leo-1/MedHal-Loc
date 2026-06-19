"""Pairwise agreement among LLM annotators for the MedHal-Loc pilot.

Reads annotation/llm_annot_*.json (each = list of
{id, spans:[{text,type}], no_locatable_span}) produced by independent LLM
annotators, plus annotation/pilot_template.jsonl, and reports pairwise +
mean span-F1 and type Cohen's kappa (reusing compute_agreement's robust
span-locating logic).

Run:  python src/_llm_agreement.py
"""
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_agreement import covered_tokens  # noqa: E402

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

WORKSPACE = Path(__file__).resolve().parent.parent
ANN = WORKSPACE / "annotation"


def load_annot(path):
    """-> {id: [(text, type), ...]}"""
    d = json.load(open(path, encoding="utf-8"))
    items = d.get("annotations", d) if isinstance(d, dict) else d
    out = {}
    for r in items:
        spans = [(s.get("text", ""), (s.get("type", "") or "").lower())
                 for s in (r.get("spans") or []) if s.get("text")]
        out[r["id"]] = spans
    return out


def pair_agreement(text, A, B):
    f1s, tA, tB = [], [], []
    for i in text:
        if i not in A or i not in B:
            continue
        ca, ta, _ = covered_tokens(text[i], A[i])
        cb, tb, _ = covered_tokens(text[i], B[i])
        if not ca and not cb:
            f1s.append(1.0)
        elif not ca or not cb:
            f1s.append(0.0)
        else:
            inter = len(ca & cb)
            p = inter / len(cb); r = inter / len(ca)
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        for ti in (ca & cb):
            tA.append(ta.get(ti, "?")); tB.append(tb.get(ti, "?"))
    k = (cohen_kappa_score(tA, tB)
         if (cohen_kappa_score and tA and len(set(tA + tB)) > 1)
         else float("nan"))
    return float(np.mean(f1s)) if f1s else float("nan"), k, len(tA)


def main():
    text = {}
    for line in open(ANN / "pilot_template.jsonl", encoding="utf-8"):
        r = json.loads(line)
        text[r["id"]] = r["hallucinated_answer"]

    files = sorted(glob.glob(str(ANN / "llm_annot_*.json")))
    if len(files) < 2:
        print(f"Need >=2 llm_annot_*.json files, found {len(files)}: {files}")
        return
    annots = {Path(f).stem.replace("llm_annot_", ""): load_annot(f)
              for f in files}
    print("=" * 60)
    print("MedHal-Loc pilot: inter-LLM-annotator agreement")
    print("=" * 60)
    print(f"annotators: {list(annots)}  | items: {len(text)}")
    print("-" * 60)
    f1s, ks = [], []
    for x, y in combinations(annots, 2):
        f1, k, n = pair_agreement(text, annots[x], annots[y])
        f1s.append(f1)
        if k == k:
            ks.append(k)
        kshow = f"{k:.3f}" if k == k else "n/a"
        print(f"  {x} vs {y}: span-F1={f1:.3f}   type-kappa={kshow}  "
              f"(co-marked tokens={n})")
    print("-" * 60)
    mf1 = np.nanmean(f1s) if f1s else float("nan")
    mk = np.mean(ks) if ks else float("nan")
    print(f"MEAN span-F1 : {mf1:.3f}   [viable >= 0.50]")
    print(f"MEAN kappa   : {mk:.3f}   [viable >= 0.60]")
    verdict = ("LLMs AGREE -> auto-annotate + human verify a subset is viable"
               if (mf1 >= 0.5 and mk >= 0.6)
               else "LLMs DISAGREE -> localization is subjective; needs human "
                    "adjudication or coarser granularity")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
