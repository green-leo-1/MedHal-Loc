"""Compute the 50-item human-verification outcomes (Route C credibility number).

Reads annotation/verify_50_H1.csv (+ optional _H2.csv) and reports:
  * controlled QC pass-rate  : % injected gold judged a clean, locatable error
  * natural LLM-human span precision + type agreement (the headline credibility
    number: "LLM-annotated gold matches human judgement X% of the time")
  * human-human agreement (Cohen's kappa) if both annotators provided

Run:  python src/compute_validation.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

WORKSPACE = Path(__file__).resolve().parent.parent
ANN = WORKSPACE / "annotation"
SLOTS = [("llm_span_1", "span1_ok(Y/N/P)", "type1_ok(Y/N)"),
         ("llm_span_2", "span2_ok(Y/N/P)", "type2_ok(Y/N)")]

# Chinese header / value -> canonical, so the CN-translated verify file parses.
ALIAS = {
    "编号": "id", "子集": "subset",
    "LLM标注_错误片段1(英文)": "llm_span_1", "LLM标注_类型1": "llm_type_1",
    "片段1对吗(Y对/P部分/N错)": "span1_ok(Y/N/P)", "类型1对吗(Y/N)": "type1_ok(Y/N)",
    "LLM标注_错误片段2(英文)": "llm_span_2", "LLM标注_类型2": "llm_type_2",
    "片段2对吗(Y对/P部分/N错)": "span2_ok(Y/N/P)", "类型2对吗(Y/N)": "type2_ok(Y/N)",
    "漏标的错误-逐字复制英文": "missed_error", "漏标类型": "missed_type",
    "备注": "notes",
}
SUBSET_CN = {"受控": "controlled", "自然": "natural"}


def load(path):
    import io
    text = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):  # Excel often re-saves as GBK
        try:
            text = open(path, encoding=enc, newline="").read()
            break
        except Exception:
            continue
    if text is None:
        raise RuntimeError(f"cannot decode {path}")
    rows = {}
    for r in csv.DictReader(io.StringIO(text)):
        rc = {ALIAS.get(k, k): v for k, v in r.items()}
        cid = (rc.get("id") or "").strip()
        # subset by id prefix (robust to a corrupted/translated 子集 cell)
        if cid.startswith("inj_"):
            rc["subset"] = "controlled"
        elif cid.startswith("mh_"):
            rc["subset"] = "natural"
        else:
            sub = (rc.get("subset") or "").strip()
            rc["subset"] = SUBSET_CN.get(sub, sub)
        rows[cid] = rc
    return rows


def verdict(s):
    s = (s or "").strip().upper()
    return s if s in ("Y", "N", "P") else ""


def summarize(rows, name):
    ctrl_ok, ctrl_type = [], []
    nat_prec, nat_prec_yp, nat_type, nat_missed = [], [], [], []
    for r in rows.values():
        sub = r.get("subset", "").strip()
        spans = []
        for span_c, ok_c, type_c in SLOTS:
            if (r.get(span_c) or "").strip():
                spans.append((verdict(r.get(ok_c)), verdict(r.get(type_c))))
        if sub == "controlled":
            if spans:
                v, t = spans[0]
                if v:
                    ctrl_ok.append(1 if v == "Y" else 0)
                if t:
                    ctrl_type.append(1 if t == "Y" else 0)
        elif sub == "natural":
            for v, t in spans:
                if v:
                    nat_prec.append(1 if v == "Y" else 0)
                    nat_prec_yp.append(1 if v in ("Y", "P") else 0)
                if t:
                    nat_type.append(1 if t == "Y" else 0)
            nat_missed.append(1 if (r.get("missed_error") or "").strip() else 0)

    def pct(x):
        return 100.0 * np.mean(x) if x else float("nan")

    print(f"\n--- {name} ---")
    print(f"CONTROLLED QC: injected gold judged clean/correct (span Y): "
          f"{pct(ctrl_ok):.1f}%  (n={len(ctrl_ok)})   [gate >= 90%]")
    print(f"  controlled type correct: {pct(ctrl_type):.1f}%")
    print(f"NATURAL LLM-human span precision (Y):       {pct(nat_prec):.1f}%  "
          f"(n={len(nat_prec)} spans)   [gate >= 80%]")
    print(f"  natural precision (Y or partial):         {pct(nat_prec_yp):.1f}%")
    print(f"  natural type agreement (among judged):    {pct(nat_type):.1f}%")
    print(f"  natural items where LLM missed an error:  {pct(nat_missed):.1f}%")
    return dict(ctrl_ok=pct(ctrl_ok), nat_prec=pct(nat_prec))


def hh_agreement(h1, h2):
    a, b = [], []
    for i in set(h1) & set(h2):
        for _, ok_c, _ in SLOTS:
            va, vb = verdict(h1[i].get(ok_c)), verdict(h2[i].get(ok_c))
            if va and vb:
                a.append(va); b.append(vb)
    if not a:
        return
    agree = 100.0 * np.mean([1 if x == y else 0 for x, y in zip(a, b)])
    print(f"\n--- human-human ---")
    print(f"verdict agreement: {agree:.1f}%  (on {len(a)} co-judged spans)")
    if cohen_kappa_score and len(set(a + b)) > 1:
        print(f"Cohen's kappa    : {cohen_kappa_score(a, b):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", default=str(ANN / "verify_natural_18.csv"))
    ap.add_argument("--h2", default=str(ANN / "verify_natural_18_H2.csv"))
    args = ap.parse_args()

    p1, p2 = Path(args.h1), Path(args.h2)
    if not p1.exists() and not p2.exists():
        fallback = ANN / "verify_50.csv"
        print(f"No verify_50_H1.csv / _H2.csv found. Fill them (copy from "
              f"{fallback}) and re-run.")
        return
    print("=" * 60)
    print("MedHal-Loc 50-item human verification")
    print("=" * 60)
    res = []
    h1 = h2 = None
    if p1.exists():
        h1 = load(p1); res.append(summarize(h1, "annotator H1"))
    if p2.exists():
        h2 = load(p2); res.append(summarize(h2, "annotator H2"))
    if h1 and h2:
        hh_agreement(h1, h2)
        print("\n--- POOLED headline (mean of H1,H2) ---")
        print(f"controlled QC pass : "
              f"{np.nanmean([r['ctrl_ok'] for r in res]):.1f}%")
        print(f"natural LLM-human  : "
              f"{np.nanmean([r['nat_prec'] for r in res]):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
