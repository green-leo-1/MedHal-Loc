"""Inter-annotator agreement for the MedHal-Loc pilot (Route C, step 1 gate).

Reads annotation/pilot_A.csv and pilot_B.csv (filled by two annotators) plus
annotation/pilot_template.jsonl (for the hallucinated_answer text), and reports:

  * span-F1   : token-level localization agreement (do A and B mark the same
                error region?), averaged over samples. Both-empty -> 1.0.
  * type kappa: Cohen's kappa on the error TYPE of tokens both marked as error
                (do they agree on WHAT kind of error it is?).
  * presence  : per-sample agreement on "has a locatable error span at all".

Decision gate:  kappa >= 0.6 AND span-F1 >= 0.5  -> schema is viable.

Run:  python src/compute_agreement.py
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

WORKSPACE = Path(__file__).resolve().parent.parent
ANN = WORKSPACE / "annotation"
SPAN_SLOTS = [("span1_text", "span1_type"), ("span2_text", "span2_type"),
              ("span3_text", "span3_type")]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def tokenize(text):
    """Return list of (start, end) char spans for word tokens."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def load_csv(path):
    """id -> list of (span_text, type)."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            spans = []
            for tcol, ycol in SPAN_SLOTS:
                t = (row.get(tcol) or "").strip()
                y = (row.get(ycol) or "").strip().lower()
                if t:
                    spans.append((t, y))
            out[row["id"].strip()] = spans
    return out


def covered_tokens(answer, spans):
    """Return (set of token-indices covered by any span, {tok_idx: type})."""
    tokpos = tokenize(answer)
    na = norm(answer)
    # map normalized-answer char index back is messy; instead locate each span
    # in the ORIGINAL answer via normalized matching on a lowercased copy.
    low = answer.lower()
    covered, typ = set(), {}
    unlocated = 0
    for text, t in spans:
        cs = low.find(text.lower())
        if cs < 0:  # try whitespace-normalized fallback
            nt = norm(text)
            # build a regex that tolerates variable whitespace
            pat = re.escape(nt).replace(r"\ ", r"\s+")
            m = re.search(pat, low)
            cs = m.start() if m else -1
            ce = m.end() if m else -1
        else:
            ce = cs + len(text)
        if cs < 0:
            unlocated += 1
            continue
        for ti, (a, b) in enumerate(tokpos):
            if a < ce and b > cs:
                covered.add(ti)
                typ.setdefault(ti, t)
    return covered, typ, unlocated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ANN / "pilot_A.csv"))
    ap.add_argument("--b", default=str(ANN / "pilot_B.csv"))
    ap.add_argument("--template", default=str(ANN / "pilot_template.jsonl"))
    args = ap.parse_args()

    text = {}
    with open(args.template, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text[r["id"]] = r["hallucinated_answer"]

    A, B = load_csv(args.a), load_csv(args.b)
    ids = [i for i in text if i in A and i in B]
    if not ids:
        print("No overlapping annotated ids found. Did both A and B fill the "
              "same pilot file ids?")
        return

    span_f1s, presence_match = [], []
    typeA, typeB = [], []
    n_unloc = 0
    rows = []

    for i in ids:
        ans = text[i]
        ca, ta, ua = covered_tokens(ans, A[i])
        cb, tb, ub = covered_tokens(ans, B[i])
        n_unloc += ua + ub

        if not ca and not cb:
            f1 = 1.0  # both said "no locatable span"
        elif not ca or not cb:
            f1 = 0.0
        else:
            inter = len(ca & cb)
            p = inter / len(cb) if cb else 0.0   # treat B as reference
            r = inter / len(ca) if ca else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        span_f1s.append(f1)
        presence_match.append(1 if (bool(ca) == bool(cb)) else 0)

        for ti in (ca & cb):
            typeA.append(ta.get(ti, "?"))
            typeB.append(tb.get(ti, "?"))

        rows.append((i, len(A[i]), len(B[i]), round(f1, 2)))

    print("=" * 60)
    print("MedHal-Loc pilot inter-annotator agreement")
    print("=" * 60)
    print(f"samples compared          : {len(ids)}")
    print(f"unlocatable spans (typos) : {n_unloc}  "
          "(copy spans EXACTLY if >0)")
    print(f"presence agreement        : {100*np.mean(presence_match):.1f}%  "
          "(both found a span, or both none)")
    print(f"mean span-F1 (localization): {np.mean(span_f1s):.3f}   "
          "[gate >= 0.50]")
    if cohen_kappa_score and len(set(typeA + typeB)) > 1 and typeA:
        k = cohen_kappa_score(typeA, typeB)
        print(f"type Cohen's kappa        : {k:.3f}   [gate >= 0.60]  "
              f"(on {len(typeA)} co-marked tokens)")
    else:
        print(f"type Cohen's kappa        : n/a  "
              f"(co-marked tokens={len(typeA)}, distinct labels="
              f"{len(set(typeA + typeB))})")
    print("-" * 60)
    span_f1 = float(np.mean(span_f1s))
    kap = (cohen_kappa_score(typeA, typeB)
           if (cohen_kappa_score and typeA and len(set(typeA + typeB)) > 1)
           else float("nan"))
    verdict = ("VIABLE -> proceed to full annotation"
               if (span_f1 >= 0.5 and (kap >= 0.6 if kap == kap else False))
               else "NOT YET -> reconcile disagreements / tighten the guide")
    print(f"DECISION: {verdict}")
    print("-" * 60)
    print(f"{'id':10s} {'nA':>3s} {'nB':>3s} {'spanF1':>7s}")
    for i, na, nb, f1 in sorted(rows, key=lambda x: x[3]):
        print(f"{i:10s} {na:3d} {nb:3d} {f1:7.2f}")


if __name__ == "__main__":
    main()
