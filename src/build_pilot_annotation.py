"""Build the MedHal-Loc pilot annotation set (Route C, step 1).

Samples ~40 diverse MedHallu hallucinated answers (stratified over category x
difficulty, over-sampling rare categories so all error types appear), and emits:

  annotation/pilot_template.jsonl  -- machine-readable, full evidence, for the
                                      eventual faithfulness pipeline
  annotation/pilot_A.csv           -- annotator A template (Excel, utf-8-sig)
  annotation/pilot_B.csv           -- annotator B template (identical content)

Annotators copy the EXACT error substring(s) from `hallucinated_answer` into the
spanN_text columns and pick spanN_type; see annotation/ANNOTATION_GUIDE.md.

Run:  python src/build_pilot_annotation.py --n 40
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
OUT = WORKSPACE / "annotation"
SEED = 42

# Over-sample rare categories so the pilot exercises every error type.
CATEGORY_TARGET = {
    "Misinterpretation of #Question#": 16,
    "Incomplete Information": 12,
    "Mechanism and Pathway Misattribution": 9,
    "Methodological and Evidence Fabrication": 3,
}

CONTEXT_COLS = ["id", "difficulty", "category", "question",
                "evidence", "ground_truth", "hallucinated_answer"]
FILL_COLS = ["span1_text", "span1_type", "span2_text", "span2_type",
             "span3_text", "span3_type", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--evidence_cap", type=int, default=1500)
    args = ap.parse_args()

    data = json.load(open(WORKSPACE / "data" / "medhallu_pqa_labeled.json",
                          encoding="utf-8"))
    by_cat = defaultdict(list)
    for i, it in enumerate(data):
        by_cat[it.get("category", "?")].append((i, it))

    rng = np.random.RandomState(SEED)
    picked = []
    for cat, target in CATEGORY_TARGET.items():
        items = by_cat.get(cat, [])
        # stratify by difficulty within the category
        by_diff = defaultdict(list)
        for idx, it in items:
            by_diff[it.get("difficulty", "?")].append((idx, it))
        order = []
        diffs = sorted(by_diff)
        for d in diffs:
            lst = by_diff[d]
            rng.shuffle(lst)
            order.append(lst)
        # round-robin across difficulties until we hit target
        take, di = [], 0
        while len(take) < min(target, len(items)) and any(order):
            bucket = order[di % len(order)]
            if bucket:
                take.append(bucket.pop())
            di += 1
            order = [b for b in order if b] or []
            if not order:
                break
        picked.extend(take)

    # de-dup, cap to n, stable order by category then difficulty
    seen, rows = set(), []
    for idx, it in picked:
        if idx in seen:
            continue
        seen.add(idx)
        rows.append((idx, it))
    rows = rows[:args.n]

    OUT.mkdir(parents=True, exist_ok=True)

    # --- jsonl (full) ---
    with open(OUT / "pilot_template.jsonl", "w", encoding="utf-8") as f:
        for idx, it in rows:
            kn = it.get("knowledge", [])
            kn = " ".join(map(str, kn)) if isinstance(kn, list) else str(kn)
            rec = {
                "id": f"mh_{idx:04d}",
                "difficulty": it.get("difficulty", ""),
                "category": it.get("category", ""),
                "question": str(it.get("question", "")),
                "evidence": kn,
                "ground_truth": str(it.get("ground_truth", "")),
                "hallucinated_answer": str(it.get("hallucinated_answer", "")),
                "error_spans": [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- annotator CSVs (utf-8-sig so Excel shows non-ASCII correctly) ---
    def write_csv(path):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(CONTEXT_COLS + FILL_COLS)
            for idx, it in rows:
                kn = it.get("knowledge", [])
                kn = " ".join(map(str, kn)) if isinstance(kn, list) else str(kn)
                w.writerow([
                    f"mh_{idx:04d}", it.get("difficulty", ""),
                    it.get("category", ""), str(it.get("question", "")),
                    kn[:args.evidence_cap],
                    str(it.get("ground_truth", ""))[:args.evidence_cap],
                    str(it.get("hallucinated_answer", "")),
                ] + [""] * len(FILL_COLS))

    write_csv(OUT / "pilot_A.csv")
    write_csv(OUT / "pilot_B.csv")

    # report
    cat_count = defaultdict(int)
    diff_count = defaultdict(int)
    for _, it in rows:
        cat_count[it.get("category")] += 1
        diff_count[it.get("difficulty")] += 1
    print(f"Wrote {len(rows)} pilot items to {OUT}")
    print("  by category:", dict(cat_count))
    print("  by difficulty:", dict(diff_count))
    print("  files: pilot_template.jsonl, pilot_A.csv, pilot_B.csv")


if __name__ == "__main__":
    main()
