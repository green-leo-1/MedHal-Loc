"""Build the 50-item HUMAN VERIFICATION anchor (Route C credibility step).

The human does NOT annotate from scratch -- the LLM gold is pre-filled and the
human only marks each proposed error span correct / wrong / partial (+ type ok,
+ any missed error). This certifies the gold cheaply and yields the LLM-human
agreement number reviewers want.

  32 controlled items (8 per error type) -> QC the injected gold.
  18 natural items (LLM annotator A spans) -> LLM-human agreement on REAL halluc.

Writes:
  annotation/verify_50.csv          (Excel template, utf-8-sig, two annotators
                                     copy to verify_50_H1.csv / _H2.csv)
  annotation/verify_50_context.jsonl (machine-readable, for compute_validation)

Run:  python src/build_verification_set.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
BM = WORKSPACE / "benchmark"
ANN = WORKSPACE / "annotation"
SEED = 11
N_CTRL_PER_TYPE = 8
N_NAT = 18
CAP = 900

COLS = ["id", "subset", "question", "evidence", "source_text",
        "hallucinated_text",
        "llm_span_1", "llm_type_1", "span1_ok(Y/N/P)", "type1_ok(Y/N)",
        "llm_span_2", "llm_type_2", "span2_ok(Y/N/P)", "type2_ok(Y/N)",
        "missed_error", "missed_type", "notes"]


def main():
    rng = np.random.RandomState(SEED)
    rows = []          # csv rows
    ctx = []           # jsonl context

    # --- controlled: 8 per type ---
    ctrl = [json.loads(l) for l in
            open(BM / "medhal_loc_controlled.jsonl", encoding="utf-8")]
    by_t = defaultdict(list)
    for it in ctrl:
        by_t[it["target_type"]].append(it)
    for t, lst in by_t.items():
        idx = rng.permutation(len(lst))[:N_CTRL_PER_TYPE]
        for j in idx:
            it = lst[j]
            rows.append({
                "id": it["id"], "subset": "controlled",
                "question": it.get("question", ""),
                "evidence": it.get("evidence", "")[:CAP],
                "source_text": it.get("clean_text", "")[:CAP],
                "hallucinated_text": it["hallucinated_text"],
                "llm_span_1": it["gold_span"], "llm_type_1": it["target_type"],
                "span1_ok(Y/N/P)": "", "type1_ok(Y/N)": "",
                "llm_span_2": "", "llm_type_2": "",
                "span2_ok(Y/N/P)": "", "type2_ok(Y/N)": "",
                "missed_error": "", "missed_type": "", "notes": "",
            })
            ctx.append({"id": it["id"], "subset": "controlled",
                        "hallucinated_text": it["hallucinated_text"],
                        "llm_spans": [{"text": it["gold_span"],
                                       "type": it["target_type"]}]})

    # --- natural: 18 pilot items with annotator A's spans ---
    pilot = {json.loads(l)["id"]: json.loads(l)
             for l in open(ANN / "pilot_template.jsonl", encoding="utf-8")}
    annA = json.load(open(ANN / "llm_annot_A.json", encoding="utf-8"))
    annA = {r["id"]: (r.get("spans") or [])
            for r in annA.get("annotations", annA)}
    nat_ids = list(pilot)
    pick = [nat_ids[j] for j in rng.permutation(len(nat_ids))[:N_NAT]]
    for i in pick:
        it = pilot[i]
        spans = annA.get(i, [])[:2]
        s1 = spans[0] if len(spans) > 0 else {"text": "", "type": ""}
        s2 = spans[1] if len(spans) > 1 else {"text": "", "type": ""}
        rows.append({
            "id": i, "subset": "natural",
            "question": it.get("question", ""),
            "evidence": it.get("evidence", "")[:CAP],
            "source_text": it.get("ground_truth", "")[:CAP],
            "hallucinated_text": it.get("hallucinated_answer", ""),
            "llm_span_1": s1["text"], "llm_type_1": s1["type"],
            "span1_ok(Y/N/P)": "", "type1_ok(Y/N)": "",
            "llm_span_2": s2["text"], "llm_type_2": s2["type"],
            "span2_ok(Y/N/P)": "", "type2_ok(Y/N)": "",
            "missed_error": "", "missed_type": "", "notes": "",
        })
        ctx.append({"id": i, "subset": "natural",
                    "hallucinated_text": it.get("hallucinated_answer", ""),
                    "llm_spans": [s for s in spans if s.get("text")]})

    ANN.mkdir(parents=True, exist_ok=True)
    with open(ANN / "verify_50.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(ANN / "verify_50_context.jsonl", "w", encoding="utf-8") as f:
        for c in ctx:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    nc = sum(1 for r in rows if r["subset"] == "controlled")
    nn = sum(1 for r in rows if r["subset"] == "natural")
    print(f"wrote {len(rows)} verification items "
          f"({nc} controlled, {nn} natural) -> annotation/verify_50.csv")
    print("Two annotators: copy verify_50.csv to verify_50_H1.csv and "
          "verify_50_H2.csv, fill independently, then run compute_validation.py")


if __name__ == "__main__":
    main()
