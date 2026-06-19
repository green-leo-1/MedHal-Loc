"""Sample clean medical statements for CONTROLLED error injection (MedHal-Loc).

Draws clean, evidence-supported statements (MedHallu ground_truths) and assigns
each a target localizable error type. An LLM editor (see the inject workflow)
will inject ONE error of that type and return its exact span -> gold by
construction. This oversamples the *localizable* error types (entity/relation/
mechanism/invented), which the pilot showed MedHallu's natural hallucinations
lack.

Run:  python src/build_inject_input.py --n 120
"""
import argparse
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
OUT = WORKSPACE / "benchmark"
SEED = 7  # different from the pilot (seed 42) so items don't overlap

TARGET_TYPES = ["entity_substitution", "relation_error",
                "mechanism_misattribution", "invented"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--evidence_cap", type=int, default=1200)
    args = ap.parse_args()

    data = json.load(open(WORKSPACE / "data" / "medhallu_pqa_labeled.json",
                          encoding="utf-8"))
    # keep items with a substantive ground_truth (room to inject a localized error)
    cand = [(i, it) for i, it in enumerate(data)
            if 40 <= len(str(it.get("ground_truth", ""))) <= 600]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(cand))[:args.n]
    rows = [cand[j] for j in idx]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "inject_input.jsonl", "w", encoding="utf-8") as f:
        for k, (i, it) in enumerate(rows):
            kn = it.get("knowledge", [])
            kn = " ".join(map(str, kn)) if isinstance(kn, list) else str(kn)
            rec = {
                "id": f"inj_{i:04d}",
                "target_type": TARGET_TYPES[k % len(TARGET_TYPES)],
                "question": str(it.get("question", "")),
                "evidence": kn[:args.evidence_cap],
                "clean_text": str(it.get("ground_truth", "")),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    tc = Counter(TARGET_TYPES[k % len(TARGET_TYPES)] for k in range(len(rows)))
    print(f"wrote {len(rows)} items -> {OUT/'inject_input.jsonl'}")
    print("  target-type allocation:", dict(tc))


if __name__ == "__main__":
    main()
