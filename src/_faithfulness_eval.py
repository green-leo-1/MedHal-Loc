"""Localization-faithfulness eval against the LLM-consensus gold (Route C main result).

For each of the 40 pilot items, runs AdaTriple, ranks its triples by
hallucination_score, and asks: does the top-ranked suspicious triple's
entities overlap the CONSENSUS gold error words (>=2/3 LLM annotators)?

Compared to the v1/v2 auto-gold (token/entity diff, random baseline ~40-44%),
the consensus gold is high precision, so the random baseline should drop and a
real ranking signal (if any) should show.

Response-level baselines (NLI/HHEM/SelfCheck/LLM-judge) emit NO fine-grained
output, so their localization is 0% by construction -- reported as the contrast.

Run:  python src/_faithfulness_eval.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import logging
logging.basicConfig(level=logging.ERROR)

from adatriple import AdaTriple  # noqa: E402

WORKSPACE = Path(__file__).resolve().parent.parent
ANN = WORKSPACE / "annotation"

STOP = set("""
a an the of to in on for and or but with without within from by as at is are was
were be been being this that these those it its their our your his her them they
we you he she i which who whom whose what when where why how than then so such not
no nor can could should would may might will shall do does did has have had having
into onto over under above below between among during after before through results
result study studies showed show shows shown using used use also however both each
other more most less least very much many few patient patients group groups data
significant significantly compared comparison versus vs approach approaches found
""".split())


def toks(s):
    return set(w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", (s or "").lower())
               if w not in STOP)


def main():
    gold = json.load(open(ANN / "consensus_gold.json", encoding="utf-8"))
    text = {}
    for line in open(ANN / "pilot_template.jsonl", encoding="utf-8"):
        r = json.loads(line)
        text[r["id"]] = {"hal": r["hallucinated_answer"], "ev": r["evidence"]}

    cfg = {
        "kg_path": str(WORKSPACE / "data" / "hetionet_medical_kg.json"),
        "kg_format": "json", "device": "cuda", "lang": "en",
        "tau_h": 0.5, "tau_e": 0.4, "beta": 0.5,
        "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "use_cmtv": True, "use_uctt": False, "use_hcd": True,
        "use_enhanced_lambda": True,
    }
    print("Loading AdaTriple...")
    pipe = AdaTriple(cfg)

    cover, hit1, hit3, rand, margins = [], [], [], [], []
    n_eval = n_notri = n_nogold = 0
    for i, g in gold.items():
        gw = set(g["gold_words"])
        if not gw:
            n_nogold += 1
            continue
        ans, ev = text[i]["hal"], text[i]["ev"]
        try:
            res = pipe.detect(ans, evidence=ev, verbose=False)
        except Exception:
            continue
        tr = list(res.triples)
        if not tr:
            n_notri += 1
            n_eval += 1
            cover.append(0); hit1.append(0); hit3.append(0); rand.append(0.0)
            continue

        def ew(t):
            return toks(t.head_entity.text) | toks(t.tail_entity.text)

        def hit(t):
            return len(ew(t) & gw) > 0

        H = [float(t.hallucination_score) for t in tr]
        order = sorted(range(len(tr)), key=lambda j: H[j], reverse=True)
        flags = [hit(t) for t in tr]
        n_err = sum(flags)
        n_eval += 1
        cover.append(1 if n_err else 0)
        hit1.append(1 if flags[order[0]] else 0)
        hit3.append(1 if any(flags[j] for j in order[:3]) else 0)
        rand.append(n_err / len(tr))
        g_H = [H[j] for j in range(len(tr)) if flags[j]]
        o_H = [H[j] for j in range(len(tr)) if not flags[j]]
        if g_H and o_H:
            margins.append(float(np.mean(g_H) - np.mean(o_H)))

    def pct(x):
        return 100.0 * np.mean(x) if x else 0.0

    print("\n" + "=" * 64)
    print("AdaTriple localization faithfulness vs LLM-CONSENSUS gold (n=40)")
    print("=" * 64)
    print(f"evaluable items           : {n_eval}  (0-triple {n_notri}, "
          f"no-gold {n_nogold})")
    print(f"mean gold words / item    : "
          f"{np.mean([len(set(g['gold_words'])) for g in gold.values()]):.1f}")
    print("-" * 64)
    print(f"COVERAGE (any triple hits) : {pct(cover):5.1f}%")
    print(f"hit@1                      : {pct(hit1):5.1f}%")
    print(f"hit@3                      : {pct(hit3):5.1f}%")
    print(f"random-triple baseline     : {pct(rand):5.1f}%   <- chance")
    lift = pct(hit1) - pct(rand)
    se = 100.0 * np.std(hit1) / np.sqrt(max(len(hit1), 1))
    print(f"hit@1 lift over random     : {lift:+5.1f}pp   (hit@1 SE~{se:.1f}pp)")
    if margins:
        m, mse = np.mean(margins), np.std(margins) / np.sqrt(len(margins))
        print(f"localization MARGIN        : {m:+.3f}  (SE {mse:.3f}, n={len(margins)})")
    print("-" * 64)
    print("Response-level baselines (NLI / HHEM / SelfCheck / LLM-judge):")
    print("  localization = 0% by construction (no fine-grained output).")
    print("=" * 64)


if __name__ == "__main__":
    main()
