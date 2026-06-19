"""Build consensus localization gold from the 3 LLM annotators (Route C).

A content word is a GOLD error word for an item iff it falls inside an error
span marked by >= 2 of the 3 annotators (majority). This denoises individual
annotators and yields a high-precision gold for the faithfulness eval.

Reads:  annotation/pilot_template.jsonl, annotation/llm_annot_{A,B,C}.json
Writes: annotation/consensus_gold.json  (per item: gold_words, dominant types)

Run:  python src/build_consensus_gold.py
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

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
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", (s or "").lower())
            if w not in STOP]


def annot_words_and_types(answer, spans):
    """-> ({word: type}) for one annotator: words inside located error spans."""
    low = answer.lower()
    wt = {}
    for s in spans:
        t = (s.get("text") or "")
        typ = (s.get("type") or "").lower()
        cs = low.find(t.lower())
        if cs < 0:
            nt = re.sub(r"\s+", " ", t.strip().lower())
            m = re.search(re.escape(nt).replace(r"\ ", r"\s+"), low)
            if not m:
                continue
            cs, ce = m.start(), m.end()
        else:
            ce = cs + len(t)
        for w in toks(answer[cs:ce]):
            wt.setdefault(w, typ)
    return wt


def main():
    text = {}
    for line in open(ANN / "pilot_template.jsonl", encoding="utf-8"):
        r = json.loads(line)
        text[r["id"]] = r["hallucinated_answer"]

    annot_files = sorted(glob.glob(str(ANN / "llm_annot_*.json")))
    annots = {}
    for f in annot_files:
        d = json.load(open(f, encoding="utf-8"))
        items = d.get("annotations", d) if isinstance(d, dict) else d
        annots[Path(f).stem] = {r["id"]: (r.get("spans") or []) for r in items}
    n_ann = len(annots)
    print(f"loaded {n_ann} annotators: {list(annots)}")

    gold = {}
    sizes, type_counter = [], Counter()
    for i, ans in text.items():
        word_votes = Counter()
        type_votes = {}
        for name, byid in annots.items():
            wt = annot_words_and_types(ans, byid.get(i, []))
            for w, typ in wt.items():
                word_votes[w] += 1
                type_votes.setdefault(w, Counter())[typ] += 1
        gold_words = {w for w, c in word_votes.items() if c >= 2}  # majority
        gw_types = {w: type_votes[w].most_common(1)[0][0] for w in gold_words}
        for t in gw_types.values():
            type_counter[t] += 1
        gold[i] = {"gold_words": sorted(gold_words), "types": gw_types}
        sizes.append(len(gold_words))

    with open(ANN / "consensus_gold.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)

    print(f"wrote consensus_gold.json: {len(gold)} items")
    print(f"  mean gold words / item : {np.mean(sizes):.1f}  "
          f"(min {min(sizes)}, max {max(sizes)})")
    print(f"  items with >=1 gold word: {sum(1 for s in sizes if s>0)}/{len(sizes)}")
    print(f"  consensus error-type mix: {dict(type_counter.most_common())}")


if __name__ == "__main__":
    main()
