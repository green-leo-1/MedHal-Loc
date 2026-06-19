"""FAVA localization eval on MedHal-Loc controlled benchmark (Route C horizontal).

FAVA (fava-uw/fava-model, a dedicated 7B fine-grained hallucination DETECTOR)
reads (evidence, passage) and emits the passage with error spans tagged, e.g.
  <entity><delete>WRONG</delete><mark>fix</mark></entity>
The <delete> contents are FAVA's localized error predictions. We check whether
any FAVA-flagged span overlaps the gold error span.

This is the key EXTERNAL fine-grained detector for the "multiple methods" panel.

Run:  python src/fava_eval.py --n 300 --batch 4
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
WORKSPACE = Path(__file__).resolve().parent.parent
BM = WORKSPACE / "benchmark"

INPUT = ("Read the following references:\n{evidence}\nPlease identify all the "
         "errors in the following text using the information in the references "
         "provided and suggest edits if necessary:\n[Text] {output}\n[Edited] ")

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


def parse_fava(edited):
    """Return (flagged error tokens, n_delete_spans, any_tag) from FAVA output."""
    deletes = re.findall(r"<delete>(.*?)</delete>", edited, flags=re.S)
    # also <mark> insertions inside <invented>/<unverifiable> mark a hallucinated
    # ADD with no <delete>; count those tagged regions too
    marks_only = re.findall(r"<(?:invented|unverifiable|subjective)>(?:(?!<delete>).)*?"
                            r"<mark>(.*?)</mark>", edited, flags=re.S)
    spans = deletes + marks_only
    flagged = set()
    for sp in spans:
        flagged |= toks(re.sub(r"<.*?>", " ", sp))
    any_tag = bool(re.search(r"<(entity|relation|contradictory|invented|"
                             r"unverifiable|subjective)>", edited))
    return flagged, len(spans), any_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=600)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    items = [json.loads(l) for l in
             open(BM / "medhal_loc_controlled.jsonl", encoding="utf-8")][:args.n]

    print("Loading FAVA (fava-uw/fava-model, ~7B)...")
    tok = AutoTokenizer.from_pretrained("fava-uw/fava-model")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "fava-uw/fava-model", torch_dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = [INPUT.format(evidence=it["evidence"][:1500],
                            output=it["hallucinated_text"]) for it in items]
    edited_all = []
    bs = args.batch
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1600).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            new = out[j][enc["input_ids"].shape[1]:]
            edited_all.append(tok.decode(new, skip_special_tokens=True))
        if (i // bs) % 5 == 0:
            print(f"  {min(i+bs,len(prompts))}/{len(prompts)}")

    rows = []
    for it, ed in zip(items, edited_all):
        gold = toks(it["gold_span"])
        if not gold:
            continue
        flagged, nsp, any_tag = parse_fava(ed)
        rows.append(dict(tt=it["target_type"], flagged_any=1 if any_tag else 0,
                         n_spans=nsp,
                         hit=1 if (flagged & gold) else 0,
                         n_flagged_tok=len(flagged),
                         n_passage_tok=len(toks(it["hallucinated_text"]))))
    json.dump(rows, open(BM / "fava_results.json", "w"))

    def p(x):
        return 100.0 * np.mean(x) if x else float("nan")

    print("\n" + "=" * 70)
    print(f"FAVA localization on MedHal-Loc (n={len(rows)})")
    print("=" * 70)
    print(f"flagged anything           : {p([r['flagged_any'] for r in rows]):.1f}%")
    print(f"mean error spans / item    : {np.mean([r['n_spans'] for r in rows]):.1f}")
    print(f"HIT (a flagged span overlaps gold): {p([r['hit'] for r in rows]):.1f}%")
    # per-item chance = n_flagged_tok * gold_size / passage_size (expected overlap)
    ch = []
    for r in rows:
        # rough chance a random equal-size flag set hits the gold span
        gfrac = 1.0  # gold present in every item
        ch.append(min(1.0, r["n_flagged_tok"] / max(r["n_passage_tok"], 1)))
    print(f"random-flag chance (~flag coverage of passage): {100*np.mean(ch):.1f}%")
    print("-" * 70)
    print(f"{'error type':26s} {'n':>4s} {'flagged':>8s} {'hit':>6s}")
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r["tt"]].append(r)
    for tt in ["entity_substitution", "relation_error",
               "mechanism_misattribution", "invented"]:
        rs = g.get(tt, [])
        if rs:
            print(f"{tt:26s} {len(rs):4d} "
                  f"{p([r['flagged_any'] for r in rs]):8.1f} "
                  f"{p([r['hit'] for r in rs]):6.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
