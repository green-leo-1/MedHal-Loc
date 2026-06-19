"""Horizontal localization-faithfulness eval: MULTIPLE fine-grained detectors
on the MedHal-Loc controlled benchmark (Route C; addresses the "single-method"
reviewer attack).

Each method emits ranked candidate error UNITS (with scores); we ask whether
the top-ranked unit overlaps the gold error span, vs a per-method random
baseline (controls for unit granularity). Methods span paradigms:

  * AdaTriple          : KG triple decomposition (unit = triple)
  * NLI-clause         : entailment per clause (unit = clause)
  * SelfCheckGPT-NLI   : consistency per sentence (unit = sentence)
  (FAVA, the dedicated span detector, is evaluated separately in fava_eval.py)

DATA-PARALLEL:
  python src/horizontal_eval.py --nshards 3            # driver: parallel + report
  python src/horizontal_eval.py --shard 0 --nshards 3  # one worker
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
WORKSPACE = Path(__file__).resolve().parent.parent
BM = WORKSPACE / "benchmark"
OUTD = BM / "horizontal"

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


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_clauses(text):
    out = []
    for s in split_sentences(text):
        for c in re.split(r"[;,]|\s+(?:and|but|which|where|while|whereas|"
                          r"although|because|whereas)\s+", s):
            c = c.strip()
            if len(toks(c)) >= 2:
                out.append(c)
    return out or split_sentences(text)


def metrics_from_units(units_scores, gold):
    """units_scores: list of (unit_text, score). -> dict of per-item metrics."""
    if not units_scores:
        return dict(n=0, cover=0, hit1=0, hit3=0, rand=0.0, margin=None)
    flags = [len(toks(u) & gold) > 0 for u, _ in units_scores]
    order = sorted(range(len(units_scores)),
                   key=lambda j: units_scores[j][1], reverse=True)
    ne = sum(flags)
    g = [units_scores[j][1] for j in range(len(flags)) if flags[j]]
    o = [units_scores[j][1] for j in range(len(flags)) if not flags[j]]
    return dict(
        n=len(units_scores), cover=1 if ne else 0,
        hit1=1 if flags[order[0]] else 0,
        hit3=1 if any(flags[j] for j in order[:3]) else 0,
        rand=ne / len(units_scores),
        margin=(float(np.mean(g) - np.mean(o)) if g and o else None),
    )


def run_shard(shard, nshards):
    import logging
    logging.basicConfig(level=logging.ERROR)
    import torch
    try:
        torch.cuda.set_per_process_memory_fraction(0.30, 0)
    except Exception:
        pass
    from transformers import pipeline as hf_pipeline
    from adatriple import AdaTriple

    items = [json.loads(l) for l in
             open(BM / "medhal_loc_controlled.jsonl", encoding="utf-8")]
    mine = [it for k, it in enumerate(items) if k % nshards == shard]

    cfg = {
        "kg_path": str(WORKSPACE / "data" / "hetionet_medical_kg.json"),
        "kg_format": "json", "device": "cuda", "lang": "en",
        "tau_h": 0.5, "tau_e": 0.4, "beta": 0.5,
        "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "use_cmtv": True, "use_uctt": False, "use_hcd": True,
        "use_enhanced_lambda": True,
    }
    ada = AdaTriple(cfg)
    nli = hf_pipeline("text-classification",
                      model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                      device=0, top_k=None)
    try:
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
        sc = SelfCheckNLI(device=torch.device("cuda"))
    except Exception as e:
        logging.error("SelfCheck load failed: %s", e)
        sc = None

    def nli_hall(premise, hyp):
        try:
            res = nli({"text": premise[:512], "text_pair": hyp[:256]}, top_k=None)
            ent = con = 0.34
            for d in res:
                lab = d["label"].lower()
                if "entail" in lab:
                    ent = d["score"]
                elif "contra" in lab:
                    con = d["score"]
            return 1.0 - ent / (ent + con + 1e-8)
        except Exception:
            return 0.5

    recs = []
    for it in mine:
        H, E = it["hallucinated_text"], it["evidence"]
        gold = toks(it["gold_span"])
        tt = it["target_type"]
        if not gold:
            continue
        row = {"tt": tt}
        # AdaTriple
        try:
            res = ada.detect(H, evidence=E, verbose=False)
            us = [(t.head_entity.text + " " + t.tail_entity.text,
                   float(t.hallucination_score)) for t in res.triples]
        except Exception:
            us = []
        row["AdaTriple"] = metrics_from_units(us, gold)
        # NLI-clause
        cl = split_clauses(H)
        row["NLI-clause"] = metrics_from_units([(c, nli_hall(E, c)) for c in cl], gold)
        # SelfCheckGPT-NLI sentence
        if sc is not None:
            sents = split_sentences(H)
            try:
                ss = sc.predict(sentences=sents, sampled_passages=[E])
                us2 = [(sents[j], float(ss[j])) for j in range(len(sents))]
            except Exception:
                us2 = []
            row["SelfCheckGPT-NLI"] = metrics_from_units(us2, gold)
        recs.append(row)

    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(OUTD / f"shard_{shard}.json", "w"))
    print(f"[shard {shard}] {len(recs)} items")


def driver(nshards):
    OUTD.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(OUTD / "shard_*.json")):
        os.remove(f)
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs = []
    for k in range(nshards):
        cmd = [sys.executable, "-u", str(Path(__file__)),
               "--shard", str(k), "--nshards", str(nshards)]
        lf = open(OUTD / f"shard_{k}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, stdout=lf,
                                      stderr=subprocess.STDOUT))
    print(f"launched {nshards} parallel workers...")
    for k, p in enumerate(procs):
        p.wait(); print(f"  worker {k} exit={p.returncode}")

    recs = []
    for k in range(nshards):
        fp = OUTD / f"shard_{k}.json"
        if fp.exists():
            recs.extend(json.load(open(fp)))
    report(recs)


def report(recs):
    methods = ["AdaTriple", "NLI-clause", "SelfCheckGPT-NLI"]
    by = {m: defaultdict(list) for m in methods}
    for r in recs:
        for m in methods:
            if m in r:
                for key in ("cover", "hit1", "hit3", "rand"):
                    by[m][key].append(r[m][key])
                if r[m]["margin"] is not None:
                    by[m]["margin"].append(r[m]["margin"])

    def p(x):
        return 100.0 * np.mean(x) if x else float("nan")

    print("\n" + "=" * 78)
    print(f"MedHal-Loc HORIZONTAL localization faithfulness (n={len(recs)})")
    print("=" * 78)
    print(f"{'method':20s} {'gran.':9s} {'cover':>6s} {'hit@1':>6s} "
          f"{'hit@3':>6s} {'rand':>6s} {'lift':>6s} {'SE':>4s} {'margin':>7s}")
    gran = {"AdaTriple": "triple", "NLI-clause": "clause",
            "SelfCheckGPT-NLI": "sentence"}
    for m in methods:
        b = by[m]
        if not b["hit1"]:
            continue
        n = len(b["hit1"])
        lift = p(b["hit1"]) - p(b["rand"])
        se = 100.0 * np.std(b["hit1"]) / np.sqrt(max(n, 1))
        mg = np.mean(b["margin"]) if b["margin"] else float("nan")
        print(f"{m:20s} {gran[m]:9s} {p(b['cover']):6.1f} {p(b['hit1']):6.1f} "
              f"{p(b['hit3']):6.1f} {p(b['rand']):6.1f} {lift:+6.1f} {se:4.1f} "
              f"{mg:+7.3f}")
    print("-" * 78)
    print("Response-level (no fine-grained output): NLI-DeBERTa / HHEM / "
          "LLM-judge -> localization N/A (0%)")
    print("lift > ~2*SE => real localization above chance for that method")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=3)
    a = ap.parse_args()
    if a.shard >= 0:
        run_shard(a.shard, a.nshards)
    else:
        driver(a.nshards)


if __name__ == "__main__":
    main()
