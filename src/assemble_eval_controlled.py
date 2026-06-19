"""Assemble the controlled-injection MedHal-Loc benchmark and run the
localization-faithfulness eval on it (clean, gold-by-construction), DATA-PARALLEL.

  python src/assemble_eval_controlled.py                 # assemble + parallel eval + report
  python src/assemble_eval_controlled.py --shard 0 --nshards 5   # one eval worker

Merges benchmark/inject_out_*.json with benchmark/inject_input.jsonl, validates
each error_span is a verbatim substring of its hallucinated_text, writes
benchmark/medhal_loc_controlled.jsonl, then runs AdaTriple (sharded across
worker subprocesses) and reports hit@1 / hit@3 / random-baseline / margin
OVERALL and PER target error type.
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


def locate(span, text):
    low = text.lower()
    if low.find(span.lower()) >= 0:
        return True
    nt = re.sub(r"\s+", " ", span.strip().lower())
    return re.search(re.escape(nt).replace(r"\ ", r"\s+"), low) is not None


def assemble():
    src = {}
    for l in open(BM / "inject_input.jsonl", encoding="utf-8"):
        r = json.loads(l)
        src[r["id"]] = r
    items, bad = [], 0
    for f in sorted(glob.glob(str(BM / "inject_out_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d.get("items", d if isinstance(d, list) else []):
            i = it.get("id")
            hal = it.get("hallucinated_text", "")
            span = it.get("error_span", "")
            if i not in src or not hal or not span or not locate(span, hal):
                bad += 1
                continue
            s = src[i]
            items.append({
                "id": i, "source": "injected",
                "target_type": it.get("target_type", s.get("target_type")),
                "question": s.get("question", ""), "evidence": s.get("evidence", ""),
                "clean_text": s.get("clean_text", ""),
                "hallucinated_text": hal, "gold_span": span,
            })
    with open(BM / "medhal_loc_controlled.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"assembled {len(items)} valid items ({bad} dropped) "
          f"-> benchmark/medhal_loc_controlled.jsonl")
    return items


def run_shard(shard, nshards):
    import logging
    logging.basicConfig(level=logging.ERROR)
    import torch
    try:
        torch.cuda.set_per_process_memory_fraction(0.22, 0)
    except Exception:
        pass
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
    pipe = AdaTriple(cfg)
    recs = []
    for it in mine:
        gold = toks(it["gold_span"])
        if not gold:
            continue
        try:
            res = pipe.detect(it["hallucinated_text"],
                              evidence=it["evidence"], verbose=False)
        except Exception:
            continue
        tr = list(res.triples)
        tt = it["target_type"]
        if not tr:
            recs.append(dict(tt=tt, n=0, cover=0, hit1=0, hit3=0, rand=0.0,
                             margin=None))
            continue
        H = [float(t.hallucination_score) for t in tr]
        order = sorted(range(len(tr)), key=lambda j: H[j], reverse=True)
        flags = [len((toks(t.head_entity.text) | toks(t.tail_entity.text)) & gold) > 0
                 for t in tr]
        ne = sum(flags)
        g = [H[j] for j in range(len(tr)) if flags[j]]
        o = [H[j] for j in range(len(tr)) if not flags[j]]
        recs.append(dict(tt=tt, n=len(tr), cover=1 if ne else 0,
                         hit1=1 if flags[order[0]] else 0,
                         hit3=1 if any(flags[j] for j in order[:3]) else 0,
                         rand=ne / len(tr),
                         margin=(float(np.mean(g) - np.mean(o)) if g and o else None)))
    BM.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(BM / f"eval_shard_{shard}.json", "w"))
    print(f"[shard {shard}] {len(recs)} records")


def driver(nshards):
    assemble()
    for f in glob.glob(str(BM / "eval_shard_*.json")):
        os.remove(f)
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs = []
    for k in range(nshards):
        cmd = [sys.executable, "-u", str(Path(__file__)),
               "--shard", str(k), "--nshards", str(nshards)]
        lf = open(BM / f"eval_shard_{k}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, stdout=lf,
                                      stderr=subprocess.STDOUT))
    print(f"launched {nshards} parallel eval workers...")
    for k, p in enumerate(procs):
        p.wait(); print(f"  worker {k} exit={p.returncode}")

    recs = []
    for k in range(nshards):
        fp = BM / f"eval_shard_{k}.json"
        if fp.exists():
            recs.extend(json.load(open(fp)))

    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for scope in (r["tt"], "ALL"):
            by[scope]["cover"].append(r["cover"])
            by[scope]["hit1"].append(r["hit1"])
            by[scope]["hit3"].append(r["hit3"])
            by[scope]["rand"].append(r["rand"])
            if r["margin"] is not None:
                by[scope]["margin"].append(r["margin"])

    def p(x):
        return 100.0 * np.mean(x) if x else float("nan")

    print("\n" + "=" * 74)
    print(f"MedHal-Loc CONTROLLED faithfulness (n={len(recs)}): "
          "AdaTriple localization vs gold span")
    print("=" * 74)
    print(f"{'error type':26s} {'n':>4s} {'cover':>6s} {'hit@1':>6s} "
          f"{'hit@3':>6s} {'rand':>6s} {'lift':>6s} {'SE':>4s} {'margin':>7s}")
    for tt in ["entity_substitution", "relation_error",
               "mechanism_misattribution", "invented", "ALL"]:
        if tt not in by:
            continue
        r = by[tt]
        n = len(r["hit1"])
        lift = p(r["hit1"]) - p(r["rand"])
        se = 100.0 * np.std(r["hit1"]) / np.sqrt(max(n, 1))
        mg = np.mean(r["margin"]) if r["margin"] else float("nan")
        print(f"{tt:26s} {n:4d} {p(r['cover']):6.1f} {p(r['hit1']):6.1f} "
              f"{p(r['hit3']):6.1f} {p(r['rand']):6.1f} {lift:+6.1f} "
              f"{se:4.1f} {mg:+7.3f}")
    print("-" * 74)
    print("Response-level baselines (NLI/HHEM/SelfCheck/LLM-judge): 0% (no spans)")
    print("lift > ~2*SE => statistically real localization above chance")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=5)
    a = ap.parse_args()
    if a.shard >= 0:
        run_shard(a.shard, a.nshards)
    else:
        driver(a.nshards)


if __name__ == "__main__":
    main()
