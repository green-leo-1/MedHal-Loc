"""
Build AdaTriple-compatible medical KG from Hetionet v1.0
========================================================
Hetionet: 47,031 nodes (11 types), 2,250,197 edges (24 types)

Filters to medical-relevant node types:
  - Disease, Symptom, Compound (drugs), Anatomy, Pharmacologic Class, Side Effect

Maps Hetionet metaedges to AdaTriple relation types:
  - CtD (Compound treats Disease)      -> treats
  - CpD (Compound palliates Disease)   -> treats
  - CcSE (Compound causes Side Effect) -> causes
  - DaG (Disease associates Gene)      -> risk_factor_for
  - DlA (Disease localizes Anatomy)    -> located_in
  - DpS (Disease presents Symptom)     -> symptom_of  (new: added)
  - DrD (Disease resembles Disease)    -> complication_of
  - CuG (Compound upregulates Gene)    -> interacts_with
  - CdG (Compound downregulates Gene)  -> interacts_with

Usage:
    python build_hetionet_kg.py
"""

import sys
import io
import os
import gzip
import json
import time
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hetionet"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "hetionet_medical_kg.json"

MEDICAL_NODE_TYPES = {
    "Disease", "Symptom", "Compound", "Anatomy",
    "Pharmacologic Class", "Side Effect",
}

METAEDGE_MAP = {
    "CtD": "treats",
    "CpD": "treats",
    "CcSE": "causes",
    "DpS": "symptom_of",
    "DlA": "located_in",
    "DrD": "complication_of",
    "DaG": "risk_factor_for",
    "CuG": "interacts_with",
    "CdG": "interacts_with",
    "CbG": "interacts_with",
    "PCiC": "contraindicated_with",
}


def load_nodes(path: Path) -> dict:
    """Load nodes TSV using lowercase name as key for direct matching.

    Each entity gets:
      - id: original Hetionet ID (e.g. "Compound::DB00331")
      - name: human-readable name (e.g. "Metformin")
      - aliases: list of matchable name variants (lowercase, no prefix)
      - entity_type: mapped type for AdaTriple
    """
    nodes = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            nid, name, kind = parts[0], parts[1], parts[2]
            if kind in MEDICAL_NODE_TYPES:
                etype = {
                    "Disease": "disease",
                    "Symptom": "symptom",
                    "Compound": "drug",
                    "Anatomy": "body_part",
                    "Pharmacologic Class": "drug",
                    "Side Effect": "symptom",
                }.get(kind, "disease")

                key = name.lower()
                aliases = [name, name.lower()]
                # "Compound::DB00331" -> extract DB ID as alias
                if "::" in nid:
                    short_id = nid.split("::")[-1]
                    aliases.append(short_id.lower())

                nodes[key] = {
                    "id": key,
                    "original_id": nid,
                    "name": name,
                    "aliases": aliases,
                    "entity_type": etype,
                    "kind": kind,
                }
    return nodes


def load_edges(path: Path, nodes: dict, id_to_key: dict) -> list:
    """Load edges SIF.GZ, mapping original IDs to lowercase name keys."""
    edges = []
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            src, metaedge, tgt = parts[0], parts[1], parts[2]
            rel = METAEDGE_MAP.get(metaedge)
            src_key = id_to_key.get(src)
            tgt_key = id_to_key.get(tgt)
            if rel and src_key and tgt_key:
                edges.append({
                    "head": src_key,
                    "tail": tgt_key,
                    "relation": rel,
                    "weight": 1.0,
                })
    return edges


def main():
    t0 = time.time()
    print("=" * 60)
    print("Building AdaTriple medical KG from Hetionet v1.0")
    print("=" * 60)

    nodes_path = DATA_DIR / "nodes.tsv"
    edges_path = DATA_DIR / "edges.sif.gz"

    if not nodes_path.exists() or not edges_path.exists():
        print(f"ERROR: Hetionet files not found in {DATA_DIR}")
        print("Run download first.")
        sys.exit(1)

    print(f"\n[1/3] Loading nodes from {nodes_path}...")
    nodes = load_nodes(nodes_path)
    print(f"  Total medical nodes: {len(nodes)}")
    kind_counts = defaultdict(int)
    for n in nodes.values():
        kind_counts[n["kind"]] += 1
    for k, c in sorted(kind_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {c}")

    # Build reverse mapping: original Hetionet ID -> lowercase name key
    id_to_key = {}
    for key, info in nodes.items():
        id_to_key[info["original_id"]] = key

    print(f"\n[2/3] Loading edges from {edges_path}...")
    edges = load_edges(edges_path, nodes, id_to_key)
    print(f"  Total medical edges: {len(edges)}")
    rel_counts = defaultdict(int)
    for e in edges:
        rel_counts[e["relation"]] += 1
    for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c}")

    print(f"\n[3/3] Saving to {OUTPUT}...")
    kg = {
        "metadata": {
            "source": "Hetionet v1.0",
            "url": "https://github.com/hetio/hetionet",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
        "entities": list(nodes.values()),
        "relations": edges,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"KG: {len(nodes)} entities, {len(edges)} relations")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
