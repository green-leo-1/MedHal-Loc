"""Filter Hetionet KG to keep only clinically relevant edges, reduce size."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "hetionet_kg.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

print("=== Original KG ===")
print(f"Entities: {len(kg['entities'])}")
print(f"Relations: {len(kg['relations'])}")

edge_types = Counter(r["relation"] for r in kg["relations"])
for et, count in edge_types.most_common():
    print(f"  {et:40s} {count:>8d}")

# Keep only clinically relevant edge types
KEEP_TYPES = {
    "treats", "palliates", "causes", "presents",
    "localizes", "resembles", "includes", "associates",
}

filtered_rels = [r for r in kg["relations"] if r["relation"] in KEEP_TYPES]

# Collect entity IDs that appear in filtered relations
used_ids = set()
for r in filtered_rels:
    used_ids.add(r["head"])
    used_ids.add(r["tail"])

filtered_ents = [e for e in kg["entities"] if e["id"] in used_ids]

print(f"\n=== Filtered KG (clinical only) ===")
print(f"Entities: {len(filtered_ents)}")
print(f"Relations: {len(filtered_rels)}")

edge_types2 = Counter(r["relation"] for r in filtered_rels)
for et, count in edge_types2.most_common():
    print(f"  {et:40s} {count:>8d}")

ent_types = Counter(e.get("type", "unknown") for e in filtered_ents)
for et, count in ent_types.most_common():
    print(f"  Entity type: {et:30s} {count:>6d}")

out = {"entities": filtered_ents, "relations": filtered_rels}
out_path = DATA_DIR / "hetionet_medical_kg.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"\nSaved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
