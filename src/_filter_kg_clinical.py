"""Filter Hetionet KG to only clinical entities (no Gene), keep manageable size."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "hetionet_medical_kg.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

CLINICAL_TYPES = {"Compound", "Disease", "Symptom", "Side Effect",
                  "Anatomy", "Pharmacologic Class"}

CLINICAL_EDGE_TYPES = {
    "treats", "palliates", "causes", "presents",
    "localizes", "resembles", "includes",
}

clinical_ent_ids = {e["id"] for e in kg["entities"]
                    if e.get("type", "") in CLINICAL_TYPES}
print(f"Clinical entity IDs: {len(clinical_ent_ids)}")

filtered_rels = []
for r in kg["relations"]:
    if r["relation"] not in CLINICAL_EDGE_TYPES:
        continue
    if r["head"] in clinical_ent_ids and r["tail"] in clinical_ent_ids:
        filtered_rels.append(r)

used_ids = set()
for r in filtered_rels:
    used_ids.add(r["head"])
    used_ids.add(r["tail"])

filtered_ents = [e for e in kg["entities"] if e["id"] in used_ids]

print(f"\n=== Clinical-Only KG ===")
print(f"Entities: {len(filtered_ents)}")
print(f"Relations: {len(filtered_rels)}")

for et, cnt in Counter(e.get("type", "?") for e in filtered_ents).most_common():
    print(f"  {et:25s} {cnt:>6d}")
for et, cnt in Counter(r["relation"] for r in filtered_rels).most_common():
    print(f"  {et:25s} {cnt:>6d}")

out = {"entities": filtered_ents, "relations": filtered_rels}
out_path = DATA_DIR / "hetionet_clinical_kg.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
