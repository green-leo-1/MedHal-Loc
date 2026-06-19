# MedHal-Loc

A **localization-faithfulness benchmark and metric** for medical hallucination
detectors: it measures whether a detector's top-ranked error unit actually
overlaps the true erroneous span — *where* a method localizes, separate from
*whether* it detects.

## What's here

```
benchmark/
  medhal_loc_controlled.jsonl     # 300 controlled items, gold spans by construction (75 / type)
  inject_input.jsonl, inject_out_*.json   # error-injection inputs/outputs (provenance)
  horizontal/shard_*.json         # per-item localization outputs (AdaTriple / NLI-clause / SelfCheckGPT-NLI)
  fava_results.json               # FAVA (7B) per-item outputs
  _error_probe.txt                # per-item triple-extraction dump (error analysis)
annotation/
  llm_annot_{A,B,C}.json          # natural subset: 3-LLM span annotations
  consensus_gold.json             # consensus gold
  verify_natural_18*.{csv,xlsx}   # human audit (1/18 accepted)
  *_GUIDE.md                      # annotation guidelines
results/real/v9_realbaselines/    # detection panel source (mean F1 / AUC-PR, 5 datasets, 3 seeds)
src/                              # metric, evaluation, and construction code
data/hetionet_medical_kg.json     # 8,397-entity medical projection used by AdaTriple
paper/figures/make_medhalloc_figs.py + fig_*.pdf
```

## Reproduce

```bash
pip install -r requirements.txt   # transformers, torch, numpy, sklearn, selfcheckgpt, matplotlib

# Localization faithfulness (n=295) — self-contained on the benchmark jsonl
python src/horizontal_eval.py --nshards 3     # AdaTriple / NLI-clause / SelfCheckGPT-NLI
python src/fava_eval.py                         # FAVA (downloads fava-uw/fava-model, 7B)

# Natural-subset inter-annotator agreement (span-F1 0.87 / Fleiss' kappa 0.88)
python src/_llm_agreement.py

# Figures
python paper/figures/make_medhalloc_figs.py
```

The detection panel (`results/real/v9_realbaselines/per_method.json`) is provided
ready-made; a full re-run needs the five public source datasets and
`src/run_real_experiments.py`.

## Metric (in one line)

For each item, the method emits ranked candidate error units; we score
**hit@1 / hit@3** (does the top-ranked / a top-3 unit's tokens overlap the gold
span), subtract a **per-method random baseline** (expected overlap of a uniformly
random unit, controlling for unit granularity) to get the **lift**, and test
significance against the binomial SE. See the paper, Section "Localization-
Faithfulness Metric".

## Data sources & license

Add a permissive code license (e.g. MIT/Apache-2.0) before public release.
The benchmark derives from **MedHallu** (built on **PubMedQA**); detection uses
MedHallu, PubMedQA, MedQA, SciFact, and MMLU-Medical — each keeps its own license;
download from the official releases (not redistributed here). The KG projection
derives from **Hetionet v1.0 (CC0)**, https://het.io/.
