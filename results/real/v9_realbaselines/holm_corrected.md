# Holm-Bonferroni multiple-testing correction

Family-wise error rate controlled at alpha = 0.05 separately within each metric.  Step-down rule: reject H_(i) iff p_(i) <= alpha / (m - i + 1) for all earlier i in sorted order.  Hypotheses are 'AdaTriple+ > baseline' for one-sided paired bootstrap; ablation comparisons are excluded from the family.

## F1 family

Family size m = 35.  alpha = 0.05.  **17/35** comparisons survive Holm-Bonferroni.

| Rank | Dataset | Baseline | Raw p | Holm threshold | Survive? |
| ---:| --- | --- | ---:| ---:| :---:|
| 1 | MedHallu | HHEM | 0.0000 | 0.00143 | ✓ |
| 2 | MedHallu | LLM-Judge | 0.0000 | 0.00147 | ✓ |
| 3 | MedHallu | Keyword-Match | 0.0000 | 0.00152 | ✓ |
| 4 | MedHallu | Random | 0.0000 | 0.00156 | ✓ |
| 5 | MedHallu | Always-Positive | 0.0000 | 0.00161 | ✓ |
| 6 | MedQA | NLI-DeBERTa | 0.0000 | 0.00167 | ✓ |
| 7 | MedQA | LLM-Judge | 0.0000 | 0.00172 | ✓ |
| 8 | MedQA | Keyword-Match | 0.0000 | 0.00179 | ✓ |
| 9 | MMLU-Med | NLI-DeBERTa | 0.0000 | 0.00185 | ✓ |
| 10 | MMLU-Med | LLM-Judge | 0.0000 | 0.00192 | ✓ |
| 11 | MMLU-Med | Keyword-Match | 0.0000 | 0.00200 | ✓ |
| 12 | SciFact | SelfCheckGPT-NLI | 0.0000 | 0.00208 | ✓ |
| 13 | SciFact | HHEM | 0.0000 | 0.00217 | ✓ |
| 14 | SciFact | LLM-Judge | 0.0000 | 0.00227 | ✓ |
| 15 | SciFact | Keyword-Match | 0.0000 | 0.00238 | ✓ |
| 16 | SciFact | Random | 0.0000 | 0.00250 | ✓ |
| 17 | SciFact | Always-Positive | 0.0000 | 0.00263 | ✓ |
| 18 | PubMedQA | NLI-DeBERTa | 0.4490 | 0.00278 | — |
| 19 | PubMedQA | Keyword-Match | 0.8070 | 0.00294 | — |
| 20 | MedHallu | SelfCheckGPT-NLI | 0.8820 | 0.00313 | — |
| 21 | MedHallu | NLI-DeBERTa | 0.9720 | 0.00333 | — |
| 22 | SciFact | NLI-DeBERTa | 0.9950 | 0.00357 | — |
| 23 | MMLU-Med | Random | 0.9970 | 0.00385 | — |
| 24 | PubMedQA | Always-Positive | 0.9980 | 0.00417 | — |
| 25 | PubMedQA | Random | 0.9990 | 0.00455 | — |
| 26 | MedQA | SelfCheckGPT-NLI | 1.0000 | 0.00500 | — |
| 27 | MedQA | HHEM | 1.0000 | 0.00556 | — |
| 28 | MedQA | Random | 1.0000 | 0.00625 | — |
| 29 | MedQA | Always-Positive | 1.0000 | 0.00714 | — |
| 30 | MMLU-Med | SelfCheckGPT-NLI | 1.0000 | 0.00833 | — |
| 31 | MMLU-Med | HHEM | 1.0000 | 0.01000 | — |
| 32 | MMLU-Med | Always-Positive | 1.0000 | 0.01250 | — |
| 33 | PubMedQA | SelfCheckGPT-NLI | 1.0000 | 0.01667 | — |
| 34 | PubMedQA | HHEM | 1.0000 | 0.02500 | — |
| 35 | PubMedQA | LLM-Judge | 1.0000 | 0.05000 | — |


## AUC-PR family

Family size m = 35.  alpha = 0.05.  **15/35** comparisons survive Holm-Bonferroni.

| Rank | Dataset | Baseline | Raw p | Holm threshold | Survive? |
| ---:| --- | --- | ---:| ---:| :---:|
| 1 | MedHallu | LLM-Judge | 0.0000 | 0.00143 | ✓ |
| 2 | MedHallu | Keyword-Match | 0.0000 | 0.00147 | ✓ |
| 3 | MedHallu | Random | 0.0000 | 0.00152 | ✓ |
| 4 | MedHallu | Always-Positive | 0.0000 | 0.00156 | ✓ |
| 5 | PubMedQA | HHEM | 0.0000 | 0.00161 | ✓ |
| 6 | PubMedQA | Keyword-Match | 0.0000 | 0.00167 | ✓ |
| 7 | PubMedQA | Random | 0.0000 | 0.00172 | ✓ |
| 8 | PubMedQA | Always-Positive | 0.0000 | 0.00179 | ✓ |
| 9 | SciFact | SelfCheckGPT-NLI | 0.0000 | 0.00185 | ✓ |
| 10 | SciFact | HHEM | 0.0000 | 0.00192 | ✓ |
| 11 | SciFact | Keyword-Match | 0.0000 | 0.00200 | ✓ |
| 12 | SciFact | Random | 0.0000 | 0.00208 | ✓ |
| 13 | SciFact | Always-Positive | 0.0000 | 0.00217 | ✓ |
| 14 | MedHallu | HHEM | 0.0010 | 0.00227 | ✓ |
| 15 | SciFact | LLM-Judge | 0.0020 | 0.00238 | ✓ |
| 16 | MMLU-Med | Keyword-Match | 0.0110 | 0.00250 | — |
| 17 | MMLU-Med | Always-Positive | 0.0450 | 0.00263 | — |
| 18 | MedQA | Always-Positive | 0.3300 | 0.00278 | — |
| 19 | MedQA | Keyword-Match | 0.3450 | 0.00294 | — |
| 20 | MedQA | HHEM | 0.4620 | 0.00313 | — |
| 21 | MMLU-Med | Random | 0.5580 | 0.00333 | — |
| 22 | MedQA | NLI-DeBERTa | 0.6110 | 0.00357 | — |
| 23 | MedQA | LLM-Judge | 0.6470 | 0.00385 | — |
| 24 | MedQA | SelfCheckGPT-NLI | 0.6600 | 0.00417 | — |
| 25 | MMLU-Med | HHEM | 0.7150 | 0.00455 | — |
| 26 | MedQA | Random | 0.8430 | 0.00500 | — |
| 27 | PubMedQA | NLI-DeBERTa | 0.9330 | 0.00556 | — |
| 28 | MMLU-Med | NLI-DeBERTa | 0.9400 | 0.00625 | — |
| 29 | MMLU-Med | SelfCheckGPT-NLI | 0.9730 | 0.00714 | — |
| 30 | PubMedQA | LLM-Judge | 0.9750 | 0.00833 | — |
| 31 | MedHallu | NLI-DeBERTa | 0.9770 | 0.01000 | — |
| 32 | PubMedQA | SelfCheckGPT-NLI | 0.9970 | 0.01250 | — |
| 33 | MedHallu | SelfCheckGPT-NLI | 1.0000 | 0.01667 | — |
| 34 | MMLU-Med | LLM-Judge | 1.0000 | 0.02500 | — |
| 35 | SciFact | NLI-DeBERTa | 1.0000 | 0.05000 | — |


