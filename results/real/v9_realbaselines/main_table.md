# AdaTriple Bootstrap Confidence Intervals

Bootstrap B = **1000**, threshold re-tuned on each resample (per-sample arrays pooled across all seeds before resampling).


## F1 with 95% CI (bootstrap)

| Method | medhallu | medqa | mmlu_medical | pubmedqa | scifact | Avg F1 |
| --- | --- | --- | --- | --- | --- | --- |
| AdaTriple (fixed_lambda) | 0.693 [0.676, 0.710] | 0.575 [0.555, 0.595] | 0.651 [0.636, 0.668] | 0.593 [0.574, 0.612] | 0.522 [0.489, 0.553] | 0.607 |
| AdaTriple (w/o KG) | 0.692 [0.674, 0.710] | 0.577 [0.558, 0.597] | 0.653 [0.638, 0.669] | 0.559 [0.538, 0.578] | 0.530 [0.497, 0.561] | 0.602 |
| AdaTriple (w/o NLI) | 0.690 [0.674, 0.706] | 0.602 [0.584, 0.621] | 0.652 [0.637, 0.669] | 0.606 [0.587, 0.624] | 0.467 [0.430, 0.504] | 0.603 |
| AdaTriple+ | 0.691 [0.675, 0.708] | 0.568 [0.549, 0.589] | 0.650 [0.634, 0.666] | 0.600 [0.581, 0.619] | 0.536 [0.503, 0.566] | 0.609 |
| Always-Positive | 0.667 [0.651, 0.683] | 0.667 [0.652, 0.681] | 0.667 [0.652, 0.681] | 0.618 [0.601, 0.635] | 0.342 [0.323, 0.362] | 0.592 |
| HHEM | 0.667 [0.652, 0.683] | 0.667 [0.652, 0.681] | 0.667 [0.652, 0.681] | 0.625 [0.608, 0.643] | 0.344 [0.323, 0.364] | 0.594 |
| Keyword-Match | 0.630 [0.614, 0.647] | 0.004 [0.000, 0.009] | 0.420 [0.396, 0.445] | 0.606 [0.589, 0.623] | 0.342 [0.323, 0.362] | 0.401 |
| LLM-Judge | 0.614 [0.595, 0.635] | 0.203 [0.179, 0.226] | 0.555 [0.533, 0.575] | 0.623 [0.604, 0.641] | 0.394 [0.361, 0.425] | 0.478 |
| NLI-DeBERTa | 0.696 [0.679, 0.713] | 0.398 [0.374, 0.421] | 0.621 [0.603, 0.640] | 0.600 [0.581, 0.618] | 0.560 [0.533, 0.586] | 0.575 |
| Random | 0.661 [0.645, 0.677] | 0.661 [0.646, 0.676] | 0.661 [0.646, 0.676] | 0.619 [0.600, 0.636] | 0.343 [0.322, 0.364] | 0.589 |
| SelfCheckGPT-NLI | 0.700 [0.682, 0.719] | 0.666 [0.650, 0.680] | 0.665 [0.651, 0.680] | 0.623 [0.605, 0.641] | 0.401 [0.376, 0.426] | 0.611 |

## AUC-PR with 95% CI (bootstrap)

| Method | medhallu | medqa | mmlu_medical | pubmedqa | scifact | Avg F1 |
| --- | --- | --- | --- | --- | --- | --- |
| AdaTriple (fixed_lambda) | 0.674 [0.649, 0.702] | 0.504 [0.480, 0.528] | 0.519 [0.492, 0.545] | 0.553 [0.526, 0.580] | 0.438 [0.397, 0.479] | 0.538 |
| AdaTriple (w/o KG) | 0.669 [0.643, 0.696] | 0.505 [0.481, 0.528] | 0.525 [0.498, 0.550] | 0.548 [0.522, 0.576] | 0.435 [0.394, 0.476] | 0.536 |
| AdaTriple (w/o NLI) | 0.665 [0.638, 0.692] | 0.499 [0.478, 0.521] | 0.499 [0.479, 0.519] | 0.553 [0.526, 0.580] | 0.387 [0.347, 0.426] | 0.521 |
| AdaTriple+ | 0.672 [0.646, 0.700] | 0.505 [0.480, 0.530] | 0.516 [0.488, 0.543] | 0.551 [0.523, 0.579] | 0.432 [0.391, 0.473] | 0.535 |
| Always-Positive | 0.500 [0.483, 0.518] | 0.500 [0.483, 0.517] | 0.500 [0.483, 0.517] | 0.448 [0.430, 0.466] | 0.207 [0.192, 0.221] | 0.431 |
| HHEM | 0.623 [0.596, 0.649] | 0.503 [0.479, 0.530] | 0.522 [0.496, 0.547] | 0.485 [0.460, 0.512] | 0.274 [0.247, 0.303] | 0.481 |
| Keyword-Match | 0.514 [0.490, 0.538] | 0.501 [0.484, 0.517] | 0.493 [0.472, 0.514] | 0.426 [0.402, 0.451] | 0.207 [0.192, 0.221] | 0.428 |
| LLM-Judge | 0.596 [0.569, 0.623] | 0.508 [0.485, 0.535] | 0.604 [0.577, 0.629] | 0.585 [0.558, 0.611] | 0.361 [0.323, 0.400] | 0.531 |
| NLI-DeBERTa | 0.679 [0.655, 0.706] | 0.508 [0.482, 0.533] | 0.533 [0.507, 0.558] | 0.556 [0.531, 0.583] | 0.529 [0.489, 0.566] | 0.561 |
| Random | 0.514 [0.490, 0.539] | 0.518 [0.494, 0.543] | 0.518 [0.494, 0.543] | 0.460 [0.434, 0.484] | 0.214 [0.192, 0.237] | 0.445 |
| SelfCheckGPT-NLI | 0.742 [0.717, 0.765] | 0.509 [0.486, 0.536] | 0.537 [0.512, 0.564] | 0.592 [0.564, 0.619] | 0.326 [0.291, 0.361] | 0.541 |

## F1 across seeds: mean ± std

| Method | medhallu | medqa | mmlu_medical | pubmedqa | scifact | Avg F1 |
| --- | --- | --- | --- | --- | --- | --- |
| AdaTriple (fixed_lambda) | 0.692 ± 0.004 (n=3) | 0.575 ± 0.002 (n=3) | 0.651 ± 0.002 (n=3) | 0.594 ± 0.000 (n=3) | 0.524 ± 0.034 (n=3) | 0.607 |
| AdaTriple (w/o KG) | 0.691 ± 0.004 (n=3) | 0.577 ± 0.002 (n=3) | 0.653 ± 0.001 (n=3) | 0.560 ± 0.000 (n=3) | 0.529 ± 0.034 (n=3) | 0.602 |
| AdaTriple (w/o NLI) | 0.689 ± 0.004 (n=3) | 0.602 ± 0.004 (n=3) | 0.653 ± 0.001 (n=3) | 0.606 ± 0.000 (n=3) | 0.466 ± 0.046 (n=3) | 0.603 |
| AdaTriple+ | 0.690 ± 0.003 (n=3) | 0.568 ± 0.002 (n=3) | 0.650 ± 0.002 (n=3) | 0.601 ± 0.000 (n=3) | 0.534 ± 0.025 (n=3) | 0.609 |
| Always-Positive | 0.667 ± 0.000 (n=3) | 0.667 ± 0.000 (n=3) | 0.667 ± 0.000 (n=3) | 0.619 ± 0.000 (n=3) | 0.342 ± 0.008 (n=3) | 0.592 |
| HHEM | 0.667 ± 0.000 (n=3) | 0.667 ± 0.000 (n=3) | 0.667 ± 0.000 (n=3) | 0.624 ± 0.000 (n=3) | 0.344 ± 0.008 (n=3) | 0.594 |
| Keyword-Match | 0.630 ± 0.008 (n=3) | 0.004 ± 0.000 (n=3) | 0.421 ± 0.012 (n=3) | 0.607 ± 0.000 (n=3) | 0.342 ± 0.008 (n=3) | 0.401 |
| LLM-Judge | 0.614 ± 0.011 (n=3) | 0.203 ± 0.011 (n=3) | 0.556 ± 0.005 (n=3) | 0.624 ± 0.001 (n=3) | 0.395 ± 0.023 (n=3) | 0.478 |
| NLI-DeBERTa | 0.695 ± 0.002 (n=3) | 0.397 ± 0.009 (n=3) | 0.622 ± 0.004 (n=3) | 0.601 ± 0.000 (n=3) | 0.563 ± 0.019 (n=3) | 0.575 |
| Random | 0.660 ± 0.004 (n=3) | 0.661 ± 0.001 (n=3) | 0.661 ± 0.001 (n=3) | 0.619 ± 0.001 (n=3) | 0.345 ± 0.010 (n=3) | 0.589 |
| SelfCheckGPT-NLI | 0.701 ± 0.005 (n=3) | 0.665 ± 0.002 (n=3) | 0.666 ± 0.001 (n=3) | 0.623 ± 0.000 (n=3) | 0.399 ± 0.006 (n=3) | 0.611 |

## Paired bootstrap one-sided p-values (F1)

H0: AdaTriple+ ≤ baseline.  p < 0.05 = significant win.


### medhallu

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.972 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.594 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.301 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.855 |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.882 |
| AdaTriple+ vs HHEM | 0.000 * |
| AdaTriple+ vs LLM-Judge | 0.000 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.000 * |
| AdaTriple+ vs Always-Positive | 0.000 * |

### medqa

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.000 * |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.977 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 1.000 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.974 |
| AdaTriple+ vs SelfCheckGPT-NLI | 1.000 |
| AdaTriple+ vs HHEM | 1.000 |
| AdaTriple+ vs LLM-Judge | 0.000 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 1.000 |
| AdaTriple+ vs Always-Positive | 1.000 |

### mmlu_medical

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.000 * |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.942 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.883 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.814 |
| AdaTriple+ vs SelfCheckGPT-NLI | 1.000 |
| AdaTriple+ vs HHEM | 1.000 |
| AdaTriple+ vs LLM-Judge | 0.000 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.997 |
| AdaTriple+ vs Always-Positive | 1.000 |

### pubmedqa

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.449 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.000 * |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.881 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.021 * |
| AdaTriple+ vs SelfCheckGPT-NLI | 1.000 |
| AdaTriple+ vs HHEM | 1.000 |
| AdaTriple+ vs LLM-Judge | 1.000 |
| AdaTriple+ vs Keyword-Match | 0.807 |
| AdaTriple+ vs Random | 0.999 |
| AdaTriple+ vs Always-Positive | 0.998 |

### scifact

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.995 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.143 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.000 * |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.009 * |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.000 * |
| AdaTriple+ vs HHEM | 0.000 * |
| AdaTriple+ vs LLM-Judge | 0.000 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.000 * |
| AdaTriple+ vs Always-Positive | 0.000 * |

## Paired bootstrap one-sided p-values (AUC-PR)

H0: AdaTriple+ ≤ baseline.  p < 0.05 = AdaTriple+ significantly better-ranked.  Note: large p (e.g. > 0.95) on the AdaTriple+ vs NLI-DeBERTa row would mean NLI-DeBERTa is *significantly better* (one-sided); this is the strongest test of the AUC-PR trade-off.


### medhallu

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.977 (loss) |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.052 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.019 * |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.943 |
| AdaTriple+ vs SelfCheckGPT-NLI | 1.000 (loss) |
| AdaTriple+ vs HHEM | 0.001 * |
| AdaTriple+ vs LLM-Judge | 0.000 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.000 * |
| AdaTriple+ vs Always-Positive | 0.000 * |

### medqa

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.611 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.490 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.276 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.371 |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.660 |
| AdaTriple+ vs HHEM | 0.462 |
| AdaTriple+ vs LLM-Judge | 0.647 |
| AdaTriple+ vs Keyword-Match | 0.345 |
| AdaTriple+ vs Random | 0.843 |
| AdaTriple+ vs Always-Positive | 0.330 |

### mmlu_medical

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.940 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.980 (loss) |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.032 * |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.830 |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.973 (loss) |
| AdaTriple+ vs HHEM | 0.715 |
| AdaTriple+ vs LLM-Judge | 1.000 (loss) |
| AdaTriple+ vs Keyword-Match | 0.011 * |
| AdaTriple+ vs Random | 0.558 |
| AdaTriple+ vs Always-Positive | 0.045 * |

### pubmedqa

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 0.933 |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.155 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.715 |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.767 |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.997 (loss) |
| AdaTriple+ vs HHEM | 0.000 * |
| AdaTriple+ vs LLM-Judge | 0.975 (loss) |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.000 * |
| AdaTriple+ vs Always-Positive | 0.000 * |

### scifact

| Comparison | p-value |
| --- | --- |
| AdaTriple+ vs NLI-DeBERTa | 1.000 (loss) |
| AdaTriple+ vs AdaTriple (w/o KG) | 0.842 |
| AdaTriple+ vs AdaTriple (w/o NLI) | 0.000 * |
| AdaTriple+ vs AdaTriple (fixed_lambda) | 0.897 |
| AdaTriple+ vs SelfCheckGPT-NLI | 0.000 * |
| AdaTriple+ vs HHEM | 0.000 * |
| AdaTriple+ vs LLM-Judge | 0.002 * |
| AdaTriple+ vs Keyword-Match | 0.000 * |
| AdaTriple+ vs Random | 0.000 * |
| AdaTriple+ vs Always-Positive | 0.000 * |