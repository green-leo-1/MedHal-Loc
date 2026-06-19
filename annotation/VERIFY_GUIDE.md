# MedHal-Loc 人工核验指南（50 条锚点，勾选式 · 约半天）

**你不用从头标。** LLM 已经把错误 span 标好填在表里了，你只需**判断它标得对不对**。

## 准备
把 `annotation/verify_50.csv` 复制成两份：`verify_50_H1.csv`（核验者 1）、`verify_50_H2.csv`（核验者 2）。两人**各自独立**填，不要互看。

## 每行怎么填
读 `evidence`（证据）+ `source_text`（正确原文）判断，然后对 `hallucinated_text` 里 LLM 标出的 `llm_span_1`（必要时还有 `llm_span_2`）打分：

| 列 | 填什么 |
|---|---|
| `span1_ok(Y/N/P)` | **Y**=这个 span 确实是错误且位置对；**P**=部分对（位置偏了/范围不准）；**N**=根本不是错误，或标错地方 |
| `type1_ok(Y/N)` | LLM 给的错误类型（`llm_type_1`）对不对：**Y**/**N** |
| `span2_*` | 若有第 2 个 span，同样判定（没有就留空）|
| `missed_error` | 如果 LLM **漏标**了一个真错误：把那段错误文字**逐字复制**进来 |
| `missed_type` | 漏标错误的类型（6 类之一）|
| `notes` | 任何疑难备注 |

**6 种错误类型**：`entity_substitution`（实体换错）、`relation_error`（关系错）、`mechanism_misattribution`（机制张冠李戴）、`invented`（凭空捏造）、`contradictory`（与证据矛盾）、`unverifiable`（证据无法证实/证伪）。

## 判定原则
- 只看 `evidence` + `source_text` 判断对错，不要用外部知识脑补。
- `controlled` 子集：每条只有 1 个注入错误，核验它是否**干净、唯一、可定位**（Y）。
- `natural` 子集：真实幻觉，可能多错或弥散；如实判 Y/P/N + 补漏标。

## 填完后
```
python src/compute_validation.py
```
直接给出：①受控注入 gold 的**通过率**；②自然子集的 **LLM-人 span 精确率 + 类型一致率**（这就是论文要写的"LLM 标注经人工认证"的数字）。

**达标线**：受控通过率 ≥ 90%、自然 LLM-人 精确率 ≥ 80% → gold 可信度达标，可写进论文。
