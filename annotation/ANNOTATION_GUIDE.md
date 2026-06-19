# MedHal-Loc 标注指南（试标版 v0.1）

## 你的任务
在每条 **`hallucinated_answer`（幻觉答案）** 里，标出**使它成为"幻觉"的最小错误片段（error span）**——即相对 `evidence`（证据）和 `ground_truth`（正确答案）而言，**事实上错误或无依据**的那部分文字。

## 怎么标（在 `pilot_A.csv` / `pilot_B.csv` 里）
每行一条样本。读 `evidence` + `ground_truth` 判断对错，然后：
1. 把幻觉答案里的**错误文字原样复制**到 `span1_text`（**必须逐字一致**，脚本要靠它定位；不要改写、不要加引号）。
2. 在 `span1_type` 填**错误类型**（见下表，填英文小写代号）。
3. 一条答案有多处独立错误 → 用 `span2_*`、`span3_*`（最多 3 处）。
4. 实在找不到可定位的错误片段（纯属遗漏/信息不全、没有具体错词）→ 三个 span 留空，在 `notes` 写 `no-locatable-span`。
5. 只标**事实错误**；不要标语法、语气、措辞风格。

## 6 种错误类型
| 代号 | 含义 | 例 |
|---|---|---|
| `entity_substitution` | 实体被换错（药物/疾病/数值/剂量/解剖位置）| "appointed in **2017**"（实为 2016）|
| `relation_error` | 实体都对，但**关系**错 | "Metformin **causes** diabetes"（实为 treats）|
| `mechanism_misattribution` | 机制/通路张冠李戴 | "regulates perforations through **calcium channels**"（证据无此机制）|
| `invented` | 凭空捏造、证据完全无依据 | "activating **specific proteases that degrade...**" |
| `contradictory` | 与证据**直接矛盾** | 证据说"显著更好"，答案说"**comparable / 无差异**" |
| `unverifiable` | 听起来合理但**证据无法证实/证伪** | "shows **slightly higher** risk"（证据未提及）|

类型有重叠时，选**最贴近错误本质**的那个。`mechanism_misattribution` 优先于 `invented`（当错的是"机制"而非"凭空新实体"）。

## 关键规则
- **最小充分**：只标导致错误的核心词组，不要把整句都框进去。
- **逐字复制**：`spanN_text` 必须是幻觉答案里的连续子串（脚本用字符串匹配定位，差一个字符就定位失败）。
- **独立错误分开标**：两个互不相关的错 → 两个 span；同一个错的不同说法 → 一个 span。
- **以证据为准**：判断对错只看 `evidence` + `ground_truth`，不要用你的外部知识脑补。

## 两人独立标
A、B 两人**各自独立**标完 `pilot_A.csv` / `pilot_B.csv`（不要互相看），然后跑
`python src/compute_agreement.py` 算一致性（span-F1 + 类型 Cohen's κ）。
- **κ ≥ 0.6 且 span-F1 ≥ 0.5** → schema 可行，进入全量 200–300 条标注。
- **κ < 0.4** → 错误片段太主观，先一起过分歧、收紧本指南，或退到"实体级"粒度再评。
