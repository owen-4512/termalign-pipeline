# termalign-pipeline

这个仓库用于术语对齐处理，支持：

1. 从原始 JSONL 中提取 `proper` 字段（中文术语 -> 英文术语）到单独 JSONL。
2. 使用预测 TSV（第 0 列中文术语、第 1 列英文术语）计算中文、英文、alignment 三类 precision/recall/F1。

---

## 1. 环境

- Python 3.9+（仅标准库）

---

## 2. 输入格式

### 2.1 原始 gold JSONL（含 `proper`）

每行是一个 JSON 对象；`proper` 为字典，键是中文术语，值支持：

- 单个英文术语字符串
- 英文术语列表

示例：

```json
{"id": 1, "proper": {"高血压": "hypertension", "糖尿病": ["diabetes", "diabetes mellitus"]}}
{"id": 2, "proper": {"心电图": ["electrocardiogram", "ECG"]}}
```

### 2.2 预测 TSV

- 第 0 列：中文术语
- 第 1 列：英文术语

```tsv
高血压	hypertension
心电图	ecg
```

---

## 3. 提取 proper

```bash
python termalign_metrics.py extract-proper \
  --input data/gold.jsonl \
  --output data/proper_only.jsonl
```

输出 JSONL 每行格式：

```json
{"zh": "高血压", "en": ["hypertension"]}
{"zh": "糖尿病", "en": ["diabetes", "diabetes mellitus"]}
```

可选参数：

- `--field`：自定义映射字段名，默认 `proper`。

---

## 4. 评估 evaluate

### 4.1 使用原始 gold JSONL

```bash
python termalign_metrics.py evaluate \
  --pred-tsv data/pred.tsv \
  --gold-jsonl data/gold.jsonl
```

### 4.2 使用提取后的 proper JSONL

```bash
python termalign_metrics.py evaluate \
  --pred-tsv data/pred.tsv \
  --gold-proper-jsonl data/proper_only.jsonl
```

说明：`--gold-jsonl` 和 `--gold-proper-jsonl` 必须二选一。

### 4.3 输出分析过程 TSV

```bash
python termalign_metrics.py evaluate \
  --pred-tsv data/pred.tsv \
  --gold-jsonl data/gold.jsonl \
  --analysis-tsv data/analysis.tsv
```

`analysis.tsv` 包含 8 列（按你的要求）：

1. `pred_zh`：提取出的中文术语（预测第 0 列）
2. `matched_zh_key`：若该中文术语在 gold 键中存在，则输出该键；否则 `NAN`
3. `zh_score`：中文术语命中得分（1/0）
4. `pred_en`：提取出的英文术语（预测第 1 列）
5. `matched_en_value`：若该英文术语在 gold 值集合中存在，则输出该值；否则 `NAN`
6. `en_score`：英文术语命中得分（1/0）
7. `correct_alignment_en`：以第一列中文术语为基础，若 alignment 正确则列出该中文术语在 gold 中的正确英文术语（多个用 `|` 连接）；不正确则 `NAN`
8. `alignment_score`：alignment 命中得分（1/0）

---

## 5. 计算过程与公式

记：

- 预测术语对（逐行）为 `PredPairs = [(zh_i, en_i)]`，总数 `|PredPairs|`
- gold 中文术语集合 `GoldZH`
- gold 英文术语集合 `GoldEN`
- gold 对齐集合 `GoldPairs = {(zh, en)}`

其中：

- `|GoldZH|`：gold 去重后的中文术语数
- `|GoldEN|`：gold 去重后的英文术语数
- `|GoldPairs|`：gold 去重后的 `(zh,en)` 对数

> 判定方式是逐个预测样本做 0/1 命中，再汇总。

### 5.1 中文术语

对每个预测 `(zh_i, en_i)`：

- 若 `zh_i ∈ GoldZH`，记 1；否则 0。

累计：

- `ZH_correct = Σ 1[zh_i ∈ GoldZH]`

指标：

- `ZH_precision = ZH_correct / |PredPairs|`
- `ZH_recall = ZH_correct / |GoldZH|`
- `ZH_f1 = 2 * ZH_precision * ZH_recall / (ZH_precision + ZH_recall)`（分母 0 记 0）

### 5.2 英文术语

对每个预测 `(zh_i, en_i)`：

- 若 `en_i ∈ GoldEN`，记 1；否则 0。

累计：

- `EN_correct = Σ 1[en_i ∈ GoldEN]`

指标：

- `EN_precision = EN_correct / |PredPairs|`
- `EN_recall = EN_correct / |GoldEN|`
- `EN_f1 = 2 * EN_precision * EN_recall / (EN_precision + EN_recall)`（分母 0 记 0）

### 5.3 Alignment（中英配对）

对每个预测 `(zh_i, en_i)`：

- 若 `(zh_i, en_i) ∈ GoldPairs`，记 1；否则 0。
- 当某个中文术语在 gold 里对应多个英文术语时，只要预测英文术语在该中文的 gold 值集合中，即视为命中。

累计：

- `ALIGN_correct = Σ 1[(zh_i, en_i) ∈ GoldPairs]`

指标：

- `ALIGN_precision = ALIGN_correct / |PredPairs|`
- `ALIGN_recall = ALIGN_correct / |GoldPairs|`
- `ALIGN_f1 = 2 * ALIGN_precision * ALIGN_recall / (ALIGN_precision + ALIGN_recall)`（分母 0 记 0）

---

## 6. 输出字段

`evaluate` 输出 JSON 包括：

- `counts`
  - `pred_pairs`
  - `gold_zh_terms`
  - `gold_en_terms`
  - `gold_align_pairs`
  - `zh_correct`
  - `en_correct`
  - `alignment_correct`
- `zh` / `en` / `alignment`
  - `precision`
  - `recall`
  - `f1`

---

## 7. 注意事项

1. TSV 每行至少两列，否则报错。
2. JSONL 每行必须是 JSON 对象。
3. `proper` 必须是字典；其值支持字符串或列表。
4. 空行会自动跳过。
