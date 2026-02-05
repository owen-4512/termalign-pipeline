# termalign-pipeline

这个仓库提供一个简单的术语对齐评估脚本，完成三件事：

1. 从原始 JSONL 中提取 `proper` 字段（中文术语 -> 英文术语）并输出为单独 JSONL。
2. 使用预测 TSV（第 0 列中文术语、第 1 列英文术语）对比 gold JSONL，计算中文、英文、对齐三类指标。
3. 明确给出 precision / recall / F1 的计算公式与统计口径。

---

## 1. 环境

- Python 3.9+（仅标准库，无第三方依赖）

---

## 2. 数据格式

### 2.1 输入 gold JSONL

每行是一个 JSON 对象，并包含 `proper` 字段（dict）：

```json
{"id": 1, "proper": {"高血压": "hypertension", "糖尿病": "diabetes"}}
{"id": 2, "proper": {"心电图": "electrocardiogram"}}
```

### 2.2 提取后的 proper JSONL

脚本会输出如下格式（每行一个术语对）：

```json
{"zh": "高血压", "en": "hypertension"}
{"zh": "糖尿病", "en": "diabetes"}
{"zh": "心电图", "en": "electrocardiogram"}
```

### 2.3 预测 TSV

- 第 0 列：中文术语
- 第 1 列：英文术语

例如：

```tsv
高血压	hypertension
心电图	ecg
```

---

## 3. 功能一：提取 proper 为独立 JSONL

命令：

```bash
python termalign_metrics.py extract-proper \
  --input data/gold.jsonl \
  --output data/proper_only.jsonl
```

可选参数：

- `--field`：指定术语映射字段名，默认 `proper`。

---

## 4. 功能二：计算 precision / recall / f1

命令：

```bash
python termalign_metrics.py evaluate \
  --pred-tsv data/pred.tsv \
  --gold-jsonl data/gold.jsonl \
  --output data/metrics.json
```

若不传 `--output`，指标会直接打印到终端。

---

## 5. 计算过程与公式（详细）

记：

- 预测术语对集合（按行计数）为 `PredPairs`，总数 `|PredPairs|`
- gold 的 `proper` 映射为 `GoldMap: zh -> en`
- gold 中文术语集合为 `GoldZH`
- gold 英文术语集合为 `GoldEN`
- gold 术语对集合为 `GoldPairs = {(zh, en)}`

其中：

- `|GoldZH|` = gold 中去重后的中文术语数量
- `|GoldEN|` = gold 中去重后的英文术语数量
- `|GoldPairs|` = gold 中术语对数量（按 `(zh,en)` 去重后计数）

> 脚本的判定与你描述一致：对每个预测项做 0/1 命中判断，再汇总。

### 5.1 中文术语命中

对每个预测行 `(zh_i, en_i)`：

- 若 `zh_i ∈ GoldZH`，记 1
- 否则记 0

累计得到：

- `ZH_correct = Σ 1[zh_i ∈ GoldZH]`

指标：

- `ZH_precision = ZH_correct / |PredPairs|`
- `ZH_recall = ZH_correct / |GoldZH|`
- `ZH_f1 = 2 * ZH_precision * ZH_recall / (ZH_precision + ZH_recall)`（分母为 0 时记 0）

### 5.2 英文术语命中

对每个预测行 `(zh_i, en_i)`：

- 若 `en_i ∈ GoldEN`，记 1
- 否则记 0

累计得到：

- `EN_correct = Σ 1[en_i ∈ GoldEN]`

指标：

- `EN_precision = EN_correct / |PredPairs|`
- `EN_recall = EN_correct / |GoldEN|`
- `EN_f1 = 2 * EN_precision * EN_recall / (EN_precision + EN_recall)`（分母为 0 时记 0）

### 5.3 对齐（alignment）命中

对每个预测行 `(zh_i, en_i)`：

- 若 `zh_i` 在 gold 中存在，且 `en_i` 出现在该 `zh_i` 对应的 gold 值中（即 `(zh_i, en_i) ∈ GoldPairs`），记 1
- 否则记 0

累计得到：

- `ALIGN_correct = Σ 1[(zh_i, en_i) ∈ GoldPairs]`

指标：

- `ALIGN_precision = ALIGN_correct / |PredPairs|`
- `ALIGN_recall = ALIGN_correct / |GoldPairs|`
- `ALIGN_f1 = 2 * ALIGN_precision * ALIGN_recall / (ALIGN_precision + ALIGN_recall)`（分母为 0 时记 0）

---

## 6. 输出结果说明

`evaluate` 输出 JSON 结构如下：

- `counts`
  - `pred_pairs`: 预测术语对总数
  - `gold_zh_terms`: gold 中文术语数（去重）
  - `gold_en_terms`: gold 英文术语数（去重）
  - `gold_align_pairs`: gold 对齐术语对数
  - `zh_correct`: 中文命中数
  - `en_correct`: 英文命中数
  - `alignment_correct`: 对齐命中数
- `zh / en / alignment`
  - `precision`
  - `recall`
  - `f1`

---

## 7. 注意事项

1. TSV 至少需要两列，否则报错。
2. JSONL 中 `proper` 必须是对象（dict），否则报错。
3. 空行会自动跳过。
4. 当前实现按“逐行计数”计算 precision；recall 的分母分别是 gold 的中文集合、英文集合、对齐对集合大小。
