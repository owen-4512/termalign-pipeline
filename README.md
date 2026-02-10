# Term Translation Evaluation Pipeline

`term_eval_pipeline.py` 提供一个可直接运行的术语翻译评估流程，支持：

- Accuracy（F1 / Precision / Recall）
- Consistency（基于术语译法分布熵）
- Shortest Distance Penalty（同一术语不同译法在目标文本中的最短距离惩罚）

最终分数：

```text
final_score = f1 - alpha * consistency - beta * distance_penalty
```

## 输入

1. **term align TSV**，包含字段（至少）：
   - `source_file`
   - `zh_term`
   - `en_term`
2. **gold dictionary JSONL**，每行一个 JSON object，建议字段：
   - 术语字段：`zh_term`（也兼容 `source_term`/`term`）
   - 参考翻译字段：`en_terms`（list）或 `en_term`（string）
3. **目标译文 txt**
   - simple 模式：单个 `--target-txt`
   - batch 模式：`--target-dir`（按 `source_file` 文件名匹配）

## 用法

### Simple

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode simple \
  --target-txt data/target.txt \
  --alpha 0.2 \
  --beta 0.1
```

### Batch

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode batch \
  --target-dir data/targets \
  --alpha 0.2 \
  --beta 0.1
```

程序输出 JSON，包括：`f1`, `precision`, `recall`, `consistency`, `distance_penalty`, `final_score`。

## 指标说明（实现口径）

- Accuracy：逐 occurrence 判断 extracted variant 是否命中 gold 参考译法（归一化后比较）。
- Consistency：按 source term 统计译法分布熵（Shannon entropy, log2），再取均值。
- Distance penalty：对于同一 source term 的不同译法，在 token 序列中计算最短距离，惩罚定义为 `1 - d/len(tokens)`，最后对所有 term 取均值。

