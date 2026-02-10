# Term Translation Evaluation Project

这个项目把 term translation evaluation 拆成可复用模块，并提供可配置 CLI。

## 项目结构

```text
term_eval_pipeline.py          # 兼容入口
requirements.txt
term_eval/
  __init__.py
  cli.py                       # CLI 参数解析与执行
  pipeline.py                  # 调度层（可选 metric）
  io_utils.py                  # TSV/JSONL/TXT 读取
  normalization.py             # 文本归一化
  data_model.py                # 数据转换与 token map
  metrics_accuracy.py          # compute_accuracy
  metrics_consistency.py       # compute_consistency
  metrics_distance.py          # compute_shortest_distance_penalty
```

## 支持指标

- `accuracy`：计算 `f1 / precision / recall`
- `consistency`：计算术语译法熵均值
- `distance`：计算 shortest distance penalty

当你选择 `all`（默认）时，额外输出：

```text
final_score = f1 - alpha * consistency - beta * distance_penalty
```

## 输入

1. term align TSV（至少包含）
   - `source_file`
   - `zh_term`
   - `en_term`
2. gold dictionary JSONL（每行一个 JSON object）
   - 术语字段：`zh_term`（兼容 `source_term`/`term`/`source`）
   - 参考翻译字段：`en_terms`（list）或 `en_term`（string）
3. 目标译文
   - simple：`--target-txt`
   - batch：`--target-dir`

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 当前实现仅使用 Python 标准库。

## CLI 用法

### 1) 跑全部指标（默认）

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode simple \
  --target-txt data/target.txt \
  --alpha 0.2 \
  --beta 0.1
```

### 2) 只跑单个指标

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode simple \
  --target-txt data/target.txt \
  --metrics consistency
```

### 3) 跑多个指定指标

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode batch \
  --target-dir data/targets \
  --metrics accuracy distance
```

也支持逗号写法：`--metrics accuracy,distance`。

## 输出

输出 JSON，字段按选择的指标动态出现，例如：

- `f1`, `precision`, `recall`（accuracy）
- `consistency`
- `distance_penalty`
- `final_score`（仅当 metrics=all）

