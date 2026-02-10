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
2. gold dictionary JSONL（每行一个 JSON object）支持两种格式：
   - 字段格式：`{"zh_term": "术语", "en_terms": ["translation"]}`（兼容 `source_term`/`term`/`source` + `en_term`）
   - 映射格式：`{"术语A": ["译法1", "译法2"], "术语B": ["译法"]}`
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

### 1) 跑全部指标（默认，输出 batch 聚合分数）

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

### 4) 输出 document 级别 + batch 级别

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode batch \
  --target-dir data/targets \
  --report-level both \
  --metrics all
```

`--report-level` 可选：
- `document`：仅输出每个 `source_file` 的分数
- `batch`：仅输出全量聚合分数（默认）
- `both`：同时输出 document 与 batch

## 输出

输出 JSON 会包含：

- 元信息：`mode`, `report_level`, `metrics`, `alpha`, `beta`, `num_records`
- 当 `report_level=batch|both`：`batch_score`
- 当 `report_level=document|both`：`document_scores`（列表，每个元素对应一个 `source_file`）

各 score 对象字段按选择的指标动态出现，例如：
- `f1`, `precision`, `recall`（accuracy）
- `consistency`
- `distance_penalty`
- `final_score`（仅当 `--metrics all`）

