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
```

## 支持指标

- `accuracy`：计算 `precision`（先用提取出的中文术语对齐 gold；若术语不在 gold 中则跳过不计分。对齐后：若提取译法包含 gold 译法则得 1 分；若 gold 译法包含提取译法则按 token 比例得分，如 `risk assessment` 对 `cybersecurity risk assessment` 得 `2/3`）
- `consistency`：计算术语译法熵均值（仅按出现分布统计，不考虑该译法是否准确）

当你选择 `all`（默认）时，额外输出：

```text
final_score = precision - alpha * consistency
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

需要安装 spaCy 用于标准化（tokenization + lemmatization）。

可选安装英文模型（推荐）：

```bash
python -m spacy download en_core_web_sm
```

## CLI 用法

### 1) 跑全部指标（默认，输出 batch 聚合分数）

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode simple \
  --target-txt data/target.txt \
  --alpha 0.2
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
  --metrics accuracy consistency
```

也支持逗号写法：`--metrics accuracy,consistency`。

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


### 5) 输出详细 debug 日志

```bash
python term_eval_pipeline.py \
  --term-align-tsv data/align.tsv \
  --gold-jsonl data/gold.jsonl \
  --mode batch \
  --target-dir data/targets \
  --report-level both \
  --metrics all \
  --alpha 0.2 \
  --debug-log debug_metrics.json
```

`debug_metrics.json` 中会记录：
- accuracy：每个术语 occurrence 的 gold 候选、单项得分（可为 0~1 的小数）
- consistency：每个中文术语的译法计数、概率分布、entropy
- 额外元信息：record 数、source term 数、所选 metrics 等

## 输出

CLI 标准输出会打印一个精简 JSON，仅包含：

- `precision`
- `consistency`
- `final_score`

如果你传入 `--output-json`，文件中会保存完整结果 JSON（含元信息、document/batch 结构等）。

完整结果 JSON 会包含：

- 元信息：`mode`, `report_level`, `metrics`, `alpha`, `beta`, `num_records`
- 当 `report_level=batch|both`：`batch_score`
- 当 `report_level=document|both`：`document_scores`（列表，每个元素对应一个 `source_file`）

各 score 对象字段按选择的指标动态出现，例如：
- `precision`（accuracy）
- `consistency`
- `final_score`（仅当 `--metrics all`）
