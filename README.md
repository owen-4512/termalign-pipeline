# TermAlign Pipeline (Prefect)

用于“文件术语翻译打分”的三阶段 pipeline：

1. `bertalign`：句段对齐（支持外部 bertalign 命令，默认提供可运行的回退逐行对齐）
2. `termalign`：术语抽取 + 术语配对（支持 Hugging Face 模型或 API）
3. `evaluation`：基于自定义词典进行准确性、一致性和熵统计（支持按置信度加权）

---

## 安装

### 1) 安装总依赖（整条 pipeline 可运行）

```bash
pip install -r requirements.txt
```

### 2) 安装子项目依赖（每个步骤可单独运行）

```bash
pip install -r bertalign_step/requirements.txt
pip install -r termalign_step/requirements.txt
pip install -r evaluation_step/requirements.txt
```

---

## 数据格式

## 项目目录建议（统一管理输入输出）

```
data/
├─ Task1/
│  ├─ source.txt
│  ├─ target.txt
├─ Task2/
│  └─ bertalign.jsonl
├─ Task3/
│  ├─ all_alignments_high_conf.tsv
│  └─ term_dict.json
└─ results/
   └─ evaluation_result.json

logs/
└─ *.log
```

默认 `pipeline/runner.py` 会按这个结构找文件；也可以用 `--data-dir` 或显式参数覆盖。
每次运行会写日志到 `logs/`，也可以用 `--logs-dir` 指定目录。

---

## 子项目独立数据目录

因为每个子项目都可以单独运行，已分别提供独立数据目录：

- `bertalign_step/data/inputs`、`bertalign_step/data/outputs`
- `termalign_step/data/inputs`、`termalign_step/data/outputs`
- `evaluation_step/data/inputs`、`evaluation_step/data/outputs`

---

### 输入文件
- `source_file`: 源语言文本（每行一个句段）
- `target_file`: 目标语言文本（每行一个句段）

### 词典文件（JSON）

```json
{
  "equity": ["股权", "权益"],
  "bond": ["债券"]
}
```

---

## 单独运行各步骤

### 0) 在 pipeline 中接入官方 Bertalign（根据官网文档）

`bertalign` 官方 README 的基本方式是：`from bertalign import Bertalign`，初始化后执行 `align_sents()`（可选 `print_sents()`），并支持参数如 `max_align / top_k / win / skip / margin / len_penalty / is_split`。  
在本项目中要稳定使用官方 bertalign，建议做这几件事：

1. 安装 bertalign 及其依赖（本仓库已放到 `bertalign_step/requirements.txt`）：`bertalign`、`numba`、`faiss`、`sentence-transformers`、`sentence-splitter`、`googletrans`。  
2. 准备 Task1 输入文件：`data/Task1/source.txt` 和 `data/Task1/target.txt`。  
3. 如果你有自己的 bertalign 执行脚本（例如要显式传 `is_split=True` 或其他参数），在本项目里通过 `--external-command` 接入；命令可使用占位符 `{src}` `{tgt}` `{out}`。  
4. 如果不传 `--external-command`，本项目会使用回退对齐逻辑（逐行 1-1），用于快速跑通 pipeline（但效果不等价于官方 bertalign）。
5. 本仓库已内置两个可用脚本：  
   - `bertalign_step/single_align.py`：单文件对齐  
   - `bertalign_step/batch_align.py`：批量文件对齐（文件名模式 `YYYY_ID_lang.txt`）

---

### A. bertalign 步骤

#### A1. 单文件（推荐给 pipeline）

```bash
python bertalign_step/run_bertalign.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --output-file data/Task1/bertalign.jsonl \
  --default-confidence 0.85 \
  --max-align 3 \
  --top-k 5 \
  --win 8 \
  --src-lang zh \
  --tgt-lang en
```

如果你希望强制使用官方 bertalign，可传入 `--external-command`，命令中支持占位符：`{src}` `{tgt}` `{out}`。

```bash
python bertalign_step/run_bertalign.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --output-file data/Task1/bertalign.jsonl \
  --external-command "python your_bertalign_runner.py --src {src} --tgt {tgt} --out {out}"
```

#### A2. 批量文件（使用你给的 batch 脚本逻辑）

```bash
python bertalign_step/run_bertalign.py \
  --batch-data-dir data/Task1 \
  --batch-output-dir data/Task2/batch_tsv \
  --src-lang zh \
  --tgt-lang en \
  --max-align 3 \
  --top-k 5 \
  --win 8
```

或直接调用：

```bash
python bertalign_step/batch_align.py \
  --data-dir data/Task1 \
  --output-dir data/Task2/batch_tsv \
  --src-lang zh \
  --tgt-lang en
```

批量输出文件名会跟随输入文件名的基名：
- 输入：`2016_01_zh.txt` + `2016_01_en.txt`
- 输出：`2016_01_zh_en_align.tsv`

通用规则：对于任意 `<base>_{src_lang}.txt` 与 `<base>_{tgt_lang}.txt` 配对，输出为 `<base>_{src_lang}_{tgt_lang}_align.tsv`。

### B. termalign 步骤（模型模式）

`termalign_step/run_termalign.py` 现已对接你提供的完整 termalign 脚本体系（`align.py / extractors.py / io_utils.py / pipeline.py / cli.py`），执行流程是：  
`bertalign.jsonl -> 中间 TSV(src_text/tgt_text) -> termalign pipeline -> 输出 terms/alignment 系列 TSV（含 high_conf）`。

当 `--bertalign-output` 指向 **批量 bertalign 输出目录**（例如包含 `2016_01_zh_en_align.tsv`、`2016_02_zh_en_align.tsv` ...）时，
termalign 会在 `--output-dir` 下输出：
- 每个文件对应的中文术语：`<bertalign_stem>_terms_zh.tsv`
- 每个文件对应的英文术语：`<bertalign_stem>_terms_en.tsv`
- 每个文件对应的全部术语配对：`<bertalign_stem>_alignments.tsv`
- 每个文件对应的高置信术语配对：`<bertalign_stem>_alignments_high_conf.tsv`
- 以及总汇总文件：
  - `all_terms_zh.tsv`
  - `all_terms_en.tsv`
  - `all_alignments.tsv`
  - `all_alignments_high_conf.tsv`

其中 `<bertalign_stem>` 直接来自 bertalign 文件名（例如 `2016_01_zh_en_align`）。

另外，`termalign_step/term_list/` 下已提供默认术语表文件：  
- `zh_terms.txt`（中文术语）  
- `en_terms.txt`（英文术语）  
如果 CLI 未传 `--dict-zh-path/--dict-en-path`（或 termalign 子项目 CLI 未传 `--dict-zh/--dict-en`），会自动读取这两个 txt 文件（若存在）。

```bash
python termalign_step/run_termalign.py \
  --bertalign-output data/Task1/bertalign.jsonl \
  --output-file termalign_step/data/outputs/all_alignments_high_conf.tsv \
  --output-dir termalign_step/data/outputs \
  --extraction-mode model \
  --aligner-model owen4512/minilm-finance-term-aligner \
  --zh-extractor-model owen4512/bert-base-chinese-finance-term-extractor \
  --en-extractor-model owen4512/bert-base-cased-finance-term-extractor \
  --min-pair-confidence 0.5
```

### C. termalign 步骤（API 模式）

新增了专用 API 子脚本 `termalign_step/run_termalign_api.py`，并且 `run_termalign.py --extraction-mode api` 会自动调用它。

```bash
python termalign_step/run_termalign.py \
  --bertalign-output data/Task1/bertalign.jsonl \
  --output-file data/Task2/termalign.jsonl \
  --extraction-mode api \
  --api-endpoint https://your-api/term-align \
  --api-key YOUR_KEY
```

### D. evaluation 步骤

`evaluation_step/evaluate_terms.py` 现已对接 evaluation 全量脚本（`evaluation_pipeline/*`），会自动把当前 pipeline 的输入转成 evaluation 所需格式后计算 `accuracy + consistency + final_score`（并输出 debug 信息）。

如果 `termalign` 已输出 `all_alignments_high_conf.tsv`，evaluation 可以直接使用这个文件作为输入。

```bash
python evaluation_step/evaluate_terms.py \
  --termalign-output termalign_step/data/outputs/all_alignments_high_conf.tsv \
  --dictionary-path data/Task3/term_dict.json \
  --output-file data/Task3/evaluation.json \
  --mode batch \
  --report-level both \
  --metrics all \
  --alpha 0.2 \
  --debug-log evaluation_step/data/outputs/debug_metrics.json
```

核心加权逻辑：
- 默认从 termalign TSV 中读取 `similarity`（若缺失则回退到 `weighted_confidence / model_pair_confidence / confidence / 1.0`）作为每次对齐的权重
- accuracy：按权重计算加权平均（`weighted_score_sum / total_effective_weight`），低 similarity 对 precision 影响更小
- consistency：按权重聚合每个译法的有效次数，再计算比例与熵（`consistency = 1 - normalized_entropy`）

---

### D2. 输出详细 debug 日志（总 log + 子 log）

```bash
python evaluation_step/term_eval_pipeline.py \
  --term-align-tsv termalign_step/data/outputs/all_alignments_high_conf.tsv \
  --gold-jsonl data/Task3/gold.jsonl \
  --mode batch \
  --target-dir data/targets \
  --report-level both \
  --metrics all \
  --alpha 0.2 \
  --debug-log evaluation_step/data/outputs/debug_metrics.json \
  --output-json evaluation_step/data/outputs/evaluation_result_full.json
```

会输出：
- 一个总 log：`debug_metrics.json`
- 一个子 log 目录：默认 `debug_metrics_sublogs/`（可通过 `--debug-sublogs-dir` 指定）
  - `accuracy.json`
  - `consistency.json`

总 log `debug_metrics.json` 中会记录：
- accuracy：每个术语 occurrence 的 `gold_candidates`、单项得分（`score`，范围 0~1）
- consistency：统一分区；单文档时记录单文件术语明细，多文档时记录跨文件术语明细；多文档时 summary 额外含 `per_file_consistency_scores`
- 额外元信息：`num_records`、`num_source_terms`、`selected_metrics` 等

CLI 标准输出会打印精简 JSON，仅包含：
- `precision`
- `consistency`
- `final_score`

如果传入 `--output-json`，文件中会保存完整结果 JSON（含元信息、document/batch 结构等）。

完整结果 JSON 会包含：
- 元信息：`mode`, `report_level`, `metrics`, `alpha`, `beta`, `num_records`
- 当 `report_level=batch|both`：`batch_score`
- 当 `report_level=document|both`：`document_scores`（列表，每个元素对应一个 `source_file`）

各 score 对象字段按选择指标动态出现，例如：
- `precision`（accuracy）
- `consistency`
- `final_score`（仅当 `--metrics all`）

---

## Prefect 总流程运行

```bash
python pipeline/prefect_flow.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --dictionary-path data/Task3/term_dict.json \
  --bertalign-output data/Task2/bertalign.jsonl \
  --termalign-output data/Task3/all_alignments_high_conf.tsv \
  --evaluation-output data/results/evaluation_result.json \
  --extraction-mode model \
  --min-term-confidence 0.5 \
  --min-pair-confidence 0.5 \
  --eval-mode batch \
  --eval-report-level both \
  --eval-metrics all \
  --eval-alpha 0.2 \
  --eval-debug-log evaluation_step/data/outputs/debug_metrics.json
```

Prefect 也支持在总流程中启用 batch bertalign（后续自动进入 termalign + evaluation）：

```bash
python pipeline/prefect_flow.py \
  --dictionary-path data/Task3/term_dict.json \
  --bertalign-output data/Task2/bertalign.jsonl \
  --termalign-output data/Task3/all_alignments_high_conf.tsv \
  --evaluation-output data/results/evaluation_result.json \
  --bertalign-batch-data-dir data/Task1 \
  --bertalign-batch-output-dir data/Task2/batch_tsv \
  --extraction-mode model \
  --min-pair-confidence 0.5 \
  --eval-mode batch \
  --eval-report-level both \
  --eval-metrics all
```

说明（batch 全流程最终评分）：
- 该命令会完成：batch bertalign → batch termalign 聚合 → evaluation；
- evaluation CLI 会在终端打印精简分数 JSON（含 `final_score`）；
- 默认结果文件写入 `data/results/evaluation_result.json`（可用 `--evaluation-output` 覆盖）。

### Prefect/runner 之外的一键 API 全流程脚本

```bash
python pipeline/run_pipeline_api.py \
  --api-endpoint https://your-api/term-align \
  --api-key YOUR_KEY
```

---


## 额外统一入口（pipeline runner）

如果你希望通过**一个脚本**来运行全流程或任意子步骤，可使用：

```bash
python pipeline/runner.py full
# 默认读写 data/Task1, data/Task2, data/Task3, data/results
```

- 默认 `full` 是顺序执行（不依赖 Prefect 服务端部署）。
- 如果想强制走 Prefect 编排，可加 `--use-prefect`。
- 如果你把数据放在别处，可传 `--data-dir /path/to/your-data-root`。
- 日志目录可通过 `--logs-dir /path/to/logs` 指定。
- `termalign` 步骤新增统一模式参数：`--termalign-mode {api,local,hf}`  
  - `hf`：自动下载并使用你提供的 Hugging Face 模型（默认）  
    - `owen4512/bert-base-chinese-finance-term-extractor`（中文术语提取）  
    - `owen4512/bert-base-cased-finance-term-extractor`（英文术语提取）  
    - `owen4512/minilm-finance-term-aligner`（中英术语对齐）  
  - `local`：使用本地模型路径（通过 `--aligner-model / --zh-extractor-model / --en-extractor-model` 传入）  
  - `api`：使用 API 术语对齐（可传 GPT 模型名）
- 也可以单独跑子命令：

```bash
python pipeline/runner.py bertalign --source-file ... --target-file ... --bertalign-output ...
python pipeline/runner.py termalign --bertalign-output ... --termalign-output ...
python pipeline/runner.py evaluation --termalign-output ... --dictionary-path ... --evaluation-output ...
```

### 总 pipeline 中使用 bertalign 批量模式

如果 Task1 是一批 `*_zh.txt` / `*_en.txt` 文件（命名 `YYYY_ID_lang.txt`），可直接在总 pipeline 里开启 batch bertalign：

```bash
python pipeline/runner.py full \
  --bertalign-batch-data-dir data/Task1 \
  --bertalign-batch-output-dir data/Task2/batch_tsv \
  --termalign-output data/Task3/all_alignments_high_conf.tsv \
  --dictionary-path data/Task3/term_dict.json \
  --evaluation-output data/results/evaluation_result.json
```

说明：
- `--bertalign-batch-data-dir` 开启 batch bertalign；
- batch 结果会作为目录输入直接传给 termalign；
- termalign 会输出“每文件结果 + 总汇总结果”（文件名前缀沿用 bertalign 输出文件名）；
- evaluation 使用 `all_alignments_high_conf.tsv` 打分并产出 `final_score`。

### 总 pipeline CLI：termalign 模式示例

使用 Hugging Face 模型（默认）：

```bash
python pipeline/runner.py full \
  --termalign-mode hf
```

使用本地模型：

```bash
python pipeline/runner.py full \
  --termalign-mode local \
  --aligner-model /path/to/local/embed_model \
  --zh-extractor-model /path/to/local/zh_ner \
  --en-extractor-model /path/to/local/en_ner
```

使用 API（可选 GPT），并在后面传 API 与 prompt txt：

```bash
python pipeline/runner.py full \
  --termalign-mode api \
  --api-endpoint https://your-api/term-align \
  --api-key YOUR_KEY \
  --api-model gpt-4.1 \
  --api-prompt-file data/Task3/termalign_prompt.txt
```

---

## 可调参数（重点）

- 路径参数：输入、各阶段输出路径
- 术语抽取方式：`--extraction-mode model|api`
- 术语抽取阈值：`--min-term-confidence`
- 术语配对阈值：`--min-pair-confidence`
- 评估模式：`--eval-mode simple|batch`
- 输出粒度：`--eval-report-level document|batch|both`
- 指标选择：`--eval-metrics all|accuracy|consistency`
- 最终分数组合权重：`--eval-alpha`（`final_score = (1-alpha)*precision + alpha*consistency`）
- debug 输出：`--eval-debug-log`、`--eval-debug-sublogs-dir`
