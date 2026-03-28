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
│  └─ bertalign.jsonl
├─ Task2/
│  └─ termalign.jsonl
└─ Task3/
   ├─ term_dict.json
   └─ evaluation.json

logs/
└─ *.log
```

默认 `pipeline/runner.py` 会按这个结构找文件；也可以用 `--data-dir` 或显式参数覆盖。
每次运行会写日志到 `logs/`，也可以用 `--logs-dir` 指定目录。

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

---

### A. bertalign 步骤

```bash
python bertalign_step/run_bertalign.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --output-file data/Task1/bertalign.jsonl \
  --default-confidence 0.85
```

如果你希望强制使用官方 bertalign，可传入 `--external-command`，命令中支持占位符：`{src}` `{tgt}` `{out}`。

```bash
python bertalign_step/run_bertalign.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --output-file data/Task1/bertalign.jsonl \
  --external-command "python your_bertalign_runner.py --src {src} --tgt {tgt} --out {out}"
```

### B. termalign 步骤（模型模式）

```bash
python termalign_step/run_termalign.py \
  --bertalign-output data/Task1/bertalign.jsonl \
  --output-file data/Task2/termalign.jsonl \
  --extraction-mode model \
  --aligner-model owen4512/minilm-finance-term-aligner \
  --zh-extractor-model owen4512/bert-base-chinese-finance-term-extractor \
  --en-extractor-model owen4512/bert-base-cased-finance-term-extractor \
  --min-term-confidence 0.5 \
  --min-pair-confidence 0.5
```

### C. termalign 步骤（API 模式）

```bash
python termalign_step/run_termalign.py \
  --bertalign-output data/Task1/bertalign.jsonl \
  --output-file data/Task2/termalign.jsonl \
  --extraction-mode api \
  --api-endpoint https://your-api/term-extract \
  --api-key YOUR_KEY
```

### D. evaluation 步骤

```bash
python evaluation_step/evaluate_terms.py \
  --termalign-output data/Task2/termalign.jsonl \
  --dictionary-path data/Task3/term_dict.json \
  --output-file data/Task3/evaluation.json \
  --min-confidence 0.4 \
  --smoothing-alpha 0.1 \
  --normalize-entropy
```

核心加权逻辑：
- 对每个翻译变体 `v`，有效次数 `effective_count(v) = count(v) * confidence(v)`
- 默认当 `count_field` 不提供时，`count(v)=1`，等价于“每次出现按置信度计权”

---

## Prefect 总流程运行

```bash
python pipeline/prefect_flow.py \
  --source-file data/Task1/source.txt \
  --target-file data/Task1/target.txt \
  --dictionary-path data/Task3/term_dict.json \
  --bertalign-output data/Task1/bertalign.jsonl \
  --termalign-output data/Task2/termalign.jsonl \
  --evaluation-output data/Task3/evaluation.json \
  --extraction-mode model \
  --min-term-confidence 0.5 \
  --min-pair-confidence 0.5 \
  --eval-min-confidence 0.4 \
  --eval-smoothing-alpha 0.1 \
  --eval-normalize-entropy
```

---


## 额外统一入口（pipeline runner）

如果你希望通过**一个脚本**来运行全流程或任意子步骤，可使用：

```bash
python pipeline/runner.py full
# 默认读写 data/Task1, data/Task2, data/Task3
```

- 默认 `full` 是顺序执行（不依赖 Prefect 服务端部署）。
- 如果想强制走 Prefect 编排，可加 `--use-prefect`。
- 如果你把数据放在别处，可传 `--data-dir /path/to/your-data-root`。
- 日志目录可通过 `--logs-dir /path/to/logs` 指定。
- 也可以单独跑子命令：

```bash
python pipeline/runner.py bertalign --source-file ... --target-file ... --bertalign-output ...
python pipeline/runner.py termalign --bertalign-output ... --termalign-output ...
python pipeline/runner.py evaluation --termalign-output ... --dictionary-path ... --evaluation-output ...
```

---

## 可调参数（重点）

- 路径参数：输入、各阶段输出路径
- 术语抽取方式：`--extraction-mode model|api`
- 术语抽取阈值：`--min-term-confidence`
- 术语配对阈值：`--min-pair-confidence`
- 统计阈值：`--eval-min-confidence`
- 加权字段：`--eval-confidence-field`、`--eval-count-field`
- 熵参数：`--eval-entropy-base`、`--eval-normalize-entropy`
- 可选平滑：`--eval-smoothing-alpha`
- 统计截断：`--eval-top-k-variants`
