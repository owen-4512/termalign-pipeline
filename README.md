# TermAlign Pipeline

This project extracts Chinese and English terms from bilingual text using **custom dictionaries** and **fine-tuned BERT token-classification models**, then aligns bilingual terms with an embedding model.

## Features
- Dictionary-first extraction (label: `dict`).
- BERT token-classification extraction (label: `bert`) with confidence scores.
- No deduplication by default at raw extraction stage; language-specific post-filters are then applied.
- Alignment via multilingual embeddings (default: `./multi-embedding`).
- Supports **single TSV** input or **batch folder** input.

---

## Input Format
A TSV file (or folder of TSV files) with **two required columns**:

- `src_text`: Chinese sentence (Traditional Chinese allowed; converted to Simplified internally for extraction).
- `tgt_text`: English sentence.

Example (`data/input.tsv`):

```tsv
src_text	tgt_text
本產品支援術語擷取。	This product supports term extraction.
```

---

## Dictionary Format
A newline-separated text file, one term per line.

Example:

```txt
financial markets
black swans
credit loss
```

> Notes:
> - English dictionary matching is **whole-word + case-sensitive**.
> - Chinese dictionary matching is direct literal matching.

---

## BERT Models: What They Do
You can provide one model for Chinese and one for English:

- `--bert-model-zh`: Fine-tuned Chinese token-classification model for term extraction.
- `--bert-model-en`: Fine-tuned English token-classification model for term extraction.

### How BERT Extraction Works
1. Sentence is tokenized by the tokenizer bundled with the model path (`AutoTokenizer.from_pretrained(model_path)`).
2. Model outputs token logits (`AutoModelForTokenClassification`).
3. Softmax is applied over labels for each token.
4. Label with max probability is selected per token.
5. Tokens with `B-*` / `I-*` (also compatible with plain `B` / `I`) are grouped into term spans.
6. Span text is reconstructed from original sentence offsets.

### Confidence Definition
For each extracted BERT term span:

- Let the span contain token-level selected-label probabilities: `p1, p2, ..., pn`.
- Confidence is computed as the arithmetic mean:

`confidence = (p1 + p2 + ... + pn) / n`

This is exactly what the code does in `_flush`:
- `confidence = float(sum(scores) / max(len(scores), 1))`.

For dictionary hits, confidence is empty (`None` in code, blank in TSV).

---

## Alignment Model: What It Does
`--embed-model` provides a model used **only for zh-en term alignment**, not extraction.

- Default: `multi-embedding` (local folder in repo root).
- You can override with a local path or a Hugging Face model id.

### Alignment Procedure
1. Encode zh terms and en terms into embeddings.
2. L2-normalize embeddings.
3. Compute cosine similarity matrix.
4. For each zh term, choose best en match.
5. Alignment is restricted to terms from the **same input sentence pair**.

---

## CLI Arguments (Detailed)

```bash
python -m termalign.cli [OPTIONS]
```

### Required Arguments
- `--input`
  - Type: `str` (path)
  - Meaning: Single TSV file OR folder containing multiple TSV files.
  - Input columns must include `src_text` and `tgt_text`.

- `--output-dir`
  - Type: `str` (path)
  - Meaning: Output directory for result TSV files.

### Optional Arguments
- `--dict-zh`
  - Type: `str` (path)
  - Meaning: Chinese dictionary file (one term per line).
  - If omitted: Chinese dictionary matching is disabled.

- `--dict-en`
  - Type: `str` (path)
  - Meaning: English dictionary file (one term per line).
  - If omitted: English dictionary matching is disabled.

- `--bert-model-zh`
  - Type: `str` (path or repo id)
  - Meaning: Chinese fine-tuned BERT token-classification model.
  - If omitted: No Chinese BERT extraction.

- `--bert-model-en`
  - Type: `str` (path or repo id)
  - Meaning: English fine-tuned BERT token-classification model.
  - If omitted: No English BERT extraction.

- `--embed-model`
  - Type: `str` (path or repo id)
  - Default: `multi-embedding`
  - Meaning: Embedding model used for zh-en alignment.

- `--skip-bert`
  - Type: flag (no value)
  - Meaning: Disable all BERT extraction (dictionary-only extraction).

---

## Usage Examples

### 1) Single-file processing

```bash
python -m termalign.cli \
  --input data/input.tsv \
  --dict-zh dict/proper_terms_sc.txt \
  --dict-en dict/proper_terms_en.txt \
  --bert-model-zh model_zh_term \
  --bert-model-en model_en_term \
  --embed-model multi-embedding \
  --output-dir outputs
```

### 2) Batch processing (folder of TSV files)

```bash
python -m termalign.cli \
  --input data/batch_inputs \
  --dict-zh dict/proper_terms_sc.txt \
  --dict-en dict/proper_terms_en.txt \
  --bert-model-zh model_zh_term \
  --bert-model-en model_en_term \
  --output-dir outputs
```

### 3) Dictionary-only mode

```bash
python -m termalign.cli \
  --input data/input.tsv \
  --dict-zh dict/proper_terms_sc.txt \
  --dict-en dict/proper_terms_en.txt \
  --skip-bert \
  --output-dir outputs
```

### 4) BERT-only mode

```bash
python -m termalign.cli \
  --input data/input.tsv \
  --bert-model-zh model_zh_term \
  --bert-model-en model_en_term \
  --output-dir outputs
```

---

## Output Files

## Single-file input
- `terms_zh.tsv`
- `terms_en.tsv`
- `alignments.tsv`
- `alignments_high_conf.tsv` (similarity > 0.5)

## Batch input (folder)
For each file `<name>.tsv`:
- `<name>_terms_zh.tsv`
- `<name>_terms_en.tsv`
- `<name>_alignments.tsv`
- `<name>_alignments_high_conf.tsv`

Merged outputs across all input files:
- `all_terms_zh.tsv`
- `all_terms_en.tsv`
- `all_alignments.tsv`
- `all_alignments_high_conf.tsv`

### Output Columns

#### `terms_zh.tsv` / `terms_en.tsv`
- `source_file`: source TSV filename
- `term`: extracted term text
- `source`: `dict` or `bert`
- `confidence`: BERT confidence (blank for dict)
- `sentence`: original sentence where term occurs

#### `alignments.tsv` / `alignments_high_conf.tsv`
- `source_file`: source TSV filename
- `zh_term`, `en_term`
- `similarity`
- `zh_source`, `en_source`
- `zh_confidence`, `en_confidence`
- `zh_sentence`, `en_sentence`

---

## Important Processing Rules
- Chinese input is converted Traditional → Simplified for extraction, then converted back to Traditional in Chinese outputs.
- English preprocessing replaces escaped/newline tokens with spaces and collapses whitespace.
- English terms with length <= 2 are dropped.
- Chinese single-character terms are dropped.
- For nested/overlapping terms in the same sentence, only longer non-overlapping terms are kept.
- Dictionary spans are prioritized over identical BERT spans.
- Alignment is strictly within each paired input sentence (no cross-sentence alignment).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Troubleshooting

### 1) Model load error (e.g. `mode` not found)
If you mistype a model path (`mode` vs `model_en`), loading fails. Check:
- local folder exists and contains model/tokenizer files
- or Hugging Face repo id is correct and accessible

### 2) Unexpected matching from dictionary
- English dictionary now uses whole-word + case-sensitive matching.
- Ensure dictionary entries are exactly the form you want extracted.

### 3) Why confidence is blank for some terms?
Those are dictionary hits (`source=dict`), not BERT hits.
