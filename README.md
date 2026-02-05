# TermAlign Pipeline

This project extracts Chinese and English terms from bilingual text using **custom dictionaries** and a **fine-tuned BERT NER model**, then aligns the bilingual term lists using an open-source BERT embedding model.

## Features
- Dictionary-first extraction (label: `dict`).
- BERT token-classification extraction (label: `bert`) with confidence scores.
- No deduplication: every occurrence is kept.
- Alignment via multilingual BERT embeddings (default: `./multi-embedding`).

## Input Format
A TSV file with **two columns**:
- `src_text`: Chinese sentence (Traditional Chinese accepted; will be converted to Simplified for extraction)
- `tgt_text`: English sentence

Example (`data/input.tsv`):
```tsv
src_text	tgt_text
本產品支援術語擷取。	This product supports term extraction.
```

## Dictionary Format
A newline-separated text file, one term per line.

## Usage
```bash
python -m termalign.cli \
  --input data/input.tsv \
  --dict-zh data/dict_zh.txt \
  --dict-en data/dict_en.txt \
  --bert-model-zh /path/to/zh-bert-ner \
  --bert-model-en /path/to/en-bert-ner \
  --output-dir outputs
```

### Outputs
1. `outputs/terms_zh.tsv` — Chinese terms (converted back to Traditional)
2. `outputs/terms_en.tsv` — English terms
3. `outputs/alignments.tsv` — Term alignment

Column definitions:
- `term`: extracted term
- `source`: `dict` or `bert`
- `confidence`: BERT confidence (blank for dict)
- `sentence`: sentence where the term occurs

Alignment columns:
- `zh_term`, `en_term`, `similarity`
- plus per-term metadata (source, confidence, sentence)

## Notes
- If you only have one BERT model, you can pass the same path for both `--bert-model-zh` and `--bert-model-en`.
- If you want to skip BERT extraction, pass `--skip-bert`.
- BERT extraction requires a fast tokenizer (most Hugging Face tokenizers are fast by default).
- The extractor always uses the tokenizer that is bundled with the fine-tuned model path you provide.
- Dictionary paths are optional; omit them to disable dictionary matching.
- English dictionary matching is case-insensitive and uses whole-word boundaries to avoid false positives from lowercase abbreviations.
- English term output ignores newlines, drops terms with length ≤ 2, and keeps only the longest overlapping term per sentence.
- Chinese term output keeps only the longest overlapping term per sentence and drops single-character terms.
- Alignment only pairs terms within the same input sentence pair (no cross-sentence alignment).
- The embedding model is loaded once and reused across sentence-pair alignments to avoid repeated reload warnings.
- Default alignment model path is `./multi-embedding` (override with `--embed-model`).
- `alignments_high_conf.tsv` includes only alignments with similarity > 0.5 (same columns as `alignments.tsv`).
- If a model path is mistyped (for example `mode` vs `model_en`), the pipeline now raises a clear load error with path/repo guidance.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
