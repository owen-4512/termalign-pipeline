# termalign_step data

- `inputs/`: put bertalign output files and optional dictionaries
- `outputs/`: termalign output files (TSV), including:
  - single-file: `terms_zh.tsv`, `terms_en.tsv`, `alignments.tsv`, `alignments_high_conf.tsv`
  - batch per-file: `<name>_terms_zh.tsv`, `<name>_terms_en.tsv`, `<name>_alignments.tsv`, `<name>_alignments_high_conf.tsv`
  - batch merged: `all_terms_zh.tsv`, `all_terms_en.tsv`, `all_alignments.tsv`, `all_alignments_high_conf.tsv`
