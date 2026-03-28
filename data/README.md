# Data Layout

统一输入输出目录建议如下：

- `data/input/source.txt`：源语言文本（每行一个句段）
- `data/input/target.txt`：目标语言文本（每行一个句段）
- `data/input/term_dict.json`：术语词典
- `data/intermediate/bertalign.jsonl`：bertalign 中间结果
- `data/intermediate/termalign.jsonl`：termalign 中间结果
- `data/output/evaluation.json`：最终评估结果

`pipeline/runner.py` 默认会使用这套路径；你也可以通过 `--data-dir` 替换根目录，或传具体文件参数覆盖默认值。
