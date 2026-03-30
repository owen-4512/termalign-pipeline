# Data Layout (Task1 / Task2 / Task3 / results)

按你的要求，`data/` 目录按任务阶段组织：

- `data/Task1/source.txt`：源语言文本（每行一个句段）
- `data/Task1/target.txt`：目标语言文本（每行一个句段）
- `data/Task2/bertalign.jsonl`：Task2（bertalign）输出，同时作为 termalign 输入
- `data/Task2/batch_tsv/*.tsv`：bertalign 批量输出目录（pipeline batch 模式）
- `data/Task1/*_{lang}.txt`：如果使用批量对齐，文件名需满足 `YYYY_ID_lang.txt`
- `data/Task3/high_confidence/all_alignments_high_conf.tsv`：Task3（termalign）给 evaluation 使用的高置信汇总输出
- `data/Task3/alignment_details/<bertalign_stem>_terms_zh.tsv`：每个文件的中文术语输出（batch）
- `data/Task3/alignment_details/<bertalign_stem>_terms_en.tsv`：每个文件的英文术语输出（batch）
- `data/Task3/alignment_details/<bertalign_stem>_alignments.tsv`：每个文件的全部术语配对（batch）
- `data/Task3/alignment_details/<bertalign_stem>_alignments_high_conf.tsv`：每个文件的高置信术语配对（batch）
- `data/Task3/alignment_details/all_terms_zh.tsv`：所有文件中文术语汇总
- `data/Task3/alignment_details/all_terms_en.tsv`：所有文件英文术语汇总
- `data/Task3/alignment_details/all_alignments.tsv`：所有文件术语配对汇总
- `data/Task3/alignment_details/all_alignments_high_conf.tsv`：所有文件高置信术语配对汇总（明细目录副本）
- `data/Task2/dict_zh.txt`：termalign 中文术语词典（可选）
- `data/Task2/dict_en.txt`：termalign 英文术语词典（可选）
- `data/Task2/termalign_prompt.txt`：termalign API 模式 prompt 文本（可选）
- `data/Task3/proper_terms.jsonl`：evaluation 词典（键中文，值英文翻译）推荐输入
- `data/results/evaluation_result.json`：evaluation 默认输出（pipeline CLI 未显式指定时）

补充：如果不传 termalign 的词典参数，termalign 子项目会自动尝试读取：  
`termalign_step/term_list/zh_terms.txt` 与 `termalign_step/term_list/en_terms.txt`。

`pipeline/runner.py` 默认按该布局读取/写入；你也可以通过 `--data-dir` 切换根目录，或显式传参数覆盖默认路径。
