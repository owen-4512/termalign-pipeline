# Data Layout (Task1 / Task2 / Task3)

按你的要求，`data/` 目录按任务阶段组织：

- `data/Task1/source.txt`：源语言文本（每行一个句段）
- `data/Task1/target.txt`：目标语言文本（每行一个句段）
- `data/Task1/bertalign.jsonl`：Task1（bertalign）输出
- `data/Task1/*_{lang}.txt`：如果使用批量对齐，文件名需满足 `YYYY_ID_lang.txt`
- `termalign_step/data/outputs/all_alignments_high_conf.tsv`：Task2（termalign）推荐输出（供 evaluation 直接输入）
- `data/Task3/term_dict.json`：自定义术语词典
- `data/Task3/dict_zh.txt`：termalign 中文术语词典（可选）
- `data/Task3/dict_en.txt`：termalign 英文术语词典（可选）
- `data/Task3/termalign_prompt.txt`：termalign API 模式 prompt 文本（可选）
- `evaluation_step/data/outputs/evaluation_result.json`：Task3（evaluation）推荐输出

补充：如果不传 termalign 的词典参数，termalign 子项目会自动尝试读取：  
`termalign_step/term_list/zh_terms.txt` 与 `termalign_step/term_list/en_terms.txt`。

`pipeline/runner.py` 默认按该布局读取/写入；你也可以通过 `--data-dir` 切换根目录，或显式传参数覆盖默认路径。
