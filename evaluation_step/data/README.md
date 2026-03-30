# evaluation_step data

- `inputs/`: put termalign output and evaluation dictionary files
- `outputs/`: evaluation result files and debug logs, e.g.
  - `evaluation_result.json`
  - `metrics.json`
  - `metrics_sublogs/accuracy.json`
  - `metrics_sublogs/consistency.json`

`term_eval_pipeline.py` 的 CLI 标准输出会打印精简 JSON，仅包含：
- `precision`
- `consistency`
- `final_score`
