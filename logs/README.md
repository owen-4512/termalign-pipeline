# Pipeline Logs

该目录用于记录 `pipeline/runner.py` 的运行日志。

- 默认目录：`logs/`
- 默认日志名：`<command>-YYYYMMDD-HHMMSS.log`
- 可通过 `--logs-dir` 自定义目录

示例：

```bash
python pipeline/runner.py full --logs-dir logs
python pipeline/runner.py termalign --logs-dir logs
```
