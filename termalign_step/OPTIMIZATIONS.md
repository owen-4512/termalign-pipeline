# Implemented optimizations

1. **Dictionary regex precompilation** in `termalign_pipeline/extractors.py` to avoid per-sentence `re.compile` overhead.
2. **Batch embedding inference** in `termalign_pipeline/align.py` (tokenizer/model forward in batches).
3. **Pipeline-level micro-optimizations** in `termalign_pipeline/pipeline.py`:
   - precompiled newline regex,
   - cached OpenCC conversions when building alignment rows,
   - avoided unnecessary repeated overlap checks/allocations,
   - streamlined `zip(...)` sentence-pair alignment loop.
4. Added `termalign_pipeline/io_utils.py` with lean TSV/dictionary IO utilities used by pipeline.
5. Replaced pandas-heavy IO in `io_utils.py` with csv streaming readers/writers to reduce memory overhead on large batches.
6. Further optimized `pipeline.py` hot loops with lighter dict-span keys (`(start,end)`), `defaultdict` grouping, and fast-path skip when no dictionary spans exist.
7. Optimized `run_termalign.py` HF model path with cached runtime-version checks, concurrent model snapshot downloads, and streaming JSONL→TSV conversion.
8. Optimized `run_termalign_api.py` with concurrent request dispatch (`--workers`), shared session reuse, and lower-overhead term-row materialization.
9. Optimized `runner_pipeline_api.py` by reusing `runner.build_parser()` and forcing API-mode defaults instead of duplicating CLI definitions.

## Suggested next optimizations
- Add async/concurrent API requests for API mode.
- Replace pandas-heavy organize path with streaming CSV aggregation in run-termalign wrappers.
- Add cache manifest for HF model snapshots.
