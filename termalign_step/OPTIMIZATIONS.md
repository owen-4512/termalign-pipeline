# Implemented optimizations

1. **Dictionary regex precompilation** in `termalign_pipeline/extractors.py` to avoid per-sentence `re.compile` overhead.
2. **Batch embedding inference** in `termalign_pipeline/align.py` (tokenizer/model forward in batches).
3. **Pipeline-level micro-optimizations** in `termalign_pipeline/pipeline.py`:
   - precompiled newline regex,
   - cached OpenCC conversions when building alignment rows,
   - avoided unnecessary repeated overlap checks/allocations,
   - streamlined `zip(...)` sentence-pair alignment loop.
4. Added `termalign_pipeline/io_utils.py` with lean TSV/dictionary IO utilities used by pipeline.

## Suggested next optimizations
- Add async/concurrent API requests for API mode.
- Replace pandas-heavy organize path with streaming CSV aggregation in run-termalign wrappers.
- Add cache manifest for HF model snapshots.
