# Implemented optimizations

1. **Dictionary regex precompilation** in `termalign_pipeline/extractors.py` to avoid per-sentence `re.compile` overhead.
2. **Batch embedding inference** in `termalign_pipeline/align.py` (tokenizer/model forward in batches).
3. Preserved fallback behavior for sentence-transformers / hash-embedding to keep runtime robust.

## Suggested next optimizations
- Add async/concurrent API requests for API mode.
- Replace pandas-heavy organize path with streaming CSV aggregation.
- Add cache manifest for HF model snapshots.
