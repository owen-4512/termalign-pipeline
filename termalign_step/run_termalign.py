#!/usr/bin/env python3
"""Run termalign step using the provided termalign pipeline scripts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

def _jsonl_to_tsv(input_jsonl: Path, output_tsv: Path) -> str:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append({"src_text": row["source_segment"], "tgt_text": row["target_segment"]})
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def _alignment_tsv_to_jsonl(align_tsv: Path, output_jsonl: Path, min_similarity: float, top_k_pairs: int) -> str:
    import pandas as pd

    df = pd.read_csv(align_tsv, sep="\t")
    if "similarity" in df.columns:
        df = df[df["similarity"].astype(float) >= min_similarity]
    df = df.sort_values(by="similarity", ascending=False)
    if top_k_pairs > 0:
        df = df.head(top_k_pairs)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            weighted_conf = float(row.get("similarity", 0.0))
            out = {
                "source_term": str(row.get("zh_term", "")),
                "target_term": str(row.get("en_term", "")),
                "model_pair_confidence": weighted_conf,
                "weighted_confidence": weighted_conf,
                "source_sentence": str(row.get("zh_sentence", "")),
                "target_sentence": str(row.get("en_sentence", "")),
                "zh_source": row.get("zh_source"),
                "en_source": row.get("en_source"),
                "zh_confidence": row.get("zh_confidence"),
                "en_confidence": row.get("en_confidence"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return str(output_jsonl)


def run_termalign(
    bertalign_output: str,
    output_file: str,
    extraction_mode: str = "model",
    min_term_confidence: float = 0.5,
    min_pair_confidence: float = 0.5,
    source_lang: str = "zh",
    target_lang: str = "en",
    top_k_pairs: int = 0,
    aligner_model: str = "owen4512/minilm-finance-term-aligner",
    zh_extractor_model: str = "owen4512/bert-base-chinese-finance-term-extractor",
    en_extractor_model: str = "owen4512/bert-base-cased-finance-term-extractor",
    api_endpoint: str | None = None,
    api_key: str | None = None,
    device: int = -1,
    dict_zh_path: str | None = None,
    dict_en_path: str | None = None,
    skip_bert: bool = False,
) -> str:
    del min_term_confidence, source_lang, target_lang, device

    if extraction_mode == "api":
        if not api_endpoint:
            raise ValueError("api_endpoint is required when extraction_mode='api'.")
        from termalign_step.run_termalign_api import run_termalign_api
        return run_termalign_api(
            bertalign_output=bertalign_output,
            output_file=output_file,
            api_endpoint=api_endpoint,
            api_key=api_key,
            min_pair_confidence=min_pair_confidence,
            top_k_pairs=top_k_pairs,
        )

    input_jsonl = Path(bertalign_output)
    output_jsonl = Path(output_file)

    with tempfile.TemporaryDirectory(prefix="termalign-") as tmpdir:
        from termalign_step.termalign_pipeline.pipeline import run_pipeline

        tmpdir_path = Path(tmpdir)
        input_tsv = tmpdir_path / "bertalign_input.tsv"
        output_dir = tmpdir_path / "termalign_output"

        _jsonl_to_tsv(input_jsonl, input_tsv)

        run_pipeline(
            input_path=input_tsv,
            dict_zh_path=dict_zh_path,
            dict_en_path=dict_en_path,
            bert_model_zh=zh_extractor_model,
            bert_model_en=en_extractor_model,
            embed_model=aligner_model,
            similarity_threshold=min_pair_confidence,
            output_dir=output_dir,
            skip_bert=skip_bert,
        )

        align_tsv = output_dir / "alignments.tsv"
        if not align_tsv.exists():
            align_tsv = output_dir / "all_alignments.tsv"

        return _alignment_tsv_to_jsonl(
            align_tsv=align_tsv,
            output_jsonl=output_jsonl,
            min_similarity=min_pair_confidence,
            top_k_pairs=top_k_pairs,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TermAlign step")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--extraction-mode", choices=["model", "api"], default="model")
    parser.add_argument("--min-term-confidence", type=float, default=0.5)
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--top-k-pairs", type=int, default=0)
    parser.add_argument("--aligner-model", default="owen4512/minilm-finance-term-aligner")
    parser.add_argument("--zh-extractor-model", default="owen4512/bert-base-chinese-finance-term-extractor")
    parser.add_argument("--en-extractor-model", default="owen4512/bert-base-cased-finance-term-extractor")
    parser.add_argument("--api-endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dict-zh-path", default=None)
    parser.add_argument("--dict-en-path", default=None)
    parser.add_argument("--skip-bert", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_termalign(
        bertalign_output=args.bertalign_output,
        output_file=args.output_file,
        extraction_mode=args.extraction_mode,
        min_term_confidence=args.min_term_confidence,
        min_pair_confidence=args.min_pair_confidence,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        top_k_pairs=args.top_k_pairs,
        aligner_model=args.aligner_model,
        zh_extractor_model=args.zh_extractor_model,
        en_extractor_model=args.en_extractor_model,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        device=args.device,
        dict_zh_path=args.dict_zh_path,
        dict_en_path=args.dict_en_path,
        skip_bert=args.skip_bert,
    )


if __name__ == "__main__":
    main()
