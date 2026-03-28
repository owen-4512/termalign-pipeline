#!/usr/bin/env python3
"""Compatibility wrapper that uses evaluation_pipeline.run_evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

def _jsonl_to_termalign_tsv(termalign_jsonl: Path, out_tsv: Path) -> Path:
    import pandas as pd

    rows = []
    with termalign_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append(
                    {
                        "source_file": obj.get("source_file", "__default__"),
                        "zh_term": obj.get("source_term", ""),
                        "en_term": obj.get("target_term", ""),
                    }
                )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    return out_tsv


def _dict_json_to_gold_jsonl(dict_json: Path, out_jsonl: Path) -> Path:
    data = json.loads(dict_json.read_text(encoding="utf-8"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for zh_term, refs in data.items():
            refs_list = refs if isinstance(refs, list) else [refs]
            f.write(json.dumps({"zh_term": zh_term, "en_terms": refs_list}, ensure_ascii=False) + "\n")
    return out_jsonl


def evaluate_terms(
    termalign_output: str,
    dictionary_path: str,
    output_file: str,
    min_confidence: float = 0.0,
    confidence_field: str = "weighted_confidence",
    count_field: str | None = None,
    smoothing_alpha: float = 0.5,
    entropy_base: float = 2.0,
    normalize_entropy: bool = False,
    top_k_variants: int | None = None,
) -> str:
    del min_confidence, confidence_field, count_field, entropy_base, normalize_entropy, top_k_variants

    termalign_input = Path(termalign_output)
    dict_json = Path(dictionary_path)
    out_path = Path(output_file)

    with tempfile.TemporaryDirectory(prefix="eval-") as tmp:
        from evaluation_step.evaluation_pipeline import run_evaluation

        tmpdir = Path(tmp)
        if termalign_input.suffix.lower() == ".tsv":
            termalign_tsv = termalign_input
        else:
            termalign_tsv = _jsonl_to_termalign_tsv(termalign_input, tmpdir / "termalign.tsv")

        if dict_json.suffix.lower() == ".jsonl":
            gold_jsonl = dict_json
        else:
            gold_jsonl = _dict_json_to_gold_jsonl(dict_json, tmpdir / "gold.jsonl")

        result = run_evaluation(
            term_align_tsv=termalign_tsv,
            gold_jsonl=gold_jsonl,
            mode="batch",
            metrics=["all"],
            alpha=smoothing_alpha,
            beta=0.0,
            report_level="batch",
            include_debug=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation step")
    parser.add_argument("--termalign-output", required=True)
    parser.add_argument("--dictionary-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--confidence-field", default="weighted_confidence")
    parser.add_argument("--count-field", default=None)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--entropy-base", type=float, default=2.0)
    parser.add_argument("--normalize-entropy", action="store_true")
    parser.add_argument("--top-k-variants", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_terms(
        termalign_output=args.termalign_output,
        dictionary_path=args.dictionary_path,
        output_file=args.output_file,
        min_confidence=args.min_confidence,
        confidence_field=args.confidence_field,
        count_field=args.count_field,
        smoothing_alpha=args.smoothing_alpha,
        entropy_base=args.entropy_base,
        normalize_entropy=args.normalize_entropy,
        top_k_variants=args.top_k_variants,
    )


if __name__ == "__main__":
    main()
