#!/usr/bin/env python3
"""Compatibility wrapper that uses evaluation_pipeline.run_evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


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
                        "similarity": obj.get("weighted_confidence", obj.get("model_pair_confidence", 1.0)),
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
    mode: str = "batch",
    target_txt: str | None = None,
    target_dir: str | None = None,
    report_level: str = "batch",
    metrics: list[str] | None = None,
    alpha: float = 0.2,
    debug_sublogs_dir: str | None = None,
) -> str:
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
            mode=mode,
            metrics=metrics or ["all"],
            alpha=alpha,
            target_txt=Path(target_txt) if target_txt else None,
            target_dir=Path(target_dir) if target_dir else None,
            report_level=report_level,
            include_debug=True,
            include_cross_document_consistency=True,
        )

    batch_score = result.get("batch_score", {}) if isinstance(result.get("batch_score", {}), dict) else {}
    summary = {
        "accuracy": batch_score.get("accuracy"),
        "consistency": batch_score.get("consistency"),
        "final_score": batch_score.get("final_score"),
    }
    print(json.dumps(summary, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep only per-metric sublogs, no metrics.json output.
    sublogs_dir = Path(debug_sublogs_dir) if debug_sublogs_dir else (out_path.parent / "metrics_sublogs")
    sublogs_dir.mkdir(parents=True, exist_ok=True)
    debug_obj = result.get("debug", {})
    if isinstance(debug_obj, dict):
        if "accuracy" in debug_obj:
            (sublogs_dir / "accuracy.json").write_text(
                json.dumps(debug_obj["accuracy"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if "consistency" in debug_obj:
            (sublogs_dir / "consistency.json").write_text(
                json.dumps(debug_obj["consistency"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation step")
    parser.add_argument("--termalign-output", required=True)
    parser.add_argument("--dictionary-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--mode", choices=["simple", "batch"], default="batch")
    parser.add_argument("--target-txt", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--report-level", choices=["document", "batch", "both"], default="batch")
    parser.add_argument("--metrics", nargs="+", default=["all"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--debug-sublogs-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_terms(
        termalign_output=args.termalign_output,
        dictionary_path=args.dictionary_path,
        output_file=args.output_file,
        mode=args.mode,
        target_txt=args.target_txt,
        target_dir=args.target_dir,
        report_level=args.report_level,
        metrics=args.metrics,
        alpha=args.alpha,
        debug_sublogs_dir=args.debug_sublogs_dir,
    )


if __name__ == "__main__":
    main()
