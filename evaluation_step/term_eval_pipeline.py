#!/usr/bin/env python3
"""Standalone evaluation CLI with summary + debug logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Term evaluation pipeline")
    parser.add_argument("--term-align-tsv", type=Path, required=True)
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=["simple", "batch"], default="batch")
    parser.add_argument("--target-txt", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--report-level", choices=["document", "batch", "both"], default="batch")
    parser.add_argument("--metrics", nargs="+", default=["all"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--debug-log", type=Path, default=Path("evaluation_step/data/outputs/debug_metrics.json"))
    parser.add_argument("--debug-sublogs-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from evaluation_step.evaluation_pipeline import run_evaluation

    result = run_evaluation(
        term_align_tsv=args.term_align_tsv,
        gold_jsonl=args.gold_jsonl,
        mode=args.mode,
        metrics=args.metrics,
        alpha=args.alpha,
        beta=args.beta,
        target_txt=args.target_txt,
        target_dir=args.target_dir,
        report_level=args.report_level,
        include_debug=True,
        include_cross_document_consistency=True,
    )

    batch_score = result.get("batch_score", {}) if isinstance(result.get("batch_score", {}), dict) else {}
    summary = {
        "precision": batch_score.get("precision"),
        "consistency": batch_score.get("consistency"),
        "final_score": batch_score.get("final_score"),
    }
    print(json.dumps(summary, ensure_ascii=False))

    debug_log = args.debug_log
    debug_log.parent.mkdir(parents=True, exist_ok=True)
    debug_info = result.get("debug", {})
    debug_log.write_text(json.dumps(debug_info, ensure_ascii=False, indent=2), encoding="utf-8")

    sublogs = args.debug_sublogs_dir if args.debug_sublogs_dir else debug_log.parent / "debug_metrics_sublogs"
    sublogs.mkdir(parents=True, exist_ok=True)
    if isinstance(debug_info, dict):
        if "accuracy" in debug_info:
            (sublogs / "accuracy.json").write_text(json.dumps(debug_info["accuracy"], ensure_ascii=False, indent=2), encoding="utf-8")
        if "consistency" in debug_info:
            (sublogs / "consistency.json").write_text(json.dumps(debug_info["consistency"], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
