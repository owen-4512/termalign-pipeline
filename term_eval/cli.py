"""CLI entry for term translation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .pipeline import run_evaluation


def _extract_metric_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    batch_score = result.get("batch_score")
    if isinstance(batch_score, dict):
        for key in ("precision", "consistency", "final_score"):
            if key in batch_score:
                summary[key] = batch_score[key]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Term translation evaluation pipeline")
    parser.add_argument("--term-align-tsv", type=Path, required=True)
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=["simple", "batch"], default="simple")
    parser.add_argument("--target-txt", type=Path, help="Target txt for simple mode")
    parser.add_argument("--target-dir", type=Path, help="Target txt directory for batch mode")
    parser.add_argument(
        "--report-level",
        choices=["document", "batch", "both"],
        default="batch",
        help="Output score granularity: per-document, aggregated batch, or both.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metrics to run: all, accuracy, consistency. You can pass comma-separated values.",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="Hyperparameter for consistency in final score")
    parser.add_argument("--beta", type=float, default=0.0, help="Reserved (distance metric removed).")
    parser.add_argument("--output-json", type=Path, help="Optional output JSON file")
    parser.add_argument("--debug-log", type=Path, help="Optional debug log JSON file with detailed metric traces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        include_debug=bool(args.debug_log),
    )

    metric_summary = _extract_metric_summary(result)
    payload = json.dumps(metric_summary, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.debug_log:
        debug_payload = json.dumps(result.get("debug", {}), ensure_ascii=False, indent=2)
        args.debug_log.write_text(debug_payload, encoding="utf-8")


if __name__ == "__main__":
    main()
