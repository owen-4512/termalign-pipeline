"""CLI entry for term translation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_evaluation


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
        help="Metrics to run: all, accuracy, consistency, distance. You can pass comma-separated values.",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="Hyperparameter for consistency in final score")
    parser.add_argument("--beta", type=float, default=0.0, help="Hyperparameter for distance penalty in final score")
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

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        args.output_json.write_text(payload, encoding="utf-8")
    if args.debug_log:
        debug_payload = json.dumps(result.get("debug", {}), ensure_ascii=False, indent=2)
        args.debug_log.write_text(debug_payload, encoding="utf-8")


if __name__ == "__main__":
    main()
