#!/usr/bin/env python3
"""Pipeline runner with optional visualization step and evaluation aggregation."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from visualization_step.visualization import visualize_evaluation_results


def add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visualization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run visualization right after evaluation (default: disabled).",
    )
    parser.add_argument(
        "--visualization-results-dir",
        default="data/results",
        help="Auto-discovery root for evaluation_result.json files.",
    )
    parser.add_argument(
        "--visualization-evaluation-files",
        nargs="*",
        default=None,
        help="Optional explicit evaluation_result.json files.",
    )
    parser.add_argument(
        "--visualization-output-figure",
        default="data/results/visualization/weighted_consistency_vs_accuracy.png",
    )
    parser.add_argument(
        "--visualization-output-table-csv",
        default="data/results/visualization/weighted_scores.csv",
    )
    parser.add_argument(
        "--visualization-title",
        default="Evaluation: Weighted Consistency vs Weighted Accuracy",
    )


def add_aggregation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--aggregate-eval-dir",
        default="data/results/all_evaluation_results",
        help="Directory used to gather all inputs_xxx evaluation_result.json files.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Run full pipeline")
    full.add_argument(
        "--evaluation-output",
        default="data/results/evaluation_result.json",
        help="Evaluation output path produced by previous step.",
    )
    add_visualization_args(full)
    add_aggregation_args(full)

    ev = sub.add_parser("evaluation", help="Run evaluation-only then optional visualization")
    ev.add_argument(
        "--evaluation-output",
        default="data/results/evaluation_result.json",
    )
    add_visualization_args(ev)
    add_aggregation_args(ev)

    vz = sub.add_parser("visualization", help="Run visualization standalone")
    vz.add_argument("--results-dir", default="data/results")
    vz.add_argument("--evaluation-files", nargs="*", default=None)
    vz.add_argument(
        "--output-figure",
        default="data/results/visualization/weighted_consistency_vs_accuracy.png",
    )
    vz.add_argument(
        "--output-table-csv",
        default="data/results/visualization/weighted_scores.csv",
    )
    vz.add_argument(
        "--title",
        default="Evaluation: Weighted Consistency vs Weighted Accuracy",
    )
    return parser


def _run_visualization_from_args(args: argparse.Namespace) -> str:
    return visualize_evaluation_results(
        evaluation_files=args.visualization_evaluation_files,
        results_dir=args.visualization_results_dir,
        output_figure=args.visualization_output_figure,
        output_table_csv=args.visualization_output_table_csv,
        title=args.visualization_title,
    )


def _model_suffix_from_eval(eval_path: Path) -> str:
    for part in reversed(eval_path.parts):
        if part.startswith("output_") or part.startswith("outputs_"):
            return part.split("_", 1)[1] or "default"
    parent = eval_path.parent.name
    if "_" in parent:
        return parent.split("_")[-1] or "default"
    return parent or "default"


def aggregate_evaluation_results(results_root: str = "data/results", aggregate_dir: str = "data/results/all_evaluation_results") -> Path:
    root = Path(results_root)
    out_dir = Path(aggregate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    for eval_json in sorted(root.glob("**/evaluation_result.json")):
        if out_dir in eval_json.parents:
            continue
        suffix = _model_suffix_from_eval(eval_json)
        target = out_dir / f"evaluation_result_{suffix}.json"
        shutil.copy2(eval_json, target)
        seen += 1

    logging.info("Aggregated %d evaluation files into %s", seen, out_dir)
    return out_dir


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "visualization":
        out = visualize_evaluation_results(
            evaluation_files=args.evaluation_files,
            results_dir=args.results_dir,
            output_figure=args.output_figure,
            output_table_csv=args.output_table_csv,
            title=args.title,
        )
        print(out)
        return

    eval_path = Path(args.evaluation_output)
    if not eval_path.exists():
        logging.warning("Evaluation output does not exist yet: %s", eval_path)

    # Always create aggregate folder under data/results for all inputs_xxx evaluation results.
    aggregate_dir = aggregate_evaluation_results(results_root="data/results", aggregate_dir=args.aggregate_eval_dir)
    logging.info("Aggregate evaluation folder ready: %s", aggregate_dir)

    if args.visualization:
        out = _run_visualization_from_args(args)
        logging.info("Visualization finished: %s", out)
        print(out)
    else:
        print(str(eval_path))


if __name__ == "__main__":
    main()
