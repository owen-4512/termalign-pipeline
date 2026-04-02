#!/usr/bin/env python3
"""Pipeline runner with optional aggregation and visualization stages."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from visualization_step.visualization import visualize_evaluation_results


def add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--visualization", action=argparse.BooleanOptionalAction, default=False, help="Run visualization stage.")
    parser.add_argument("--visualization-results-dir", default="data/results")
    parser.add_argument("--visualization-evaluation-files", nargs="*", default=None)
    parser.add_argument("--visualization-output-figure", default="data/results/visualization/weighted_consistency_vs_accuracy.png")
    parser.add_argument("--visualization-output-table-csv", default="data/results/visualization/weighted_scores.csv")
    parser.add_argument("--visualization-title", default="Evaluation: Weighted Consistency vs Weighted Accuracy")


def add_aggregation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aggregate-after-eval", action=argparse.BooleanOptionalAction, default=False, help="Aggregate results after evaluation/full.")
    parser.add_argument("--aggregate-eval-dir", default="data/results/all_evaluation_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Run full pipeline")
    full.add_argument("--evaluation-output", default="data/results/evaluation_result.json")
    add_aggregation_args(full)

    ev = sub.add_parser("evaluation", help="Run evaluation-only")
    ev.add_argument("--evaluation-output", default="data/results/evaluation_result.json")
    add_aggregation_args(ev)

    ag = sub.add_parser("aggregate", help="Aggregate all evaluation_result.json files once after batch runs")
    ag.add_argument("--results-root", default="data/results")
    ag.add_argument("--aggregate-eval-dir", default="data/results/all_evaluation_results")

    vz = sub.add_parser("visualization", help="Run visualization standalone")
    vz.add_argument("--results-dir", default="data/results")
    vz.add_argument("--evaluation-files", nargs="*", default=None)
    vz.add_argument("--output-figure", default="data/results/visualization/weighted_consistency_vs_accuracy.png")
    vz.add_argument("--output-table-csv", default="data/results/visualization/weighted_scores.csv")
    vz.add_argument("--title", default="Evaluation: Weighted Consistency vs Weighted Accuracy")

    # Convenience command to explicitly chain once at the end:
    # aggregate -> visualization
    av = sub.add_parser("aggregate-visualize", help="Aggregate then visualize in one final step")
    av.add_argument("--results-root", default="data/results")
    av.add_argument("--aggregate-eval-dir", default="data/results/all_evaluation_results")
    av.add_argument("--output-figure", default="data/results/visualization/weighted_consistency_vs_accuracy.png")
    av.add_argument("--output-table-csv", default="data/results/visualization/weighted_scores.csv")
    av.add_argument("--title", default="Evaluation: Weighted Consistency vs Weighted Accuracy")

    return parser


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
        shutil.copy2(eval_json, out_dir / f"evaluation_result_{suffix}.json")
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

    if args.command == "aggregate":
        out_dir = aggregate_evaluation_results(results_root=args.results_root, aggregate_dir=args.aggregate_eval_dir)
        print(str(out_dir))
        return

    if args.command == "aggregate-visualize":
        out_dir = aggregate_evaluation_results(results_root=args.results_root, aggregate_dir=args.aggregate_eval_dir)
        out = visualize_evaluation_results(
            evaluation_files=None,
            results_dir=str(out_dir),
            output_figure=args.output_figure,
            output_table_csv=args.output_table_csv,
            title=args.title,
        )
        print(out)
        return

    # full/evaluation: this runner focuses on orchestration hooks only.
    eval_path = Path(args.evaluation_output)
    if not eval_path.exists():
        logging.warning("Evaluation output does not exist yet: %s", eval_path)

    if args.aggregate_after_eval:
        aggregate_evaluation_results(results_root="data/results", aggregate_dir=args.aggregate_eval_dir)

    print(str(eval_path))


if __name__ == "__main__":
    main()
