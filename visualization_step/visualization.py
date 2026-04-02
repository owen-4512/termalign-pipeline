#!/usr/bin/env python3
"""Visualize evaluation results as scatter plot + score table.

Features:
1) Standalone CLI: read one or multiple `evaluation_result.json` files.
2) Pipeline-friendly API: `visualize_evaluation_results(...)` callable from runner.

Expected input JSON schema (subset):
{
  "batch_score": {
    "precision": 0.85,
    "consistency": 0.72,
    "final_score": 0.83
  }
}
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path



@dataclass
class EvalPoint:
    model: str
    weighted_accuracy: float
    weighted_consistency: float
    final_score: float
    source_file: Path


def _extract_model_name(eval_json_path: Path) -> str:
    """Extract model token from folder like output_xxx / outputs_xxx."""
    for part in reversed(eval_json_path.parts):
        if part.startswith("output_") or part.startswith("outputs_"):
            return part.split("_", 1)[1] or "unknown"
    parent = eval_json_path.parent.name
    if "_" in parent:
        return parent.split("_")[-1] or parent
    return parent


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_evaluation_result(path: Path) -> EvalPoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    batch = payload.get("batch_score", {}) if isinstance(payload, dict) else {}

    weighted_accuracy = _to_float(batch.get("precision", batch.get("accuracy", batch.get("weighted_accuracy"))))
    weighted_consistency = _to_float(batch.get("consistency", batch.get("weighted_consistency")))
    final_score = _to_float(batch.get("final_score"))

    return EvalPoint(
        model=_extract_model_name(path),
        weighted_accuracy=weighted_accuracy,
        weighted_consistency=weighted_consistency,
        final_score=final_score,
        source_file=path,
    )


def discover_evaluation_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("**/evaluation_result.json"))


def _render(points: list[EvalPoint], output_figure: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required for visualization. Install deps from visualization_step/requirements.txt") from exc

    output_figure.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_plot, ax_table) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    xs = [p.weighted_consistency for p in points]
    ys = [p.weighted_accuracy for p in points]

    ax_plot.scatter(xs, ys, s=90)
    for p in points:
        ax_plot.annotate(
            p.model,
            (p.weighted_consistency, p.weighted_accuracy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax_plot.set_title(title)
    ax_plot.set_xlabel("weighted consistency")
    ax_plot.set_ylabel("weighted accuracy")
    ax_plot.grid(True, linestyle="--", alpha=0.35)

    headers = ["model", "weighted accuracy", "weighted consistency", "final score"]
    rows = [
        [
            p.model,
            f"{p.weighted_accuracy:.6f}",
            f"{p.weighted_consistency:.6f}",
            f"{p.final_score:.6f}",
        ]
        for p in sorted(points, key=lambda item: item.final_score, reverse=True)
    ]

    ax_table.axis("off")
    table = ax_table.table(cellText=rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    fig.savefig(output_figure, dpi=160)
    plt.close(fig)


def _write_csv(points: list[EvalPoint], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "weighted_accuracy", "weighted_consistency", "final_score", "source_file"])
        for p in sorted(points, key=lambda item: item.final_score, reverse=True):
            writer.writerow(
                [
                    p.model,
                    f"{p.weighted_accuracy:.6f}",
                    f"{p.weighted_consistency:.6f}",
                    f"{p.final_score:.6f}",
                    str(p.source_file),
                ]
            )


def visualize_evaluation_results(
    evaluation_files: list[str] | None = None,
    results_dir: str = "data/results",
    output_figure: str = "data/results/visualization/weighted_consistency_vs_accuracy.png",
    output_table_csv: str = "data/results/visualization/weighted_scores.csv",
    title: str = "Evaluation: Weighted Consistency vs Weighted Accuracy",
) -> str:
    files = [Path(p) for p in evaluation_files] if evaluation_files else discover_evaluation_files(Path(results_dir))
    if not files:
        raise FileNotFoundError(f"No evaluation_result.json found from results_dir={results_dir}")

    points = [parse_evaluation_result(path) for path in files]
    _render(points, Path(output_figure), title)
    _write_csv(points, Path(output_table_csv))
    return output_figure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--evaluation-files",
        nargs="*",
        default=None,
        help="Optional explicit evaluation_result.json files. If empty, auto-discover under --results-dir.",
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument(
        "--output-figure",
        default="data/results/visualization/weighted_consistency_vs_accuracy.png",
    )
    parser.add_argument(
        "--output-table-csv",
        default="data/results/visualization/weighted_scores.csv",
    )
    parser.add_argument(
        "--title",
        default="Evaluation: Weighted Consistency vs Weighted Accuracy",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = visualize_evaluation_results(
        evaluation_files=args.evaluation_files,
        results_dir=args.results_dir,
        output_figure=args.output_figure,
        output_table_csv=args.output_table_csv,
        title=args.title,
    )
    print(out)


if __name__ == "__main__":
    main()
