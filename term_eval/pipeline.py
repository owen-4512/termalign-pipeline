"""Pipeline orchestration and metric selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .data_model import (
    build_extracted_records_from_tsv,
    build_gold_map,
    build_token_map_batch,
    build_token_map_simple,
    collapse_records_for_simple_mode,
)
from .io_utils import read_jsonl, read_tsv
from .metrics_accuracy import compute_accuracy
from .metrics_consistency import compute_consistency
from .metrics_distance import compute_shortest_distance_penalty

AVAILABLE_METRICS = {"accuracy", "consistency", "distance"}


def parse_metrics(metrics: Iterable[str]) -> List[str]:
    selected: List[str] = []
    for item in metrics:
        for metric in str(item).split(","):
            metric = metric.strip().lower()
            if not metric:
                continue
            if metric == "all":
                selected.extend(sorted(AVAILABLE_METRICS))
            elif metric in AVAILABLE_METRICS:
                selected.append(metric)
            else:
                raise ValueError(f"Unsupported metric '{metric}'. Available: all,{','.join(sorted(AVAILABLE_METRICS))}")

    deduped = []
    seen = set()
    for metric in selected:
        if metric not in seen:
            seen.add(metric)
            deduped.append(metric)
    return deduped or sorted(AVAILABLE_METRICS)


def run_evaluation(
    term_align_tsv: Path,
    gold_jsonl: Path,
    mode: str,
    metrics: Iterable[str],
    alpha: float,
    beta: float,
    target_txt: Path | None = None,
    target_dir: Path | None = None,
) -> Dict[str, Any]:
    selected_metrics = parse_metrics(metrics)

    rows = read_tsv(term_align_tsv)
    records = build_extracted_records_from_tsv(rows)
    gold_map = build_gold_map(read_jsonl(gold_jsonl))

    token_map = {}
    if mode == "simple":
        if target_txt is None:
            raise ValueError("--target-txt is required in simple mode")
        token_map = build_token_map_simple(target_txt)
        records = collapse_records_for_simple_mode(records)
    elif mode == "batch":
        if target_dir is None:
            raise ValueError("--target-dir is required in batch mode")
        source_files = [str(rec.get("source_file", "")) for rec in records]
        token_map = build_token_map_batch(target_dir, source_files)
    else:
        raise ValueError("mode must be 'simple' or 'batch'")

    result: Dict[str, Any] = {
        "mode": mode,
        "metrics": selected_metrics,
        "alpha": alpha,
        "beta": beta,
        "num_records": len(records),
    }

    f1 = 0.0
    consistency = 0.0
    distance_penalty = 0.0

    if "accuracy" in selected_metrics:
        f1, precision, recall = compute_accuracy(records, gold_map)
        result.update({"f1": f1, "precision": precision, "recall": recall})

    if "consistency" in selected_metrics:
        consistency = compute_consistency(records)
        result["consistency"] = consistency

    if "distance" in selected_metrics:
        distance_penalty = compute_shortest_distance_penalty(records, token_map)
        result["distance_penalty"] = distance_penalty

    if selected_metrics == sorted(AVAILABLE_METRICS):
        result["final_score"] = f1 - alpha * consistency - beta * distance_penalty

    return result
