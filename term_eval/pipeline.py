"""Pipeline orchestration and metric selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .data_model import (
    DistanceInputs,
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
AVAILABLE_REPORT_LEVELS = {"document", "batch", "both"}


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


def _compute_metric_bundle(
    records: List[Mapping[str, Any]],
    gold_map: Mapping[str, set[str]],
    token_map: Mapping[str, DistanceInputs],
    selected_metrics: List[str],
    alpha: float,
    beta: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
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


def _single_record_token_map(record: Mapping[str, Any], token_map: Mapping[str, DistanceInputs]) -> Dict[str, DistanceInputs]:
    source_file = str(record.get("source_file", "__default__"))
    if source_file in token_map:
        return {source_file: token_map[source_file]}
    return {}


def run_evaluation(
    term_align_tsv: Path,
    gold_jsonl: Path,
    mode: str,
    metrics: Iterable[str],
    alpha: float,
    beta: float,
    target_txt: Path | None = None,
    target_dir: Path | None = None,
    report_level: str = "batch",
) -> Dict[str, Any]:
    selected_metrics = parse_metrics(metrics)
    report_level = report_level.lower()
    if report_level not in AVAILABLE_REPORT_LEVELS:
        raise ValueError(f"Unsupported report level '{report_level}'. Available: {','.join(sorted(AVAILABLE_REPORT_LEVELS))}")

    rows = read_tsv(term_align_tsv)
    records = build_extracted_records_from_tsv(rows)
    gold_map = build_gold_map(read_jsonl(gold_jsonl))

    token_map: Dict[str, DistanceInputs] = {}
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
        "report_level": report_level,
        "metrics": selected_metrics,
        "alpha": alpha,
        "beta": beta,
        "num_records": len(records),
    }

    if report_level in {"document", "both"}:
        document_scores: List[Dict[str, Any]] = []
        for record in records:
            per_doc = {
                "source_file": str(record.get("source_file", "__default__")),
                **_compute_metric_bundle(
                    records=[record],
                    gold_map=gold_map,
                    token_map=_single_record_token_map(record, token_map),
                    selected_metrics=selected_metrics,
                    alpha=alpha,
                    beta=beta,
                ),
            }
            document_scores.append(per_doc)
        result["document_scores"] = document_scores

    if report_level in {"batch", "both"}:
        result["batch_score"] = _compute_metric_bundle(
            records=records,
            gold_map=gold_map,
            token_map=token_map,
            selected_metrics=selected_metrics,
            alpha=alpha,
            beta=beta,
        )

    return result
