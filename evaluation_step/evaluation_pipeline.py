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
from .debug_report import build_accuracy_debug, build_consistency_debug
from .metrics_accuracy import compute_accuracy
from .metrics_consistency import compute_consistency

AVAILABLE_METRICS = {"accuracy", "consistency"}
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
                raise ValueError(f"Unsupported metric '{metric}'.")
    deduped, seen = [], set()
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
) -> Dict[str, Any]:
    del token_map
    result: Dict[str, Any] = {}
    accuracy = consistency = 0.0
    if "accuracy" in selected_metrics:
        accuracy = compute_accuracy(records, gold_map)
        result["accuracy"] = accuracy
    if "consistency" in selected_metrics:
        consistency = compute_consistency(records)
        result["consistency"] = consistency
    if selected_metrics == sorted(AVAILABLE_METRICS):
        weight = max(0.0, min(1.0, alpha))
        result["final_score"] = (1.0 - weight) * accuracy + weight * consistency
    return result


def _single_record_token_map(record: Mapping[str, Any], token_map: Mapping[str, DistanceInputs]) -> Dict[str, DistanceInputs]:
    source_file = str(record.get("source_file", "__default__"))
    return {source_file: token_map[source_file]} if source_file in token_map else {}


def run_evaluation(
    term_align_tsv: Path,
    gold_jsonl: Path,
    mode: str,
    metrics: Iterable[str],
    alpha: float,
    target_txt: Path | None = None,
    target_dir: Path | None = None,
    report_level: str = "batch",
    include_debug: bool = False,
    include_cross_document_consistency: bool = False,
) -> Dict[str, Any]:
    selected_metrics = parse_metrics(metrics)
    report_level = report_level.lower()
    if report_level not in AVAILABLE_REPORT_LEVELS:
        raise ValueError(f"Unsupported report level '{report_level}'.")

    rows = read_tsv(term_align_tsv)
    records = build_extracted_records_from_tsv(rows)
    gold_map = build_gold_map(read_jsonl(gold_jsonl))

    token_map: Dict[str, DistanceInputs] = {}
    if mode == "simple":
        if target_txt is not None:
            token_map = build_token_map_simple(target_txt)
        records = collapse_records_for_simple_mode(records)
    elif mode == "batch":
        if target_dir is not None:
            source_files = [str(rec.get("source_file", "")) for rec in records]
            token_map = build_token_map_batch(target_dir, source_files)
    else:
        raise ValueError("mode must be 'simple' or 'batch'")

    result: Dict[str, Any] = {
        "mode": mode,
        "report_level": report_level,
        "metrics": selected_metrics,
        "alpha": alpha,
        "num_records": len(records),
        "cross_document_consistency_enabled": include_cross_document_consistency,
    }

    if report_level in {"document", "both"}:
        result["document_scores"] = [
            {
                "source_file": str(record.get("source_file", "__default__")),
                **_compute_metric_bundle(
                    [record],
                    gold_map,
                    _single_record_token_map(record, token_map),
                    selected_metrics,
                    alpha,
                ),
            }
            for record in records
        ]

    if report_level in {"batch", "both"}:
        result["batch_score"] = _compute_metric_bundle(records, gold_map, token_map, selected_metrics, alpha)

    if include_debug:
        num_source_terms = 0
        for record in records:
            extracted_terms = record.get("extracted_terms", {})
            if isinstance(extracted_terms, Mapping):
                num_source_terms += len(extracted_terms)
        batch_scores = result.get("batch_score", {}) if isinstance(result.get("batch_score", {}), Mapping) else {}
        debug_info: Dict[str, Any] = {
            "score_summary": {
                "accuracy": batch_scores.get("accuracy"),
                "consistency": batch_scores.get("consistency"),
                "final_score": batch_scores.get("final_score"),
            },
            "meta": {
                "mode": mode,
                "report_level": report_level,
                "selected_metrics": selected_metrics,
                "num_records": len(records),
                "num_source_terms": num_source_terms,
            },
        }
        if "accuracy" in selected_metrics:
            debug_info["accuracy"] = build_accuracy_debug(records, gold_map)
        if "consistency" in selected_metrics:
            debug_info["consistency"] = build_consistency_debug(records)
        result["debug"] = debug_info

    return result
