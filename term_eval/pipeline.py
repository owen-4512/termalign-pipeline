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
from .debug_report import (
    build_accuracy_debug,
    build_consistency_debug,
    build_cross_document_consistency_debug,
)
from .metrics_accuracy import compute_accuracy
from .metrics_consistency import compute_consistency, compute_cross_document_consistency

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
    include_cross_document_consistency: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    precision = 0.0
    consistency = 0.0

    if "accuracy" in selected_metrics:
        precision = compute_accuracy(records, gold_map)
        result["precision"] = precision

    if "consistency" in selected_metrics:
        consistency = compute_consistency(records)
        result["consistency"] = consistency
        if include_cross_document_consistency:
            result["cross_document_consistency"] = compute_cross_document_consistency(records)

    if selected_metrics == sorted(AVAILABLE_METRICS):
        weight = max(0.0, min(1.0, alpha))
        result["final_score"] = (1.0 - weight) * precision + weight * consistency

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
    include_debug: bool = False,
    include_cross_document_consistency: bool = False,
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
        "beta": beta,
        "num_records": len(records),
        "cross_document_consistency_enabled": include_cross_document_consistency,
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
                    include_cross_document_consistency=False,
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
            include_cross_document_consistency=include_cross_document_consistency,
        )

    if include_debug:
        batch_scores = result.get("batch_score", {}) if isinstance(result.get("batch_score", {}), Mapping) else {}
        score_summary: Dict[str, Any] = {}
        for key in ("precision", "consistency", "cross_document_consistency", "final_score"):
            if key in batch_scores:
                score_summary[key] = batch_scores[key]

        debug_info: Dict[str, Any] = {
            "score_summary": score_summary,
            "meta": {
                "mode": mode,
                "report_level": report_level,
                "selected_metrics": selected_metrics,
                "num_records": len(records),
                "num_source_terms": sum(len(rec.get("extracted_terms", {})) for rec in records if isinstance(rec.get("extracted_terms", {}), Mapping)),
            }
        }
        if "accuracy" in selected_metrics:
            debug_info["accuracy"] = build_accuracy_debug(records, gold_map)
        if "consistency" in selected_metrics:
            debug_info["consistency"] = build_consistency_debug(records)
            if include_cross_document_consistency:
                debug_info["cross_document_consistency"] = build_cross_document_consistency_debug(records)
        result["debug"] = debug_info

    return result
