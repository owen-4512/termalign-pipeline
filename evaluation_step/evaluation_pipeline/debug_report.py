"""Detailed debug reporting for metric calculations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .metrics_accuracy import compute_occurrence_best_detail
from .metrics_consistency import compute_per_file_consistency
from .normalization import normalize


def build_accuracy_debug(records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    skipped_terms: List[Dict[str, Any]] = []
    weighted_score_sum = 0.0
    total_weight = 0.0

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        extracted_weights = record.get("extracted_weights", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            weights = extracted_weights.get(src_term, []) if isinstance(extracted_weights, Mapping) else []
            norm_src = normalize(str(src_term))
            gold_refs = sorted({normalize(ref) for ref in gold_map.get(norm_src, set()) if normalize(ref)})
            if not gold_refs:
                skipped_terms.append({"source_file": source_file, "zh_term": str(src_term), "reason": "term_not_in_gold"})
                continue
            for idx, variant in enumerate(variants):
                norm_var = normalize(str(variant))
                if not norm_var:
                    continue
                weight = 1.0
                if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and idx < len(weights):
                    try:
                        weight = max(float(weights[idx]), 0.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                best_detail = compute_occurrence_best_detail(norm_var, set(gold_refs))
                best_score = float(best_detail["score"])
                total_weight += weight
                weighted_score_sum += best_score * weight
                details.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "occurrence_index": idx,
                        "predicted_variant": str(variant),
                        "gold_candidates": gold_refs,
                        "similarity_weight": weight,
                        "score": best_score,
                        "weighted_score": best_score * weight,
                        "match_rule": best_detail["rule"],
                    }
                )

    precision = (weighted_score_sum / total_weight) if total_weight > 0 else 0.0
    return {
        "summary": {
            "weighted_score_sum": weighted_score_sum,
            "total_effective_weight": total_weight,
            "skipped_terms_not_in_gold": len(skipped_terms),
            "precision": precision,
        },
        "occurrence_details": details,
        "skipped_terms": skipped_terms,
    }


def _build_single_document_consistency_debug(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    entropies: List[float] = []
    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        extracted_weights = record.get("extracted_weights", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            weights = extracted_weights.get(src_term, []) if isinstance(extracted_weights, Mapping) else []
            counts: dict[str, float] = defaultdict(float)
            for idx, v in enumerate(variants):
                norm_v = normalize(str(v))
                if not norm_v:
                    continue
                weight = 1.0
                if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and idx < len(weights):
                    try:
                        weight = max(float(weights[idx]), 0.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                counts[norm_v] += weight
            total = sum(counts.values())
            entropy = 0.0
            normalized_entropy = 0.0
            if total > 0 and len(counts) > 1:
                for c in counts.values():
                    p = c / total
                    entropy -= p * math.log2(p)
                normalized_entropy = entropy / math.log2(len(counts))
            entropies.append(entropy)
            details.append(
                {
                    "source_file": source_file,
                    "zh_term": str(src_term),
                    "effective_variant_weights": dict(counts),
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "consistency_score": 1.0 - normalized_entropy,
                }
            )
    mean_norm = (sum(d["normalized_entropy"] for d in details) / len(details)) if details else 0.0
    return {"summary": {"num_terms": len(entropies), "mean_normalized_entropy": mean_norm, "consistency_score": 1.0 - mean_norm, "consistency_mode": "single_document"}, "term_details": details}


def build_cross_document_consistency_debug(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    term_variants: dict[str, list[tuple[str, float]]] = defaultdict(list)
    term_docs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        extracted_weights = record.get("extracted_weights", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            weights = extracted_weights.get(src_term, []) if isinstance(extracted_weights, Mapping) else []
            norm_term = normalize(str(src_term))
            weighted_variants: list[tuple[str, float]] = []
            for idx, v in enumerate(variants):
                norm_v = normalize(str(v))
                if not norm_v:
                    continue
                weight = 1.0
                if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and idx < len(weights):
                    try:
                        weight = max(float(weights[idx]), 0.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                weighted_variants.append((norm_v, weight))
            if norm_term and weighted_variants:
                term_docs[norm_term].add(source_file)
                term_variants[norm_term].extend(weighted_variants)

    details: List[Dict[str, Any]] = []
    entropies: List[float] = []
    for term in sorted(term_docs.keys()):
        docs = sorted(term_docs[term])
        if len(docs) < 2:
            continue
        counts: dict[str, float] = defaultdict(float)
        for variant, weight in term_variants[term]:
            counts[variant] += weight
        total = sum(counts.values())
        entropy = 0.0
        normalized_entropy = 0.0
        if total > 0 and len(counts) > 1:
            for c in counts.values():
                p = c / total
                entropy -= p * math.log2(p)
            normalized_entropy = entropy / math.log2(len(counts))
        entropies.append(entropy)
        details.append(
            {
                "zh_term_normalized": term,
                "num_documents": len(docs),
                "effective_variant_weights": dict(counts),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "consistency_score": 1.0 - normalized_entropy,
            }
        )

    mean_norm = (sum(d["normalized_entropy"] for d in details) / len(details)) if details else 0.0
    return {"summary": {"num_cross_document_terms": len(entropies), "mean_normalized_entropy": mean_norm, "consistency_score": 1.0 - mean_norm, "consistency_mode": "cross_document"}, "term_details": details}


def build_consistency_debug(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records_list = list(records)
    doc_ids = {str(rec.get("source_file", "__default__")) for rec in records_list if isinstance(rec.get("extracted_terms", {}), Mapping)}
    if len(doc_ids) <= 1:
        return _build_single_document_consistency_debug(records_list)
    cross_debug = build_cross_document_consistency_debug(records_list)
    summary = dict(cross_debug.get("summary", {}))
    summary["per_file_consistency_scores"] = compute_per_file_consistency(records_list)
    cross_debug["summary"] = summary
    return cross_debug
