from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, List, Mapping, Sequence

from .normalization import normalize


def _entropy_of_weighted_variants(weighted_counts: Mapping[str, float]) -> float:
    if not weighted_counts:
        return 0.0
    if len(weighted_counts) <= 1:
        return 0.0
    total = sum(weighted_counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in weighted_counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _normalized_entropy_of_weighted_variants(weighted_counts: Mapping[str, float]) -> float:
    if len(weighted_counts) <= 1:
        return 0.0
    max_entropy = math.log2(len(weighted_counts))
    return _entropy_of_weighted_variants(weighted_counts) / max_entropy if max_entropy > 0 else 0.0


def _compute_single_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    scores: List[float] = []
    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        extracted_weights = record.get("extracted_weights", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            weights = extracted_weights.get(src_term, []) if isinstance(extracted_weights, Mapping) else []
            weighted_counts: dict[str, float] = defaultdict(float)
            for idx, v in enumerate(variants):
                norm = normalize(str(v))
                if not norm:
                    continue
                weight = 1.0
                if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and idx < len(weights):
                    try:
                        weight = max(float(weights[idx]), 0.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                weighted_counts[norm] += weight
            scores.append(1.0 - _normalized_entropy_of_weighted_variants(weighted_counts))
    return sum(scores) / len(scores) if scores else 0.0


def compute_cross_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
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
    scores: list[float] = []
    for term, docs in term_docs.items():
        if len(docs) < 2:
            continue
        weighted_counts: dict[str, float] = defaultdict(float)
        for variant, weight in term_variants.get(term, []):
            weighted_counts[variant] += max(weight, 0.0)
        scores.append(1.0 - _normalized_entropy_of_weighted_variants(weighted_counts))
    return sum(scores) / len(scores) if scores else 0.0


def compute_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    records_list = list(records)
    doc_ids = {
        str(rec.get("source_file", "__default__"))
        for rec in records_list
        if isinstance(rec.get("extracted_terms", {}), Mapping)
    }
    if len(doc_ids) <= 1:
        return _compute_single_document_consistency(records_list)
    return compute_cross_document_consistency(records_list)


def compute_per_file_consistency(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        if isinstance(rec.get("extracted_terms", {}), Mapping):
            grouped[str(rec.get("source_file", "__default__"))].append(rec)
    return {source_file: _compute_single_document_consistency(file_records) for source_file, file_records in grouped.items()}
