from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Mapping, Sequence

from .normalization import normalize


def _entropy_of_normalized_variants(normalized: List[str]) -> float:
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _normalized_entropy_of_normalized_variants(normalized: List[str]) -> float:
    counts = Counter(normalized)
    if len(counts) <= 1:
        return 0.0
    max_entropy = math.log2(len(counts))
    return _entropy_of_normalized_variants(normalized) / max_entropy if max_entropy > 0 else 0.0


def _compute_single_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    scores: List[float] = []
    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for _, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            normalized = [normalize(str(v)) for v in variants if normalize(str(v))]
            scores.append(1.0 - _normalized_entropy_of_normalized_variants(normalized))
    return sum(scores) / len(scores) if scores else 0.0


def compute_cross_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    term_variants: dict[str, list[str]] = defaultdict(list)
    term_docs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            norm_term = normalize(str(src_term))
            norm_variants = [normalize(str(v)) for v in variants if normalize(str(v))]
            if norm_term and norm_variants:
                term_docs[norm_term].add(source_file)
                term_variants[norm_term].extend(norm_variants)
    scores: list[float] = []
    for term, docs in term_docs.items():
        if len(docs) < 2:
            continue
        scores.append(1.0 - _normalized_entropy_of_normalized_variants(term_variants.get(term, [])))
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
