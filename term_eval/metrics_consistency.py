"""Consistency metrics derived from normalized entropy of translation variants.

We compute entropy from occurrence distribution only (independent of accuracy),
normalize it into [0,1], and then convert to a reward-style consistency score:

    consistency_score = 1 - normalized_entropy

Higher score means more consistent terminology usage.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Mapping, Sequence

from .normalization import normalize


def entropy_of_variants(variants: List[str]) -> float:
    normalized = [normalize(v) for v in variants if normalize(v)]
    return _entropy_of_normalized_variants(normalized)


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
    k = len(counts)
    if k <= 1:
        return 0.0
    entropy = _entropy_of_normalized_variants(normalized)
    max_entropy = math.log2(k)
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def compute_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    scores: List[float] = []
    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for _src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            normalized = [normalize(str(v)) for v in variants if normalize(str(v))]
            norm_entropy = _normalized_entropy_of_normalized_variants(normalized)
            scores.append(1.0 - norm_entropy)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_cross_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    """Mean consistency score for terms that appear in two or more source files."""
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
            if not norm_term:
                continue

            norm_variants = [normalize(str(v)) for v in variants if normalize(str(v))]
            if not norm_variants:
                continue

            term_docs[norm_term].add(source_file)
            term_variants[norm_term].extend(norm_variants)

    scores: list[float] = []
    for term, docs in term_docs.items():
        if len(docs) < 2:
            continue
        norm_entropy = _normalized_entropy_of_normalized_variants(term_variants.get(term, []))
        scores.append(1.0 - norm_entropy)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
