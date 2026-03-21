"""Consistency metric based on entropy of translation variants.

Consistency is computed from occurrence distribution only and is independent
of whether variants are accurate against the gold dictionary.
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


def compute_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    entropies: List[float] = []
    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for _src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            entropies.append(entropy_of_variants([str(v) for v in variants]))

    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def compute_cross_document_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    """Mean entropy for terms that appear in two or more source files."""
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

    entropies: list[float] = []
    for term, docs in term_docs.items():
        if len(docs) < 2:
            continue
        entropies.append(_entropy_of_normalized_variants(term_variants.get(term, [])))

    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)
