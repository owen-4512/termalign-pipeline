"""Consistency metric based on entropy of translation variants."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, List, Mapping, Sequence

from .normalization import normalize


def entropy_of_variants(variants: List[str]) -> float:
    normalized = [normalize(v) for v in variants if normalize(v)]
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
