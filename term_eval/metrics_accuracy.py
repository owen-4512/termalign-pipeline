"""Accuracy metric: precision/recall/F1 over term occurrences."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

from .normalization import normalize


def compute_accuracy(
    records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]
) -> Tuple[float, float, float]:
    correct = 0
    total_translation_occurrences = 0
    total_original_term_occurrences = 0

    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for src_term, variants in extracted_terms.items():
            norm_src = normalize(str(src_term))
            references = gold_map.get(norm_src, set())

            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]

            normalized_variants = [normalize(str(v)) for v in variants if normalize(str(v))]
            total_original_term_occurrences += len(normalized_variants)
            total_translation_occurrences += len(normalized_variants)

            if not references:
                continue
            for variant in normalized_variants:
                if variant in references:
                    correct += 1

    precision = (correct / total_translation_occurrences) if total_translation_occurrences else 0.0
    recall = (correct / total_original_term_occurrences) if total_original_term_occurrences else 0.0
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
