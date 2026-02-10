"""Accuracy metric: precision/recall/F1 over term occurrences.

A source term occurrence is evaluated only if the source term exists in gold.
For each evaluated occurrence, we assign a score against all accepted gold
translations for that source term and take the best score:

1) If predicted translation contains a gold translation -> score 1.0
2) Else if a gold translation contains predicted translation -> score =
   token_count(predicted) / token_count(gold)
3) Else score 0.0
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

from .normalization import normalize


def _token_count(text: str) -> int:
    return len([t for t in text.split(" ") if t])


def _occurrence_score(predicted_norm: str, gold_norm: str) -> float:
    if not predicted_norm or not gold_norm:
        return 0.0
    if predicted_norm == gold_norm:
        return 1.0
    if gold_norm in predicted_norm:
        return 1.0
    if predicted_norm in gold_norm:
        g_tokens = _token_count(gold_norm)
        if g_tokens == 0:
            return 0.0
        p_tokens = _token_count(predicted_norm)
        return p_tokens / g_tokens
    return 0.0


def compute_occurrence_best_score(predicted: str, references: set[str]) -> float:
    predicted_norm = normalize(predicted)
    if not predicted_norm or not references:
        return 0.0
    return max(_occurrence_score(predicted_norm, ref) for ref in references)


def compute_accuracy(
    records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]
) -> Tuple[float, float, float]:
    score_sum = 0.0
    total_translation_occurrences = 0
    total_original_term_occurrences = 0

    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for src_term, variants in extracted_terms.items():
            norm_src = normalize(str(src_term))
            references = gold_map.get(norm_src)
            if not references:
                # Skip terms not present in gold.
                continue

            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]

            normalized_variants = [normalize(str(v)) for v in variants if normalize(str(v))]
            total_original_term_occurrences += len(normalized_variants)
            total_translation_occurrences += len(normalized_variants)

            for variant in normalized_variants:
                score_sum += max(_occurrence_score(variant, ref) for ref in references)

    precision = (score_sum / total_translation_occurrences) if total_translation_occurrences else 0.0
    recall = (score_sum / total_original_term_occurrences) if total_original_term_occurrences else 0.0
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
