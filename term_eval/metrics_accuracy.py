"""Accuracy metric: precision over term occurrences.

A source term occurrence is evaluated only if the source term exists in gold.
For each evaluated occurrence, we assign a score against all accepted gold
translations for that source term and take the best score:

1) If predicted translation contains a gold translation -> score 1.0
2) Else if a gold translation contains predicted translation -> score =
   token_count(predicted) / token_count(gold)
3) Else score 0.0
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .normalization import normalize


def _tokens(text: str) -> list[str]:
    return [t for t in text.split(" ") if t]


def _is_ordered_subsequence(needle_tokens: Sequence[str], haystack_tokens: Sequence[str]) -> bool:
    """Return True if needle tokens appear in haystack in order (not necessarily contiguous)."""
    if not needle_tokens:
        return False
    i = 0
    for token in haystack_tokens:
        if token == needle_tokens[i]:
            i += 1
            if i == len(needle_tokens):
                return True
    return False


def _occurrence_score(predicted_norm: str, gold_norm: str) -> float:
    return _occurrence_score_with_rule(predicted_norm, gold_norm)[0]


def _occurrence_score_with_rule(predicted_norm: str, gold_norm: str) -> tuple[float, str]:
    if not predicted_norm or not gold_norm:
        return 0.0, "empty"
    if predicted_norm == gold_norm:
        return 1.0, "exact_normalized"

    predicted_tokens = _tokens(predicted_norm)
    gold_tokens = _tokens(gold_norm)
    if not predicted_tokens or not gold_tokens:
        return 0.0, "empty_tokens"

    # Rule 1: predicted covers gold (ordered token subsequence) => full score.
    if _is_ordered_subsequence(gold_tokens, predicted_tokens):
        return 1.0, "predicted_covers_gold"

    # Rule 2: predicted is part of gold (ordered token subsequence) => token ratio.
    if _is_ordered_subsequence(predicted_tokens, gold_tokens):
        g_tokens = len(gold_tokens)
        if g_tokens == 0:
            return 0.0, "empty_gold_tokens"
        p_tokens = len(predicted_tokens)
        return p_tokens / g_tokens, "predicted_part_of_gold"
    return 0.0, "no_match"


def compute_occurrence_best_score(predicted: str, references: set[str]) -> float:
    return compute_occurrence_best_detail(predicted, references)["score"]


def compute_occurrence_best_detail(predicted: str, references: set[str]) -> Mapping[str, Any]:
    predicted_norm = normalize(predicted)
    if not predicted_norm or not references:
        return {
            "predicted_variant_normalized": predicted_norm,
            "predicted_tokens_canonical": _tokens(predicted_norm),
            "best_gold_variant_normalized": None,
            "best_gold_tokens_canonical": [],
            "rule": "empty_or_no_references",
            "score": 0.0,
        }

    best_ref = None
    best_rule = "no_match"
    best_score = -1.0
    for ref in references:
        score, rule = _occurrence_score_with_rule(predicted_norm, ref)
        if score > best_score:
            best_score = score
            best_ref = ref
            best_rule = rule

    return {
        "predicted_variant_normalized": predicted_norm,
        "predicted_tokens_canonical": _tokens(predicted_norm),
        "best_gold_variant_normalized": best_ref,
        "best_gold_tokens_canonical": _tokens(best_ref) if best_ref else [],
        "rule": best_rule,
        "score": max(best_score, 0.0),
    }


def compute_accuracy(
    records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]
) -> float:
    score_sum = 0.0
    total_translation_occurrences = 0

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
            total_translation_occurrences += len(normalized_variants)

            for variant in normalized_variants:
                detail = compute_occurrence_best_detail(variant, references)
                score_sum += float(detail["score"])

    return (score_sum / total_translation_occurrences) if total_translation_occurrences else 0.0
