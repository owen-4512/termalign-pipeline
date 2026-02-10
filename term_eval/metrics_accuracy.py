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


def _tokens(text: str) -> list[str]:
    return [_canonical_token(t) for t in text.split(" ") if t]


def _canonical_token(token: str) -> str:
    """Lightweight canonicalization for tolerant token matching.

    This keeps matching deterministic while handling common English inflection
    differences that appear in term variants (e.g. ratio/ratios,
    enterprise/enterprises).
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


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
    if not predicted_norm or not gold_norm:
        return 0.0
    if predicted_norm == gold_norm:
        return 1.0

    predicted_tokens = _tokens(predicted_norm)
    gold_tokens = _tokens(gold_norm)
    if not predicted_tokens or not gold_tokens:
        return 0.0

    # Rule 1: predicted covers gold (ordered token subsequence) => full score.
    if _is_ordered_subsequence(gold_tokens, predicted_tokens):
        return 1.0

    # Rule 2: predicted is part of gold (ordered token subsequence) => token ratio.
    if _is_ordered_subsequence(predicted_tokens, gold_tokens):
        g_tokens = len(gold_tokens)
        if g_tokens == 0:
            return 0.0
        p_tokens = len(predicted_tokens)
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
