from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .normalization import normalize


def _tokens(text: str) -> list[str]:
    return [t for t in text.split(" ") if t]


def _is_ordered_subsequence(needle_tokens: Sequence[str], haystack_tokens: Sequence[str]) -> bool:
    if not needle_tokens:
        return False
    i = 0
    for token in haystack_tokens:
        if token == needle_tokens[i]:
            i += 1
            if i == len(needle_tokens):
                return True
    return False


def _token_overlap_count(predicted_norm: str, gold_norm: str) -> int:
    return len(set(_tokens(predicted_norm)) & set(_tokens(gold_norm)))


def _occurrence_score_with_rule(predicted_norm: str, gold_norm: str) -> tuple[float, str]:
    if not predicted_norm or not gold_norm:
        return 0.0, "empty"
    if predicted_norm == gold_norm:
        return 1.0, "exact_normalized"

    predicted_tokens = _tokens(predicted_norm)
    gold_tokens = _tokens(gold_norm)
    if _is_ordered_subsequence(gold_tokens, predicted_tokens):
        return 1.0, "predicted_covers_gold"
    if _is_ordered_subsequence(predicted_tokens, gold_tokens):
        return len(predicted_tokens) / max(len(gold_tokens), 1), "predicted_part_of_gold"
    return 0.0, "no_match"


def compute_occurrence_best_detail(predicted: str, references: set[str]) -> Mapping[str, Any]:
    predicted_norm = normalize(predicted)
    normalized_refs = sorted({normalize(ref) for ref in references if normalize(ref)})
    if not predicted_norm or not normalized_refs:
        return {
            "predicted_variant_normalized": predicted_norm,
            "predicted_tokens_canonical": _tokens(predicted_norm),
            "best_gold_variant_normalized": None,
            "best_gold_tokens_canonical": [],
            "rule": "empty_or_no_references",
            "score": 0.0,
        }

    best_ref, best_rule, best_score, best_overlap = None, "no_match", -1.0, -1
    for ref in normalized_refs:
        score, rule = _occurrence_score_with_rule(predicted_norm, ref)
        overlap = _token_overlap_count(predicted_norm, ref)
        if score > best_score or (score == best_score and overlap > best_overlap) or (
            score == best_score and overlap == best_overlap and (best_ref is None or ref < best_ref)
        ):
            best_ref, best_rule, best_score, best_overlap = ref, rule, score, overlap

    return {
        "predicted_variant_normalized": predicted_norm,
        "predicted_tokens_canonical": _tokens(predicted_norm),
        "best_gold_variant_normalized": best_ref,
        "best_gold_tokens_canonical": _tokens(best_ref) if best_ref else [],
        "rule": best_rule,
        "score": max(best_score, 0.0),
    }


def compute_accuracy(records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]) -> float:
    score_sum, total_weight = 0.0, 0.0
    for record in records:
        extracted_terms = record.get("extracted_terms", {})
        extracted_weights = record.get("extracted_weights", {})
        if not isinstance(extracted_terms, Mapping):
            continue
        for src_term, variants in extracted_terms.items():
            refs = gold_map.get(normalize(str(src_term)))
            if not refs:
                continue
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            weights = extracted_weights.get(src_term, []) if isinstance(extracted_weights, Mapping) else []
            for idx, raw_variant in enumerate(variants):
                variant = normalize(str(raw_variant))
                if not variant:
                    continue
                weight = 1.0
                if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and idx < len(weights):
                    try:
                        weight = max(float(weights[idx]), 0.0)
                    except (TypeError, ValueError):
                        weight = 1.0
                total_weight += weight
                score_sum += float(compute_occurrence_best_detail(variant, refs)["score"]) * weight
    return (score_sum / total_weight) if total_weight > 0 else 0.0
