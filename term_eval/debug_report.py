"""Detailed debug reporting for metric calculations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .data_model import DistanceInputs
from .metrics_accuracy import compute_occurrence_best_score
from .metrics_distance import shortest_distance, token_positions_by_variant
from .normalization import normalize


def build_accuracy_debug(records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    skipped_terms: List[Dict[str, Any]] = []
    score_sum = 0.0
    total_translation_occurrences = 0
    total_original_term_occurrences = 0

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]

            norm_src = normalize(str(src_term))
            gold_refs = sorted(gold_map.get(norm_src, set()))
            if not gold_refs:
                skipped_terms.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "reason": "term_not_in_gold",
                    }
                )
                continue

            for idx, variant in enumerate(variants):
                norm_var = normalize(str(variant))
                if not norm_var:
                    continue

                best_score = compute_occurrence_best_score(norm_var, set(gold_refs))
                total_translation_occurrences += 1
                total_original_term_occurrences += 1
                score_sum += best_score
                details.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "occurrence_index": idx,
                        "predicted_variant": str(variant),
                        "predicted_variant_normalized": norm_var,
                        "gold_variants_normalized": gold_refs,
                        "score": best_score,
                    }
                )

    precision = (score_sum / total_translation_occurrences) if total_translation_occurrences else 0.0
    recall = (score_sum / total_original_term_occurrences) if total_original_term_occurrences else 0.0
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "summary": {
            "score_sum": score_sum,
            "total_translation_occurrences": total_translation_occurrences,
            "total_original_term_occurrences": total_original_term_occurrences,
            "skipped_terms_not_in_gold": len(skipped_terms),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "occurrence_details": details,
        "skipped_terms": skipped_terms,
    }


def build_consistency_debug(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    entropies: List[float] = []

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]

            norm_variants = [normalize(str(v)) for v in variants if normalize(str(v))]
            counts = Counter(norm_variants)
            total = sum(counts.values())

            entropy = 0.0
            probs = {}
            if total > 0 and len(counts) > 1:
                for k, c in counts.items():
                    p = c / total
                    probs[k] = p
                    entropy -= p * math.log2(p)

            entropies.append(entropy)
            details.append(
                {
                    "source_file": source_file,
                    "zh_term": str(src_term),
                    "num_occurrences": total,
                    "num_distinct_variants": len(counts),
                    "variant_counts_normalized": dict(counts),
                    "variant_probabilities": probs,
                    "entropy": entropy,
                }
            )

    return {
        "summary": {
            "num_terms": len(entropies),
            "mean_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
        },
        "term_details": details,
    }


def build_distance_debug(
    records: Iterable[Mapping[str, Any]], token_map: Mapping[str, DistanceInputs]
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    penalties: List[float] = []

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        inputs = token_map.get(source_file)
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        if inputs is None or not inputs.tokens:
            details.append(
                {
                    "source_file": source_file,
                    "status": "missing_target_tokens",
                }
            )
            continue

        token_count = len(inputs.tokens)
        for src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            norm_variants = {normalize(str(v)) for v in variants if normalize(str(v))}

            if len(norm_variants) < 2:
                penalties.append(0.0)
                details.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "num_distinct_variants": len(norm_variants),
                        "status": "single_variant_or_empty",
                        "min_distance": None,
                        "penalty": 0.0,
                    }
                )
                continue

            pos_map = token_positions_by_variant(inputs.tokens, norm_variants)
            available = [v for v in norm_variants if pos_map.get(v)]
            if len(available) < 2:
                penalties.append(0.0)
                details.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "num_distinct_variants": len(norm_variants),
                        "status": "variants_not_found_in_tokens",
                        "available_variants": available,
                        "min_distance": None,
                        "penalty": 0.0,
                    }
                )
                continue

            dmin = math.inf
            best_pair: tuple[str, str] | None = None
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    d = shortest_distance(pos_map[available[i]], pos_map[available[j]])
                    if d >= 0 and d < dmin:
                        dmin = d
                        best_pair = (available[i], available[j])

            if dmin is math.inf:
                penalties.append(0.0)
                details.append(
                    {
                        "source_file": source_file,
                        "zh_term": str(src_term),
                        "status": "distance_not_computable",
                        "min_distance": None,
                        "penalty": 0.0,
                    }
                )
                continue

            penalty = 1.0 - (float(dmin) / float(token_count))
            penalty = max(0.0, min(1.0, penalty))
            penalties.append(penalty)
            details.append(
                {
                    "source_file": source_file,
                    "zh_term": str(src_term),
                    "num_distinct_variants": len(norm_variants),
                    "status": "ok",
                    "best_variant_pair": best_pair,
                    "variant_positions": {k: pos_map.get(k, []) for k in sorted(norm_variants)},
                    "token_count": token_count,
                    "min_distance": int(dmin),
                    "penalty": penalty,
                }
            )

    return {
        "summary": {
            "num_items": len(penalties),
            "mean_penalty": (sum(penalties) / len(penalties)) if penalties else 0.0,
        },
        "term_details": details,
    }
