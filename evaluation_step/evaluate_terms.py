#!/usr/bin/env python3
"""Evaluate terminology with weighted frequency, ratio, and entropy."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_dictionary(path: Path) -> dict[str, list[str]]:
    # JSON format: {"source_term": ["variant1", "variant2"]}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _entropy(probs: list[float], base: float = 2.0) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p, base) for p in probs)


def evaluate_terms(
    termalign_output: str,
    dictionary_path: str,
    output_file: str,
    min_confidence: float = 0.0,
    confidence_field: str = "weighted_confidence",
    count_field: str | None = None,
    smoothing_alpha: float = 0.0,
    entropy_base: float = 2.0,
    normalize_entropy: bool = False,
    top_k_variants: int | None = None,
) -> str:
    data = _load_jsonl(Path(termalign_output))
    dictionary = _load_dictionary(Path(dictionary_path))

    term_variant_weight = defaultdict(lambda: defaultdict(float))
    term_total_weight = defaultdict(float)

    for row in data:
        conf = float(row.get(confidence_field, 0.0))
        if conf < min_confidence:
            continue

        count = float(row.get(count_field, 1.0)) if count_field else 1.0
        eff_count = count * conf  # count(v) * confidence(v)

        source_term = row["source_term"]
        target_variant = row["target_term"]

        term_variant_weight[source_term][target_variant] += eff_count
        term_total_weight[source_term] += eff_count

    report: dict[str, Any] = {}

    for source_term, variants in term_variant_weight.items():
        allowed_variants = dictionary.get(source_term, [])
        merged = dict(variants)

        # Ensure dictionary variants exist (for smoothing and consistency metrics)
        for v in allowed_variants:
            merged.setdefault(v, 0.0)

        if top_k_variants is not None:
            merged = dict(sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k_variants])

        denom = sum(weight + smoothing_alpha for weight in merged.values())
        if denom == 0:
            ratios = {k: 0.0 for k in merged}
        else:
            ratios = {k: (w + smoothing_alpha) / denom for k, w in merged.items()}

        ent = _entropy(list(ratios.values()), base=entropy_base)
        if normalize_entropy and len(ratios) > 1:
            ent /= math.log(len(ratios), entropy_base)

        accuracy_weight = sum(weight for v, weight in variants.items() if v in allowed_variants)
        total_weight = term_total_weight[source_term]
        accuracy = accuracy_weight / total_weight if total_weight > 0 else 0.0

        report[source_term] = {
            "total_weighted_count": total_weight,
            "variant_weighted_count": merged,
            "variant_ratio": ratios,
            "entropy": ent,
            "accuracy": accuracy,
            "dictionary_variants": allowed_variants,
        }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation step")
    parser.add_argument("--termalign-output", required=True)
    parser.add_argument("--dictionary-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--confidence-field", default="weighted_confidence")
    parser.add_argument("--count-field", default=None)
    parser.add_argument("--smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--entropy-base", type=float, default=2.0)
    parser.add_argument("--normalize-entropy", action="store_true")
    parser.add_argument("--top-k-variants", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_terms(
        termalign_output=args.termalign_output,
        dictionary_path=args.dictionary_path,
        output_file=args.output_file,
        min_confidence=args.min_confidence,
        confidence_field=args.confidence_field,
        count_field=args.count_field,
        smoothing_alpha=args.smoothing_alpha,
        entropy_base=args.entropy_base,
        normalize_entropy=args.normalize_entropy,
        top_k_variants=args.top_k_variants,
    )


if __name__ == "__main__":
    main()
