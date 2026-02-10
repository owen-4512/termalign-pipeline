#!/usr/bin/env python3
"""Terminology translation evaluation pipeline.

Inputs:
- term align TSV with columns including:
  source_file, zh_term, en_term, similarity, zh_source, en_source,
  zh_confidence, en_confidence, zh_sentence, en_sentence
- gold dictionary in JSONL format
- source/target text files (simple mode) or directories/manifests (batch mode)

Final score:
    final_score = f1 - alpha * consistency - beta * distance_penalty
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


NORM_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Normalize text for tolerant matching and counting."""
    value = text.strip().casefold()
    value = NORM_RE.sub(" ", value)
    return " ".join(value.split())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {idx} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {idx} in {path} is not an object")
            records.append(obj)
    return records


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def read_txt_tokens(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def get_term_key(record: Mapping[str, Any]) -> str:
    for key in ("zh_term", "source_term", "term", "source"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"Cannot find source term key in gold record: {record}")


def get_reference_translations(record: Mapping[str, Any]) -> List[str]:
    """Extract candidate reference translations from a gold dictionary record."""
    for key in ("en_terms", "target_terms", "translations", "references"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]

    for key in ("en_term", "target", "translation", "reference"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [value]

    return []


def build_gold_map(gold_records: Iterable[Mapping[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for rec in gold_records:
        zh_term = normalize(get_term_key(rec))
        refs = {normalize(v) for v in get_reference_translations(rec) if normalize(v)}
        out[zh_term] = refs
    return out


def build_extracted_records_from_tsv(tsv_rows: Iterable[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for row in tsv_rows:
        source_file = row.get("source_file", "") or "__default__"
        zh_term = row.get("zh_term", "")
        en_term = row.get("en_term", "")
        if not zh_term.strip():
            continue
        if not en_term.strip():
            continue
        grouped[source_file][zh_term].append(en_term)

    records: List[Dict[str, Any]] = []
    for source_file, extracted_terms in grouped.items():
        records.append({"source_file": source_file, "extracted_terms": dict(extracted_terms)})
    return records


def compute_accuracy(
    records: Iterable[Mapping[str, Any]], gold_map: Mapping[str, set[str]]
) -> Tuple[float, float, float]:
    """Compute F1/precision/recall from term occurrences.

    Precision = correct / total_translation_occurrences
    Recall = correct / total_original_term_occurrences
    F1 = harmonic mean of precision and recall
    """
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


def entropy_of_variants(variants: List[str]) -> float:
    """Compute Shannon entropy (base2) after normalization."""
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
    """Compute mean entropy across all terms in all records."""
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


def token_positions_by_variant(tokens: Sequence[str], variants: set[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        ntok = normalize(tok)
        if ntok in variants:
            out[ntok].append(idx)
    return out


def shortest_distance(positions_a: Sequence[int], positions_b: Sequence[int]) -> int:
    i = j = 0
    best = math.inf
    while i < len(positions_a) and j < len(positions_b):
        a, b = positions_a[i], positions_b[j]
        best = min(best, abs(a - b))
        if a < b:
            i += 1
        else:
            j += 1
    return int(best) if best is not math.inf else -1


@dataclass
class DistanceInputs:
    source_file: str
    tokens: List[str]


def compute_shortest_distance_penalty(
    records: Iterable[Mapping[str, Any]], token_map: Mapping[str, DistanceInputs]
) -> float:
    """Compute mean variant proximity penalty on [0, 1]."""
    penalties: List[float] = []

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        inputs = token_map.get(source_file)
        if inputs is None or not inputs.tokens:
            continue

        token_count = len(inputs.tokens)
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for _src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            norm_variants = {normalize(str(v)) for v in variants if normalize(str(v))}
            if len(norm_variants) < 2:
                penalties.append(0.0)
                continue

            pos_map = token_positions_by_variant(inputs.tokens, norm_variants)
            available = [v for v in norm_variants if pos_map.get(v)]
            if len(available) < 2:
                penalties.append(0.0)
                continue

            dmin = math.inf
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    d = shortest_distance(pos_map[available[i]], pos_map[available[j]])
                    if d >= 0:
                        dmin = min(dmin, d)

            if dmin is math.inf:
                penalties.append(0.0)
                continue

            penalty = 1.0 - (float(dmin) / float(token_count))
            penalty = max(0.0, min(1.0, penalty))
            penalties.append(penalty)

    if not penalties:
        return 0.0
    return sum(penalties) / len(penalties)


def build_token_map_simple(target_txt: Path) -> Dict[str, DistanceInputs]:
    return {"__default__": DistanceInputs(source_file="__default__", tokens=read_txt_tokens(target_txt))}


def build_token_map_batch(target_dir: Path, source_files: Iterable[str]) -> Dict[str, DistanceInputs]:
    out: Dict[str, DistanceInputs] = {}
    for source_file in set(source_files):
        name = Path(source_file).name
        target_path = target_dir / name
        if not target_path.exists():
            target_path = target_dir / f"{Path(name).stem}.txt"
        if not target_path.exists():
            continue
        out[source_file] = DistanceInputs(source_file=source_file, tokens=read_txt_tokens(target_path))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Term translation evaluation pipeline")
    parser.add_argument("--term-align-tsv", type=Path, required=True)
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=["simple", "batch"], default="simple")
    parser.add_argument("--source-txt", type=Path, help="Source txt for simple mode")
    parser.add_argument("--target-txt", type=Path, help="Target txt for simple mode")
    parser.add_argument("--target-dir", type=Path, help="Target txt directory for batch mode")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, help="Optional output JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_tsv(args.term_align_tsv)
    records = build_extracted_records_from_tsv(rows)
    gold_map = build_gold_map(read_jsonl(args.gold_jsonl))

    if args.mode == "simple":
        if args.target_txt is None:
            raise ValueError("--target-txt is required in simple mode")
        token_map = build_token_map_simple(args.target_txt)
        # collapse records so all rows are evaluated against one text
        merged_terms: Dict[str, List[str]] = defaultdict(list)
        for rec in records:
            for term, variants in rec["extracted_terms"].items():
                merged_terms[term].extend(variants)
        records = [{"source_file": "__default__", "extracted_terms": dict(merged_terms)}]
    else:
        if args.target_dir is None:
            raise ValueError("--target-dir is required in batch mode")
        source_files = [str(rec.get("source_file", "")) for rec in records]
        token_map = build_token_map_batch(args.target_dir, source_files)

    f1, precision, recall = compute_accuracy(records, gold_map)
    consistency = compute_consistency(records)
    distance_penalty = compute_shortest_distance_penalty(records, token_map)
    final_score = f1 - args.alpha * consistency - args.beta * distance_penalty

    result = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "consistency": consistency,
        "distance_penalty": distance_penalty,
        "alpha": args.alpha,
        "beta": args.beta,
        "final_score": final_score,
        "num_records": len(records),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
