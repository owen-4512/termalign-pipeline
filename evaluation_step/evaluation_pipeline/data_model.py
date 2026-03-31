"""Data transformation helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .io_utils import iter_gold_pairs, read_txt_tokens
from .normalization import normalize


@dataclass
class DistanceInputs:
    source_file: str
    tokens: List[str]


def build_gold_map(gold_records: Iterable[Mapping[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for rec in gold_records:
        for raw_term, raw_refs in iter_gold_pairs(rec):
            zh_term = normalize(raw_term)
            if not zh_term:
                continue
            refs = {normalize(v) for v in raw_refs if normalize(v)}
            out.setdefault(zh_term, set()).update(refs)
    return out


def _parse_weight(row: Mapping[str, str]) -> float:
    for key in ("similarity", "weighted_confidence", "model_pair_confidence", "confidence"):
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            continue
    return 1.0


def build_extracted_records_from_tsv(tsv_rows: Iterable[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    grouped_weights: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in tsv_rows:
        source_file = row.get("source_file", "") or "__default__"
        zh_term = row.get("zh_term", "")
        en_term = row.get("en_term", "")
        if zh_term.strip() and en_term.strip():
            grouped[source_file][zh_term].append(en_term)
            grouped_weights[source_file][zh_term].append(_parse_weight(row))
    return [
        {
            "source_file": source_file,
            "extracted_terms": dict(extracted_terms),
            "extracted_weights": dict(grouped_weights[source_file]),
        }
        for source_file, extracted_terms in grouped.items()
    ]


def collapse_records_for_simple_mode(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged_terms: Dict[str, List[str]] = defaultdict(list)
    merged_weights: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        extracted = rec.get("extracted_terms", {})
        extracted_weights = rec.get("extracted_weights", {})
        if isinstance(extracted, Mapping):
            for term, variants in extracted.items():
                weights = extracted_weights.get(term, []) if isinstance(extracted_weights, Mapping) else []
                if isinstance(variants, list):
                    merged_terms[str(term)].extend(str(v) for v in variants)
                    if isinstance(weights, list) and len(weights) == len(variants):
                        merged_weights[str(term)].extend(float(w) for w in weights)
                    else:
                        merged_weights[str(term)].extend([1.0] * len(variants))
                else:
                    merged_terms[str(term)].append(str(variants))
                    merged_weights[str(term)].append(float(weights[0]) if isinstance(weights, list) and weights else 1.0)
    return [{"source_file": "__default__", "extracted_terms": dict(merged_terms), "extracted_weights": dict(merged_weights)}]


def build_token_map_simple(target_txt: Path) -> Dict[str, DistanceInputs]:
    return {"__default__": DistanceInputs(source_file="__default__", tokens=read_txt_tokens(target_txt))}


def build_token_map_batch(target_dir: Path, source_files: Iterable[str]) -> Dict[str, DistanceInputs]:
    out: Dict[str, DistanceInputs] = {}
    for source_file in set(source_files):
        name = Path(source_file).name
        target_path = target_dir / name
        if not target_path.exists():
            target_path = target_dir / f"{Path(name).stem}.txt"
        if target_path.exists():
            out[source_file] = DistanceInputs(source_file=source_file, tokens=read_txt_tokens(target_path))
    return out
