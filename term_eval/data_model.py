"""Data transformation helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .io_utils import get_reference_translations, get_term_key, read_txt_tokens
from .normalization import normalize


@dataclass
class DistanceInputs:
    source_file: str
    tokens: List[str]


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
        if not zh_term.strip() or not en_term.strip():
            continue
        grouped[source_file][zh_term].append(en_term)

    return [
        {"source_file": source_file, "extracted_terms": dict(extracted_terms)}
        for source_file, extracted_terms in grouped.items()
    ]


def collapse_records_for_simple_mode(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged_terms: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        extracted = rec.get("extracted_terms", {})
        if not isinstance(extracted, Mapping):
            continue
        for term, variants in extracted.items():
            if isinstance(variants, list):
                merged_terms[str(term)].extend(str(v) for v in variants)
            else:
                merged_terms[str(term)].append(str(variants))
    return [{"source_file": "__default__", "extracted_terms": dict(merged_terms)}]


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
