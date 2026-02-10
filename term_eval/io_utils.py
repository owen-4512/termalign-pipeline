"""I/O helpers for TSV/JSONL/TXT inputs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


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
    for key in ("en_terms", "target_terms", "translations", "references"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]

    for key in ("en_term", "target", "translation", "reference"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [value]

    return []


def iter_gold_pairs(record: Mapping[str, Any]) -> Iterable[Tuple[str, List[str]]]:
    """Yield (source_term, reference_translations) from one gold JSON object.

    Supported formats:
    1) Field-based format, e.g.:
       {"zh_term": "术语A", "en_terms": ["Term A"]}
    2) Mapping format, e.g.:
       {"术语A": ["Term A"], "术语B": ["Term B"]}
    """
    has_structured_keys = any(k in record for k in ("zh_term", "source_term", "term", "source"))

    if has_structured_keys:
        yield get_term_key(record), get_reference_translations(record)
        return

    for raw_term, raw_refs in record.items():
        term = str(raw_term).strip()
        if not term:
            continue

        if isinstance(raw_refs, list):
            refs = [str(v) for v in raw_refs if str(v).strip()]
        elif isinstance(raw_refs, str):
            refs = [raw_refs] if raw_refs.strip() else []
        else:
            refs = []

        yield term, refs
