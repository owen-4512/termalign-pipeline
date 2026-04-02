from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class SentencePair:
    zh: str
    en: str


def read_sentence_pairs(path: str | Path) -> List[SentencePair]:
    path = Path(path)
    pairs: List[SentencePair] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "src_text" not in reader.fieldnames or "tgt_text" not in reader.fieldnames:
            raise ValueError("Input TSV must contain 'src_text' and 'tgt_text' columns")
        for row in reader:
            pairs.append(SentencePair(zh=str(row.get("src_text", "")), en=str(row.get("tgt_text", ""))))
    return pairs


def read_dictionary(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_tsv(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(first.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(first)
        for row in iterator:
            writer.writerow(row)
