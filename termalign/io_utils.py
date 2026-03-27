from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class SentencePair:
    zh: str
    en: str


def read_sentence_pairs(path: str | Path) -> List[SentencePair]:
    df = pd.read_csv(path, sep="\t")
    if "src_text" not in df.columns or "tgt_text" not in df.columns:
        raise ValueError("Input TSV must contain 'src_text' and 'tgt_text' columns")
    pairs = [SentencePair(zh=str(row["src_text"]), en=str(row["tgt_text"])) for _, row in df.iterrows()]
    return pairs


def read_dictionary(path: str | Path) -> List[str]:
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            term = line.strip()
            if term:
                terms.append(term)
    return terms


def write_tsv(path: str | Path, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False, sep="\t")
