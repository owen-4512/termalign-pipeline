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
    df = pd.read_csv(path)
    if "zh" not in df.columns or "en" not in df.columns:
        raise ValueError("Input CSV must contain 'zh' and 'en' columns")
    pairs = [SentencePair(zh=str(row["zh"]), en=str(row["en"])) for _, row in df.iterrows()]
    return pairs


def read_dictionary(path: str | Path) -> List[str]:
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            term = line.strip()
            if term:
                terms.append(term)
    return terms


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
