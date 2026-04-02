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
    df = pd.read_csv(path, sep="\t", usecols=["src_text", "tgt_text"])
    return [SentencePair(zh=str(src), en=str(tgt)) for src, tgt in zip(df["src_text"], df["tgt_text"])]


def read_dictionary(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_tsv(path: str | Path, rows: Iterable[dict]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, sep="\t")
