#!/usr/bin/env python3
"""Utilities for extracting `proper` mappings from JSONL and evaluating TSV term alignment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def extract_proper(input_jsonl: Path, output_jsonl: Path, proper_field: str = "proper") -> int:
    """Extract all key-value pairs in `proper` fields into a standalone JSONL.

    Each output line is shaped as:
      {"zh": <中文术语>, "en": <英文术语>}
    """

    extracted = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for record in _iter_jsonl(input_jsonl):
            proper = record.get(proper_field, {})
            if proper is None:
                continue
            if not isinstance(proper, dict):
                raise ValueError(f"Field `{proper_field}` must be a dict, got: {type(proper)!r}")

            for zh, en in proper.items():
                line = {"zh": str(zh), "en": str(en)}
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                extracted += 1

    return extracted


def _load_gold_from_jsonl(gold_jsonl: Path) -> Tuple[set[str], set[str], Dict[str, set[str]]]:
    zh_set: set[str] = set()
    en_set: set[str] = set()
    zh_to_en: Dict[str, set[str]] = defaultdict(set)

    for record in _iter_jsonl(gold_jsonl):
        proper = record.get("proper", {})
        if proper is None:
            continue
        if not isinstance(proper, dict):
            raise ValueError(f"Field `proper` must be dict in {gold_jsonl}, got {type(proper)!r}")
        for zh, en in proper.items():
            zh_s = str(zh)
            en_s = str(en)
            zh_set.add(zh_s)
            en_set.add(en_s)
            zh_to_en[zh_s].add(en_s)

    return zh_set, en_set, zh_to_en


def _load_pred_tsv(pred_tsv: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with pred_tsv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"TSV line {idx} has <2 columns: {row}")
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def _prf(correct: int, pred_total: int, gold_total: int) -> Tuple[float, float, float]:
    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(pred_tsv: Path, gold_jsonl: Path) -> dict:
    zh_gold, en_gold, align_gold = _load_gold_from_jsonl(gold_jsonl)
    predictions = _load_pred_tsv(pred_tsv)

    pred_zh = [zh for zh, _ in predictions]
    pred_en = [en for _, en in predictions]

    zh_correct = sum(1 for zh in pred_zh if zh in zh_gold)
    en_correct = sum(1 for en in pred_en if en in en_gold)
    align_correct = sum(1 for zh, en in predictions if zh in align_gold and en in align_gold[zh])

    zh_p, zh_r, zh_f1 = _prf(zh_correct, len(pred_zh), len(zh_gold))
    en_p, en_r, en_f1 = _prf(en_correct, len(pred_en), len(en_gold))
    al_p, al_r, al_f1 = _prf(align_correct, len(predictions), sum(len(v) for v in align_gold.values()))

    return {
        "counts": {
            "pred_pairs": len(predictions),
            "gold_zh_terms": len(zh_gold),
            "gold_en_terms": len(en_gold),
            "gold_align_pairs": sum(len(v) for v in align_gold.values()),
            "zh_correct": zh_correct,
            "en_correct": en_correct,
            "alignment_correct": align_correct,
        },
        "zh": {"precision": zh_p, "recall": zh_r, "f1": zh_f1},
        "en": {"precision": en_p, "recall": en_r, "f1": en_f1},
        "alignment": {"precision": al_p, "recall": al_r, "f1": al_f1},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract/evaluate term alignment data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract-proper", help="Extract `proper` dict entries into a JSONL.")
    p_extract.add_argument("--input", type=Path, required=True, help="Input JSONL containing `proper`.")
    p_extract.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    p_extract.add_argument("--field", default="proper", help="Field name for proper mapping. Default: proper")

    p_eval = sub.add_parser("evaluate", help="Evaluate predicted TSV against JSONL gold `proper` field.")
    p_eval.add_argument("--pred-tsv", type=Path, required=True, help="Predicted TSV, zh in col0 and en in col1.")
    p_eval.add_argument("--gold-jsonl", type=Path, required=True, help="Gold JSONL with `proper` mappings.")
    p_eval.add_argument("--output", type=Path, help="Optional output JSON metrics path.")

    args = parser.parse_args()

    if args.cmd == "extract-proper":
        n = extract_proper(args.input, args.output, args.field)
        print(json.dumps({"status": "ok", "extracted_pairs": n, "output": str(args.output)}, ensure_ascii=False))
        return

    if args.cmd == "evaluate":
        metrics = evaluate(args.pred_tsv, args.gold_jsonl)
        text = json.dumps(metrics, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return


if __name__ == "__main__":
    main()
