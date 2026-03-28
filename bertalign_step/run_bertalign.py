#!/usr/bin/env python3
"""Run bertalign step and emit aligned segments with confidence JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bertalign_step.batch_align import run_batch_alignment
from bertalign_step.single_align import run_alignment


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fallback_align(src_lines: list[str], tgt_lines: list[str], default_confidence: float) -> list[dict[str, Any]]:
    n = min(len(src_lines), len(tgt_lines))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "source_segment": src_lines[i],
                "target_segment": tgt_lines[i],
                "source_index": i,
                "target_index": i,
                "alignment_confidence": default_confidence,
            }
        )
    return rows


def _run_external_command(command: str, src_path: Path, tgt_path: Path, output_path: Path) -> None:
    formatted = command.format(src=src_path, tgt=tgt_path, out=output_path)
    subprocess.run(formatted, shell=True, check=True)


def _tsv_to_jsonl(tsv_path: Path, jsonl_path: Path, default_confidence: float = 1.0) -> str:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("r", encoding="utf-8") as fin, jsonl_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for idx, row in enumerate(reader):
            out_row = {
                "source_segment": row.get("src_text", "").strip(),
                "target_segment": row.get("tgt_text", "").strip(),
                "source_index": idx,
                "target_index": idx,
                "alignment_confidence": default_confidence,
                "src_ids": row.get("src_ids", ""),
                "tgt_ids": row.get("tgt_ids", ""),
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    return str(jsonl_path)


def run_bertalign(
    source_file: str,
    target_file: str,
    output_file: str,
    external_command: str | None = None,
    default_confidence: float = 0.8,
    max_align: int = 3,
    top_k: int = 5,
    win: int = 8,
    src_lang: str = "zh",
    tgt_lang: str = "en",
) -> str:
    src_path = Path(source_file)
    tgt_path = Path(target_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if external_command:
        _run_external_command(external_command, src_path, tgt_path, out_path)
        if out_path.suffix.lower() == ".tsv":
            converted = out_path.with_suffix(".jsonl")
            return _tsv_to_jsonl(out_path, converted, default_confidence=default_confidence)
        return str(out_path)

    try:
        tsv_out = out_path.with_suffix(".tsv")
        run_alignment(
            src=src_path,
            tgt=tgt_path,
            output=tsv_out,
            max_align=max_align,
            top_k=top_k,
            win=win,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return _tsv_to_jsonl(tsv_out, out_path, default_confidence=default_confidence)
    except Exception as e:
        print(f"⚠️ Bertalign failed, fallback to line-to-line alignment. reason={e}")

    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    rows = _fallback_align(src_lines, tgt_lines, default_confidence=default_confidence)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERTAlign step")
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--output-file", default=None)

    parser.add_argument("--external-command", default=None)
    parser.add_argument("--default-confidence", type=float, default=0.8)

    parser.add_argument("--max-align", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--win", type=int, default=8)
    parser.add_argument("--src-lang", default="zh")
    parser.add_argument("--tgt-lang", default="en")

    parser.add_argument("--batch-data-dir", default=None)
    parser.add_argument("--batch-output-dir", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.batch_data_dir and args.batch_output_dir:
        code = run_batch_alignment(
            data_dir=args.batch_data_dir,
            output_dir=args.batch_output_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_align=args.max_align,
            top_k=args.top_k,
            win=args.win,
            strict=args.strict,
        )
        raise SystemExit(code)

    if not (args.source_file and args.target_file and args.output_file):
        raise ValueError("single-file mode requires --source-file --target-file --output-file")

    run_bertalign(
        source_file=args.source_file,
        target_file=args.target_file,
        output_file=args.output_file,
        external_command=args.external_command,
        default_confidence=args.default_confidence,
        max_align=args.max_align,
        top_k=args.top_k,
        win=args.win,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )


if __name__ == "__main__":
    main()
