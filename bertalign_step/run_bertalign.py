#!/usr/bin/env python3
"""Run bertalign step and emit aligned segments with confidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


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
    """Run custom bertalign command.

    The command should read source/target from args and write jsonl to output.
    Available placeholders: {src}, {tgt}, {out}
    """

    formatted = command.format(src=src_path, tgt=tgt_path, out=output_path)
    subprocess.run(formatted, shell=True, check=True)


def run_bertalign(
    source_file: str,
    target_file: str,
    output_file: str,
    external_command: str | None = None,
    default_confidence: float = 0.8,
) -> str:
    src_path = Path(source_file)
    tgt_path = Path(target_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if external_command:
        _run_external_command(external_command, src_path, tgt_path, out_path)
        return str(out_path)

    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    rows = _fallback_align(src_lines, tgt_lines, default_confidence=default_confidence)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERTAlign step")
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--external-command", default=None)
    parser.add_argument("--default-confidence", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_bertalign(
        source_file=args.source_file,
        target_file=args.target_file,
        output_file=args.output_file,
        external_command=args.external_command,
        default_confidence=args.default_confidence,
    )


if __name__ == "__main__":
    main()
