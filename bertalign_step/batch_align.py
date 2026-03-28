#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bertalign_step.single_align import run_alignment


def _source_base_name(path: Path, src_lang: str) -> str | None:
    suffix = f"_{src_lang}.txt"
    if not path.name.endswith(suffix):
        return None
    return path.name[: -len(suffix)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch sentence alignment based on source/target filename pairs."
    )

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src-lang", default="zh")
    parser.add_argument("--tgt-lang", default="en")

    parser.add_argument("--max-align", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--win", type=int, default=8)

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when pair is missing or filename is invalid.",
    )

    return parser.parse_args()


def run_batch_alignment(
    data_dir: str | Path,
    output_dir: str | Path,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    max_align: int = 3,
    top_k: int = 5,
    win: int = 8,
    strict: bool = False,
) -> int:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(data_dir.glob(f"*_{src_lang}.txt"))

    if not src_files:
        print(f"❌ No source files found in {data_dir}")
        return 1

    ok = 0
    skipped = 0

    for src_file in src_files:
        base_name = _source_base_name(src_file, src_lang)
        if not base_name:
            print(f"⚠️ Skipped (invalid source name): {src_file.name}")
            skipped += 1
            if strict:
                return 2
            continue

        tgt_file = data_dir / f"{base_name}_{tgt_lang}.txt"
        out_file = output_dir / f"{base_name}_{src_lang}_{tgt_lang}_align.tsv"

        if not tgt_file.exists():
            print(f"⚠️ Missing target file: {tgt_file.name}")
            skipped += 1
            if strict:
                return 3
            continue

        print(f"🚀 Aligning {base_name} ...")
        run_alignment(
            src=src_file,
            tgt=tgt_file,
            output=out_file,
            max_align=max_align,
            top_k=top_k,
            win=win,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        ok += 1

    print(f"🎉 Batch alignment finished! success={ok}, skipped={skipped}")
    return 0


def main() -> int:
    args = parse_args()
    return run_batch_alignment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_align=args.max_align,
        top_k=args.top_k,
        win=args.win,
        strict=args.strict,
    )


if __name__ == "__main__":
    raise SystemExit(main())
