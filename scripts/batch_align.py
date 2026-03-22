#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import re
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bertalign_runner import run_alignment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch sentence alignment (ZH-EN) based on filename pattern YYYY_ID_lang.txt"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing input txt files (e.g. 2015_01_zh.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write output TSV files",
    )

    parser.add_argument("--src-lang", default="zh")
    parser.add_argument("--tgt-lang", default="en")

    # Keep defaults aligned with bertalign_runner.py
    parser.add_argument("--max-align", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--win", type=int, default=8)

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when a pair is missing or filename is invalid.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    src_lang = re.escape(args.src_lang)
    tgt_lang = args.tgt_lang

    # Filename pattern: YYYY_ID_lang.txt
    pattern = re.compile(rf"^(?P<prefix>\d{{4}})_(?P<id>\d+)_{src_lang}\.txt$")

    src_files = sorted(data_dir.glob(f"*_{args.src_lang}.txt"))

    if not src_files:
        print(f"❌ No source files found in {data_dir}")
        return 1

    ok = 0
    skipped = 0

    for src_file in src_files:
        m = pattern.match(src_file.name)
        if not m:
            msg = f"⚠️ Skipped (invalid name): {src_file.name}"
            print(msg)
            skipped += 1
            if args.strict:
                return 2
            continue

        prefix = m.group("prefix")
        file_id = m.group("id")

        tgt_file = data_dir / f"{prefix}_{file_id}_{tgt_lang}.txt"
        out_file = output_dir / f"{prefix}_{file_id}_{args.src_lang}_{args.tgt_lang}_align.tsv"

        if not tgt_file.exists():
            msg = f"⚠️ Missing target file: {tgt_file.name}"
            print(msg)
            skipped += 1
            if args.strict:
                return 3
            continue

        print(f"🚀 Aligning {prefix}_{file_id} ...")
        run_alignment(
            src=src_file,
            tgt=tgt_file,
            output=out_file,
            max_align=args.max_align,
            top_k=args.top_k,
            win=args.win,
        )
        ok += 1

    print(f"🎉 Batch alignment finished! success={ok}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
