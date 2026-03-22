#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from bertalign.aligner import Bertalign

ZH_SENT_BOUNDARY = re.compile(r"(?<=[。！？；])")
EN_SENT_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"'\(\[])")


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)  # markdown heading marker
    return line.strip()


def is_heading_like(line: str) -> bool:
    if not line:
        return False
    if len(line) <= 30 and not re.search(r"[。！？.!?;:]", line):
        return True
    return False


def split_zh_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    segments = []

    for raw in text.split("\n"):
        line = normalize_line(raw)
        if not line:
            continue

        # keep bullet as one unit to avoid too many tiny fragments
        if line.startswith("-"):
            segments.append(line)
            continue

        if is_heading_like(line):
            segments.append(f"<H>{line}")
            continue

        parts = [p.strip() for p in ZH_SENT_BOUNDARY.split(line) if p.strip()]
        segments.extend(parts)

    return segments


def split_en_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    segments = []

    for raw in text.split("\n"):
        line = normalize_line(raw)
        if not line:
            continue

        if line.startswith("-"):
            segments.append(line)
            continue

        if is_heading_like(line):
            segments.append(f"<H>{line}")
            continue

        parts = [p.strip() for p in EN_SENT_BOUNDARY.split(line) if p.strip()]
        segments.extend(parts)

    return segments


def preprocess_for_alignment(src_text, tgt_text):
    src_sents = split_zh_text(src_text)
    tgt_sents = split_en_text(tgt_text)
    return "\n".join(src_sents), "\n".join(tgt_sents)


def id_list(ids):
    return [int(i) for i in ids] if ids else []


def run_alignment(src, tgt, output, max_align=3, top_k=5, win=8):
    src_text = read_text(src)
    tgt_text = read_text(tgt)
    src_text, tgt_text = preprocess_for_alignment(src_text, tgt_text)

    aligner = Bertalign(src_text, tgt_text, max_align=max_align, top_k=top_k, win=win)
    aligner.align_sents()

    with open(output, "w", encoding="utf-8") as out:
        out.write("src_ids\ttgt_ids\tsrc_text\ttgt_text\n")
        for src_ids, tgt_ids in aligner.result:
            src_ids_clean = id_list(src_ids)
            tgt_ids_clean = id_list(tgt_ids)
            src_line = " ".join(aligner.src_sents[i] for i in src_ids_clean)
            tgt_line = " ".join(aligner.tgt_sents[i] for i in tgt_ids_clean)
            out.write(f"{src_ids_clean}\t{tgt_ids_clean}\t{src_line}\t{tgt_line}\n")

    print(f"✅ Alignment written to {output}")
    print(f"Source segments: {len(aligner.src_sents)}")
    print(f"Target segments: {len(aligner.tgt_sents)}")


def main():
    parser = argparse.ArgumentParser(description="Sentence alignment using Bertalign")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output", default="alignment.tsv")
    parser.add_argument("--max_align", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--win", type=int, default=8)

    args = parser.parse_args()
    run_alignment(args.src, args.tgt, args.output, args.max_align, args.top_k, args.win)


if __name__ == "__main__":
    main()
