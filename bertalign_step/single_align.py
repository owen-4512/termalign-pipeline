#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

ZH_SENT_BOUNDARY = re.compile(r"(?<=[。！？；])")
EN_SENT_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"'\(\[])")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    return line.strip()


def canonicalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in text and text.count("\n") < 3:
        text = text.replace("\\n", "\n")
    return text


def is_heading_like(line: str) -> bool:
    if not line:
        return False
    return len(line) <= 30 and not re.search(r"[。！？.!?;:]", line)


def split_zh_text(text: str) -> list[str]:
    text = canonicalize_newlines(text)
    segments: list[str] = []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for para in paragraphs:
        lines = [normalize_line(x) for x in para.split("\n") if normalize_line(x)]
        if not lines:
            continue

        if all(is_heading_like(x) for x in lines):
            segments.append(f"<H>{' / '.join(lines)}")
            continue

        for line in lines:
            if line.startswith("-"):
                segments.append(line)
                continue
            if is_heading_like(line):
                segments.append(f"<H>{line}")
                continue
            parts = [p.strip() for p in ZH_SENT_BOUNDARY.split(line) if p.strip()]
            segments.extend(parts)

    return segments


def split_en_text(text: str) -> list[str]:
    text = canonicalize_newlines(text)
    segments: list[str] = []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for para in paragraphs:
        lines = [normalize_line(x) for x in para.split("\n") if normalize_line(x)]
        if not lines:
            continue

        if all(is_heading_like(x) for x in lines):
            segments.append(f"<H>{' / '.join(lines)}")
            continue

        for line in lines:
            if line.startswith("-"):
                segments.append(line)
                continue
            if is_heading_like(line):
                segments.append(f"<H>{line}")
                continue
            parts = [p.strip() for p in EN_SENT_BOUNDARY.split(line) if p.strip()]
            segments.extend(parts)

    return segments


def preprocess_for_alignment(src_text: str, tgt_text: str, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    if src_lang == "zh":
        src_sents = split_zh_text(src_text)
    else:
        src_sents = split_en_text(src_text)

    if tgt_lang == "zh":
        tgt_sents = split_zh_text(tgt_text)
    else:
        tgt_sents = split_en_text(tgt_text)

    return "\n".join(src_sents), "\n".join(tgt_sents)


def id_list(ids: list[int] | tuple[int, ...] | None) -> list[int]:
    return [int(i) for i in ids] if ids else []


def run_alignment(
    src: str | Path,
    tgt: str | Path,
    output: str | Path,
    max_align: int = 3,
    top_k: int = 5,
    win: int = 8,
    src_lang: str = "zh",
    tgt_lang: str = "en",
) -> str:
    from bertalign.aligner import Bertalign

    src_text = read_text(src)
    tgt_text = read_text(tgt)
    src_text, tgt_text = preprocess_for_alignment(src_text, tgt_text, src_lang=src_lang, tgt_lang=tgt_lang)

    init_sig = inspect.signature(Bertalign.__init__)
    aligner_kwargs: dict[str, int | str] = {"max_align": max_align, "top_k": top_k, "win": win}

    if "src_lang" in init_sig.parameters:
        aligner_kwargs["src_lang"] = src_lang
    if "tgt_lang" in init_sig.parameters:
        aligner_kwargs["tgt_lang"] = tgt_lang

    patched_detect_lang = False
    original_detect_lang = None
    if "src_lang" not in init_sig.parameters or "tgt_lang" not in init_sig.parameters:
        # bertalign versions that auto-detect language may break with newer
        # googletrans (detect returns coroutine). Force deterministic language
        # detection using provided CLI languages.
        from bertalign import aligner as aligner_module

        forced_langs = [src_lang, tgt_lang]

        def _safe_detect_lang(text: str) -> str:
            if forced_langs:
                return forced_langs.pop(0)
            return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

        original_detect_lang = getattr(aligner_module, "detect_lang", None)
        aligner_module.detect_lang = _safe_detect_lang
        patched_detect_lang = True

    try:
        aligner = Bertalign(src_text, tgt_text, **aligner_kwargs)
    finally:
        if patched_detect_lang:
            from bertalign import aligner as aligner_module
            if original_detect_lang is not None:
                aligner_module.detect_lang = original_detect_lang

    aligner.align_sents()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("src_ids\ttgt_ids\tsrc_text\ttgt_text\n")
        for src_ids, tgt_ids in aligner.result:
            src_ids_clean = id_list(src_ids)
            tgt_ids_clean = id_list(tgt_ids)
            src_line = " ".join(aligner.src_sents[i] for i in src_ids_clean)
            tgt_line = " ".join(aligner.tgt_sents[i] for i in tgt_ids_clean)
            out.write(f"{src_ids_clean}\t{tgt_ids_clean}\t{src_line}\t{tgt_line}\n")

    print(f"✅ Alignment written to {output_path}")
    print(f"Source segments: {len(aligner.src_sents)}")
    print(f"Target segments: {len(aligner.tgt_sents)}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence alignment using Bertalign")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output", default="alignment.tsv")
    parser.add_argument("--max-align", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--win", type=int, default=8)
    parser.add_argument("--src-lang", default="zh")
    parser.add_argument("--tgt-lang", default="en")

    args = parser.parse_args()
    run_alignment(
        src=args.src,
        tgt=args.tgt,
        output=args.output,
        max_align=args.max_align,
        top_k=args.top_k,
        win=args.win,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )


if __name__ == "__main__":
    main()
