#!/usr/bin/env python3
"""Prefect pipeline for bertalign -> termalign -> evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bertalign_step.run_bertalign import run_bertalign
from evaluation_step.evaluate_terms import evaluate_terms
from termalign_step.run_termalign import run_termalign


@task
def bertalign_task(**kwargs: str) -> str:
    return run_bertalign(**kwargs)


@task
def termalign_task(**kwargs: str) -> str:
    return run_termalign(**kwargs)


@task
def evaluation_task(**kwargs: str) -> str:
    return evaluate_terms(**kwargs)


@flow(name="term-translation-scoring-pipeline")
def term_pipeline(
    source_file: str,
    target_file: str,
    dictionary_path: str,
    bertalign_output: str,
    termalign_output: str,
    evaluation_output: str,
    bertalign_command: str | None = None,
    bertalign_default_confidence: float = 0.8,
    bertalign_max_align: int = 3,
    bertalign_top_k: int = 5,
    bertalign_win: int = 8,
    bertalign_src_lang: str = "zh",
    bertalign_tgt_lang: str = "en",
    extraction_mode: str = "model",
    termalign_mode: str = "hf",
    min_term_confidence: float = 0.5,
    min_pair_confidence: float = 0.5,
    source_lang: str = "en",
    target_lang: str = "zh",
    top_k_pairs: int = 5,
    aligner_model: str = "owen4512/minilm-finance-term-aligner",
    zh_extractor_model: str = "owen4512/bert-base-chinese-finance-term-extractor",
    en_extractor_model: str = "owen4512/bert-base-cased-finance-term-extractor",
    api_endpoint: str | None = None,
    api_key: str | None = None,
    api_model: str | None = None,
    api_prompt_file: str | None = None,
    device: int = -1,
    dict_zh_path: str | None = None,
    dict_en_path: str | None = None,
    skip_bert: bool = False,
    eval_min_confidence: float = 0.0,
    eval_confidence_field: str = "weighted_confidence",
    eval_count_field: str | None = None,
    eval_smoothing_alpha: float = 0.0,
    eval_entropy_base: float = 2.0,
    eval_normalize_entropy: bool = False,
    eval_top_k_variants: int | None = None,
) -> str:
    ba_out = bertalign_task.submit(
        source_file=source_file,
        target_file=target_file,
        output_file=bertalign_output,
        external_command=bertalign_command,
        default_confidence=bertalign_default_confidence,
        max_align=bertalign_max_align,
        top_k=bertalign_top_k,
        win=bertalign_win,
        src_lang=bertalign_src_lang,
        tgt_lang=bertalign_tgt_lang,
    ).result()

    ta_out = termalign_task.submit(
        bertalign_output=ba_out,
        output_file=termalign_output,
        extraction_mode=extraction_mode,
        termalign_mode=termalign_mode,
        min_term_confidence=min_term_confidence,
        min_pair_confidence=min_pair_confidence,
        source_lang=source_lang,
        target_lang=target_lang,
        top_k_pairs=top_k_pairs,
        aligner_model=aligner_model,
        zh_extractor_model=zh_extractor_model,
        en_extractor_model=en_extractor_model,
        api_endpoint=api_endpoint,
        api_key=api_key,
        api_model=api_model,
        api_prompt_file=api_prompt_file,
        device=device,
        dict_zh_path=dict_zh_path,
        dict_en_path=dict_en_path,
        skip_bert=skip_bert,
    ).result()

    ev_out = evaluation_task.submit(
        termalign_output=ta_out,
        dictionary_path=dictionary_path,
        output_file=evaluation_output,
        min_confidence=eval_min_confidence,
        confidence_field=eval_confidence_field,
        count_field=eval_count_field,
        smoothing_alpha=eval_smoothing_alpha,
        entropy_base=eval_entropy_base,
        normalize_entropy=eval_normalize_entropy,
        top_k_variants=eval_top_k_variants,
    ).result()

    return ev_out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefect term translation scoring pipeline")
    p.add_argument("--source-file", required=True)
    p.add_argument("--target-file", required=True)
    p.add_argument("--dictionary-path", required=True)
    p.add_argument("--bertalign-output", required=True)
    p.add_argument("--termalign-output", required=True)
    p.add_argument("--evaluation-output", required=True)

    p.add_argument("--bertalign-command", default=None)
    p.add_argument("--bertalign-default-confidence", type=float, default=0.8)
    p.add_argument("--bertalign-max-align", type=int, default=3)
    p.add_argument("--bertalign-top-k", type=int, default=5)
    p.add_argument("--bertalign-win", type=int, default=8)
    p.add_argument("--bertalign-src-lang", default="zh")
    p.add_argument("--bertalign-tgt-lang", default="en")

    p.add_argument("--extraction-mode", choices=["model", "api"], default="model")
    p.add_argument("--termalign-mode", choices=["api", "local", "hf"], default="hf")
    p.add_argument("--min-term-confidence", type=float, default=0.5)
    p.add_argument("--min-pair-confidence", type=float, default=0.5)
    p.add_argument("--source-lang", default="en")
    p.add_argument("--target-lang", default="zh")
    p.add_argument("--top-k-pairs", type=int, default=5)
    p.add_argument("--aligner-model", default="owen4512/minilm-finance-term-aligner")
    p.add_argument("--zh-extractor-model", default="owen4512/bert-base-chinese-finance-term-extractor")
    p.add_argument("--en-extractor-model", default="owen4512/bert-base-cased-finance-term-extractor")
    p.add_argument("--api-endpoint", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-model", default=None)
    p.add_argument("--api-prompt-file", default=None)
    p.add_argument("--device", type=int, default=-1)
    p.add_argument("--dict-zh-path", default=None)
    p.add_argument("--dict-en-path", default=None)
    p.add_argument("--skip-bert", action="store_true")

    p.add_argument("--eval-min-confidence", type=float, default=0.0)
    p.add_argument("--eval-confidence-field", default="weighted_confidence")
    p.add_argument("--eval-count-field", default=None)
    p.add_argument("--eval-smoothing-alpha", type=float, default=0.0)
    p.add_argument("--eval-entropy-base", type=float, default=2.0)
    p.add_argument("--eval-normalize-entropy", action="store_true")
    p.add_argument("--eval-top-k-variants", type=int, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    term_pipeline(
        source_file=args.source_file,
        target_file=args.target_file,
        dictionary_path=args.dictionary_path,
        bertalign_output=args.bertalign_output,
        termalign_output=args.termalign_output,
        evaluation_output=args.evaluation_output,
        bertalign_command=args.bertalign_command,
        bertalign_default_confidence=args.bertalign_default_confidence,
        bertalign_max_align=args.bertalign_max_align,
        bertalign_top_k=args.bertalign_top_k,
        bertalign_win=args.bertalign_win,
        bertalign_src_lang=args.bertalign_src_lang,
        bertalign_tgt_lang=args.bertalign_tgt_lang,
        extraction_mode=args.extraction_mode,
        termalign_mode=args.termalign_mode,
        min_term_confidence=args.min_term_confidence,
        min_pair_confidence=args.min_pair_confidence,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        top_k_pairs=args.top_k_pairs,
        aligner_model=args.aligner_model,
        zh_extractor_model=args.zh_extractor_model,
        en_extractor_model=args.en_extractor_model,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        api_model=args.api_model,
        api_prompt_file=args.api_prompt_file,
        device=args.device,
        dict_zh_path=args.dict_zh_path,
        dict_en_path=args.dict_en_path,
        skip_bert=args.skip_bert,
        eval_min_confidence=args.eval_min_confidence,
        eval_confidence_field=args.eval_confidence_field,
        eval_count_field=args.eval_count_field,
        eval_smoothing_alpha=args.eval_smoothing_alpha,
        eval_entropy_base=args.eval_entropy_base,
        eval_normalize_entropy=args.eval_normalize_entropy,
        eval_top_k_variants=args.eval_top_k_variants,
    )


if __name__ == "__main__":
    main()
