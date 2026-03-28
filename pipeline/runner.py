#!/usr/bin/env python3
"""Unified runner for single-step or full pipeline execution.

This runner is intentionally independent from Prefect deployment setup.
It can:
1) run each sub-step independently (bertalign / termalign / evaluation), or
2) run the full flow via Prefect (`--use-prefect`), or
3) run the full flow sequentially in plain Python (default).
"""

from __future__ import annotations

import argparse

from bertalign_step.run_bertalign import run_bertalign
from evaluation_step.evaluate_terms import evaluate_terms
from pipeline.prefect_flow import term_pipeline
from termalign_step.run_termalign import run_termalign


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source-file", required=True)
    p.add_argument("--target-file", required=True)
    p.add_argument("--dictionary-path", required=True)
    p.add_argument("--bertalign-output", required=True)
    p.add_argument("--termalign-output", required=True)
    p.add_argument("--evaluation-output", required=True)


def add_bertalign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bertalign-command", default=None)
    p.add_argument("--bertalign-default-confidence", type=float, default=0.8)


def add_termalign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--extraction-mode", choices=["model", "api"], default="model")
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
    p.add_argument("--device", type=int, default=-1)


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-min-confidence", type=float, default=0.0)
    p.add_argument("--eval-confidence-field", default="weighted_confidence")
    p.add_argument("--eval-count-field", default=None)
    p.add_argument("--eval-smoothing-alpha", type=float, default=0.0)
    p.add_argument("--eval-entropy-base", type=float, default=2.0)
    p.add_argument("--eval-normalize-entropy", action="store_true")
    p.add_argument("--eval-top-k-variants", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Run full pipeline")
    add_common_args(full)
    add_bertalign_args(full)
    add_termalign_args(full)
    add_eval_args(full)
    full.add_argument("--use-prefect", action="store_true", help="Run full flow through Prefect")

    ba = sub.add_parser("bertalign", help="Run only bertalign")
    ba.add_argument("--source-file", required=True)
    ba.add_argument("--target-file", required=True)
    ba.add_argument("--bertalign-output", required=True)
    add_bertalign_args(ba)

    ta = sub.add_parser("termalign", help="Run only termalign")
    ta.add_argument("--bertalign-output", required=True)
    ta.add_argument("--termalign-output", required=True)
    add_termalign_args(ta)

    ev = sub.add_parser("evaluation", help="Run only evaluation")
    ev.add_argument("--termalign-output", required=True)
    ev.add_argument("--dictionary-path", required=True)
    ev.add_argument("--evaluation-output", required=True)
    add_eval_args(ev)

    return parser


def run_full(args: argparse.Namespace) -> str:
    if args.use_prefect:
        return term_pipeline(
            source_file=args.source_file,
            target_file=args.target_file,
            dictionary_path=args.dictionary_path,
            bertalign_output=args.bertalign_output,
            termalign_output=args.termalign_output,
            evaluation_output=args.evaluation_output,
            bertalign_command=args.bertalign_command,
            bertalign_default_confidence=args.bertalign_default_confidence,
            extraction_mode=args.extraction_mode,
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
            device=args.device,
            eval_min_confidence=args.eval_min_confidence,
            eval_confidence_field=args.eval_confidence_field,
            eval_count_field=args.eval_count_field,
            eval_smoothing_alpha=args.eval_smoothing_alpha,
            eval_entropy_base=args.eval_entropy_base,
            eval_normalize_entropy=args.eval_normalize_entropy,
            eval_top_k_variants=args.eval_top_k_variants,
        )

    ba_out = run_bertalign(
        source_file=args.source_file,
        target_file=args.target_file,
        output_file=args.bertalign_output,
        external_command=args.bertalign_command,
        default_confidence=args.bertalign_default_confidence,
    )
    ta_out = run_termalign(
        bertalign_output=ba_out,
        output_file=args.termalign_output,
        extraction_mode=args.extraction_mode,
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
        device=args.device,
    )
    return evaluate_terms(
        termalign_output=ta_out,
        dictionary_path=args.dictionary_path,
        output_file=args.evaluation_output,
        min_confidence=args.eval_min_confidence,
        confidence_field=args.eval_confidence_field,
        count_field=args.eval_count_field,
        smoothing_alpha=args.eval_smoothing_alpha,
        entropy_base=args.eval_entropy_base,
        normalize_entropy=args.eval_normalize_entropy,
        top_k_variants=args.eval_top_k_variants,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "full":
        out = run_full(args)
        print(out)
        return

    if args.command == "bertalign":
        out = run_bertalign(
            source_file=args.source_file,
            target_file=args.target_file,
            output_file=args.bertalign_output,
            external_command=args.bertalign_command,
            default_confidence=args.bertalign_default_confidence,
        )
        print(out)
        return

    if args.command == "termalign":
        out = run_termalign(
            bertalign_output=args.bertalign_output,
            output_file=args.termalign_output,
            extraction_mode=args.extraction_mode,
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
            device=args.device,
        )
        print(out)
        return

    if args.command == "evaluation":
        out = evaluate_terms(
            termalign_output=args.termalign_output,
            dictionary_path=args.dictionary_path,
            output_file=args.evaluation_output,
            min_confidence=args.eval_min_confidence,
            confidence_field=args.eval_confidence_field,
            count_field=args.eval_count_field,
            smoothing_alpha=args.eval_smoothing_alpha,
            entropy_base=args.eval_entropy_base,
            normalize_entropy=args.eval_normalize_entropy,
            top_k_variants=args.eval_top_k_variants,
        )
        print(out)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
