#!/usr/bin/env python3
"""Unified runner for single-step or full pipeline execution.

This runner can:
1) run each sub-step independently (bertalign / termalign / evaluation), or
2) run the full flow via Prefect (`--use-prefect`), or
3) run the full flow sequentially in plain Python (default).

It also supports a standardized data layout under `data/`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import subprocess
import os
import venv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_ROOT = PROJECT_ROOT / ".venvs"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_step_venv(step_name: str, requirements_file: Path) -> Path:
    venv_dir = VENV_ROOT / step_name
    py = _venv_python(venv_dir)
    if not py.exists():
        logging.info("[venv] creating %s", venv_dir)
        venv.create(venv_dir, with_pip=True)
        py = _venv_python(venv_dir)
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        if requirements_file.exists():
            subprocess.run([str(py), "-m", "pip", "install", "-r", str(requirements_file)], check=True)
    return py


def run_module_in_step_venv(step_name: str, module: str, args_list: list[str]) -> subprocess.CompletedProcess[str]:
    req_file = PROJECT_ROOT / step_name / "requirements.txt"
    py = ensure_step_venv(step_name, req_file)
    cmd = [str(py), "-m", module, *args_list]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.stdout:
        logging.info("[%s stdout]\n%s", step_name, proc.stdout.strip())
    if proc.stderr:
        logging.info("[%s stderr]\n%s", step_name, proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed with code={proc.returncode}")
    return proc

def resolve_default_paths(data_dir: str) -> dict[str, str]:
    root = Path(data_dir)
    return {
        "source_file": str(root / "Task1" / "source.txt"),
        "target_file": str(root / "Task1" / "target.txt"),
        "bertalign_output": str(root / "Task2" / "bertalign.jsonl"),
        "termalign_output": str(root / "Task3" / "high_confidence" / "all_alignments_high_conf.tsv"),
        "dictionary_path": str(root / "Task3" / "proper_terms.jsonl"),
        "evaluation_output": str(root / "results" / "evaluation_result.json"),
    }


def resolve_optional_paths(data_dir: str) -> dict[str, str]:
    root = Path(data_dir)
    return {
        "dict_zh_path": str(root / "Task2" / "dict_zh.txt"),
        "dict_en_path": str(root / "Task2" / "dict_en.txt"),
        "api_prompt_file": str(root / "Task2" / "termalign_prompt.txt"),
    }


def ensure_parent_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def setup_run_logger(logs_dir: str, command: str) -> Path:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = Path(logs_dir) / f"{command}-{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Log file initialized: %s", log_path)
    return log_path


def apply_data_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = resolve_default_paths(args.data_dir)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    optional_defaults = resolve_optional_paths(args.data_dir)
    for key, value in optional_defaults.items():
        if getattr(args, key, None) is None and Path(value).exists():
            setattr(args, key, value)
    return args


def add_data_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir",
        default="data",
        help="Unified project data root (default: data). Expected layout: Task1/Task2/Task3",
    )
    p.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for run logs (default: logs)",
    )


def add_common_args(p: argparse.ArgumentParser) -> None:
    add_data_dir_arg(p)
    p.add_argument("--source-file", default=None)
    p.add_argument("--target-file", default=None)
    p.add_argument("--dictionary-path", default=None)
    p.add_argument("--bertalign-output", default=None)
    p.add_argument("--termalign-output", default=None)
    p.add_argument("--evaluation-output", default=None)


def add_bertalign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bertalign-command", default=None)
    p.add_argument("--bertalign-default-confidence", type=float, default=0.8)
    p.add_argument("--bertalign-max-align", type=int, default=3)
    p.add_argument("--bertalign-top-k", type=int, default=5)
    p.add_argument("--bertalign-win", type=int, default=8)
    p.add_argument("--bertalign-src-lang", default="zh")
    p.add_argument("--bertalign-tgt-lang", default="en")
    p.add_argument(
        "--inputs-dir",
        default=None,
        help="Alias for --bertalign-batch-data-dir. Set Task1 input folder for batch bertalign.",
    )
    p.add_argument("--bertalign-batch-data-dir", default=None, help="Run bertalign in batch mode using input dir")
    p.add_argument("--bertalign-batch-output-dir", default=None, help="Batch bertalign TSV output directory")
    p.add_argument("--bertalign-batch-strict", action="store_true", help="Batch bertalign strict mode")


def add_termalign_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--termalign-mode", choices=["api", "local", "hf"], default="hf")
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
    p.add_argument("--dict-zh-path", default=None)
    p.add_argument("--dict-en-path", default=None)
    p.add_argument("--skip-bert", action="store_true")
    p.add_argument("--api-model", default=None, help="API model name, e.g. gpt-4.1")
    p.add_argument("--api-prompt-file", default=None, help="Prompt txt path for termalign API mode")


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-mode", choices=["simple", "batch"], default="batch")
    p.add_argument("--eval-target-txt", default=None)
    p.add_argument("--eval-target-dir", default=None)
    p.add_argument("--eval-report-level", choices=["document", "batch", "both"], default="batch")
    p.add_argument("--eval-metrics", nargs="+", default=["all"])
    p.add_argument("--eval-alpha", type=float, default=0.2)
    p.add_argument("--eval-beta", type=float, default=0.0)
    p.add_argument("--eval-debug-log", default=None)
    p.add_argument("--eval-debug-sublogs-dir", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Run full pipeline")
    add_common_args(full)
    add_bertalign_args(full)
    add_termalign_args(full)
    add_eval_args(full)
    full.add_argument("--use-prefect", action="store_true", help="Run full flow through Prefect")
    full.add_argument("--isolate-venv", action="store_true", help="Run each step in its own venv via subprocess")

    ba = sub.add_parser("bertalign", help="Run only bertalign")
    add_data_dir_arg(ba)
    ba.add_argument("--source-file", default=None)
    ba.add_argument("--target-file", default=None)
    ba.add_argument("--bertalign-output", default=None)
    add_bertalign_args(ba)

    ta = sub.add_parser("termalign", help="Run only termalign")
    add_data_dir_arg(ta)
    ta.add_argument("--bertalign-output", default=None)
    ta.add_argument("--termalign-output", default=None)
    add_termalign_args(ta)

    ev = sub.add_parser("evaluation", help="Run only evaluation")
    add_data_dir_arg(ev)
    ev.add_argument("--termalign-output", default=None)
    ev.add_argument("--dictionary-path", default=None)
    ev.add_argument("--evaluation-output", default=None)
    add_eval_args(ev)

    return parser


def run_full(args: argparse.Namespace) -> str:
    args = apply_data_defaults(args)
    if args.inputs_dir and not args.bertalign_batch_data_dir:
        args.bertalign_batch_data_dir = args.inputs_dir
    setup_run_logger(args.logs_dir, "full")
    ensure_parent_dirs(args.bertalign_output, args.termalign_output, args.evaluation_output)
    logging.info("Running full pipeline with data_dir=%s", args.data_dir)

    if args.use_prefect:
        from pipeline.prefect_flow import term_pipeline
        logging.info("Execution mode: prefect")
        return term_pipeline(
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
            bertalign_batch_data_dir=args.bertalign_batch_data_dir,
            bertalign_batch_output_dir=args.bertalign_batch_output_dir,
            bertalign_batch_strict=args.bertalign_batch_strict,
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
            eval_mode=args.eval_mode,
            eval_target_txt=args.eval_target_txt,
            eval_target_dir=args.eval_target_dir,
            eval_report_level=args.eval_report_level,
            eval_metrics=args.eval_metrics,
            eval_alpha=args.eval_alpha,
            eval_beta=args.eval_beta,
            eval_debug_log=args.eval_debug_log,
            eval_debug_sublogs_dir=args.eval_debug_sublogs_dir,
        )

    if args.isolate_venv:
        logging.info("Execution mode: subprocess-per-step venv")
        if args.bertalign_batch_data_dir:
            batch_output_dir = args.bertalign_batch_output_dir or str(Path(args.data_dir) / "Task2" / "batch_tsv")
            run_module_in_step_venv(
                "bertalign_step",
                "bertalign_step.batch_align",
                [
                    "--data-dir", args.bertalign_batch_data_dir,
                    "--output-dir", batch_output_dir,
                    "--src-lang", args.bertalign_src_lang,
                    "--tgt-lang", args.bertalign_tgt_lang,
                    "--max-align", str(args.bertalign_max_align),
                    "--top-k", str(args.bertalign_top_k),
                    "--win", str(args.bertalign_win),
                    *(["--strict"] if args.bertalign_batch_strict else []),
                ],
            )
            ba_out = batch_output_dir
        else:
            run_module_in_step_venv(
                "bertalign_step",
                "bertalign_step.run_bertalign",
                [
                    "--source-file", args.source_file,
                    "--target-file", args.target_file,
                    "--output-file", args.bertalign_output,
                    "--max-align", str(args.bertalign_max_align),
                    "--top-k", str(args.bertalign_top_k),
                    "--win", str(args.bertalign_win),
                    "--src-lang", args.bertalign_src_lang,
                    "--tgt-lang", args.bertalign_tgt_lang,
                ],
            )
            ba_out = args.bertalign_output

        run_module_in_step_venv(
            "termalign_step",
            "termalign_step.run_termalign",
            [
                "--bertalign-output", ba_out,
                "--output-file", args.termalign_output,
                "--extraction-mode", args.extraction_mode,
                "--termalign-mode", args.termalign_mode,
                "--min-pair-confidence", str(args.min_pair_confidence),
                "--top-k-pairs", str(args.top_k_pairs),
                "--aligner-model", args.aligner_model,
                "--zh-extractor-model", args.zh_extractor_model,
                "--en-extractor-model", args.en_extractor_model,
                *(["--api-endpoint", args.api_endpoint] if args.api_endpoint else []),
                *(["--api-key", args.api_key] if args.api_key else []),
                *(["--api-model", args.api_model] if args.api_model else []),
                *(["--api-prompt-file", args.api_prompt_file] if args.api_prompt_file else []),
                *(["--dict-zh-path", args.dict_zh_path] if args.dict_zh_path else []),
                *(["--dict-en-path", args.dict_en_path] if args.dict_en_path else []),
                *(["--skip-bert"] if args.skip_bert else []),
            ],
        )

        run_module_in_step_venv(
            "evaluation_step",
            "evaluation_step.evaluate_terms",
            [
                "--termalign-output", args.termalign_output,
                "--dictionary-path", args.dictionary_path,
                "--output-file", args.evaluation_output,
                "--mode", args.eval_mode,
                "--report-level", args.eval_report_level,
                "--alpha", str(args.eval_alpha),
                "--beta", str(args.eval_beta),
                "--metrics", *args.eval_metrics,
                *(["--target-txt", args.eval_target_txt] if args.eval_target_txt else []),
                *(["--target-dir", args.eval_target_dir] if args.eval_target_dir else []),
                *(["--debug-log", args.eval_debug_log] if args.eval_debug_log else []),
                *(["--debug-sublogs-dir", args.eval_debug_sublogs_dir] if args.eval_debug_sublogs_dir else []),
            ],
        )
        return args.evaluation_output

    logging.info("Execution mode: sequential")
    from bertalign_step.run_bertalign import run_bertalign
    from bertalign_step.batch_align import run_batch_alignment
    from termalign_step.run_termalign import run_termalign
    from evaluation_step.evaluate_terms import evaluate_terms
    if args.bertalign_batch_data_dir:
        batch_output_dir = args.bertalign_batch_output_dir or str(Path(args.data_dir) / "Task2" / "batch_tsv")
        code = run_batch_alignment(
            data_dir=args.bertalign_batch_data_dir,
            output_dir=batch_output_dir,
            src_lang=args.bertalign_src_lang,
            tgt_lang=args.bertalign_tgt_lang,
            max_align=args.bertalign_max_align,
            top_k=args.bertalign_top_k,
            win=args.bertalign_win,
            strict=args.bertalign_batch_strict,
        )
        if code != 0:
            raise RuntimeError(f"Batch bertalign failed with exit code={code}")
        ba_out = batch_output_dir
    else:
        ba_out = run_bertalign(
            source_file=args.source_file,
            target_file=args.target_file,
            output_file=args.bertalign_output,
            external_command=args.bertalign_command,
            default_confidence=args.bertalign_default_confidence,
            max_align=args.bertalign_max_align,
            top_k=args.bertalign_top_k,
            win=args.bertalign_win,
            src_lang=args.bertalign_src_lang,
            tgt_lang=args.bertalign_tgt_lang,
        )
    ta_out = run_termalign(
        bertalign_output=ba_out,
        output_file=args.termalign_output,
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
    )
    return evaluate_terms(
        termalign_output=ta_out,
        dictionary_path=args.dictionary_path,
        output_file=args.evaluation_output,
        mode=args.eval_mode,
        target_txt=args.eval_target_txt,
        target_dir=args.eval_target_dir,
        report_level=args.eval_report_level,
        metrics=args.eval_metrics,
        alpha=args.eval_alpha,
        beta=args.eval_beta,
        debug_log=args.eval_debug_log,
        debug_sublogs_dir=args.eval_debug_sublogs_dir,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "full":
        out = run_full(args)
        logging.info("Full pipeline finished. Output: %s", out)
        print(out)
        return

    args = apply_data_defaults(args)
    if getattr(args, "inputs_dir", None) and not getattr(args, "bertalign_batch_data_dir", None):
        args.bertalign_batch_data_dir = args.inputs_dir
    setup_run_logger(args.logs_dir, args.command)
    logging.info("Running command=%s data_dir=%s", args.command, args.data_dir)

    if args.command == "bertalign":
        from bertalign_step.run_bertalign import run_bertalign
        from bertalign_step.batch_align import run_batch_alignment
        if args.bertalign_batch_data_dir:
            batch_output_dir = args.bertalign_batch_output_dir or str(Path(args.data_dir) / "Task2" / "batch_tsv")
            code = run_batch_alignment(
                data_dir=args.bertalign_batch_data_dir,
                output_dir=batch_output_dir,
                src_lang=args.bertalign_src_lang,
                tgt_lang=args.bertalign_tgt_lang,
                max_align=args.bertalign_max_align,
                top_k=args.bertalign_top_k,
                win=args.bertalign_win,
                strict=args.bertalign_batch_strict,
            )
            if code != 0:
                raise RuntimeError(f"Batch bertalign failed with exit code={code}")
            out = batch_output_dir
        else:
            ensure_parent_dirs(args.bertalign_output)
            out = run_bertalign(
                source_file=args.source_file,
                target_file=args.target_file,
                output_file=args.bertalign_output,
                external_command=args.bertalign_command,
                default_confidence=args.bertalign_default_confidence,
                max_align=args.bertalign_max_align,
                top_k=args.bertalign_top_k,
                win=args.bertalign_win,
                src_lang=args.bertalign_src_lang,
                tgt_lang=args.bertalign_tgt_lang,
            )
        logging.info("bertalign finished. Output: %s", out)
        print(out)
        return

    if args.command == "termalign":
        from termalign_step.run_termalign import run_termalign
        ensure_parent_dirs(args.termalign_output)
        out = run_termalign(
            bertalign_output=args.bertalign_output,
            output_file=args.termalign_output,
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
        )
        logging.info("termalign finished. Output: %s", out)
        print(out)
        return

    if args.command == "evaluation":
        from evaluation_step.evaluate_terms import evaluate_terms
        ensure_parent_dirs(args.evaluation_output)
        out = evaluate_terms(
            termalign_output=args.termalign_output,
            dictionary_path=args.dictionary_path,
            output_file=args.evaluation_output,
            mode=args.eval_mode,
            target_txt=args.eval_target_txt,
            target_dir=args.eval_target_dir,
            report_level=args.eval_report_level,
            metrics=args.eval_metrics,
            alpha=args.eval_alpha,
            beta=args.eval_beta,
            debug_log=args.eval_debug_log,
            debug_sublogs_dir=args.eval_debug_sublogs_dir,
        )
        logging.info("evaluation finished. Output: %s", out)
        print(out)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
