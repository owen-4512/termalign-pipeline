#!/usr/bin/env python3
"""Unified runner for single-step or full pipeline execution."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import subprocess
import os
import venv
from copy import deepcopy

HERE = Path(__file__).resolve()


def _detect_project_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "bertalign_step").exists() and (candidate / "termalign_step").exists():
            return candidate
    return start.parent


PROJECT_ROOT = _detect_project_root(HERE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_ROOT = PROJECT_ROOT / ".venvs"


def _input_suffix(name: str) -> str:
    token = (name or "").strip().replace(" ", "_")
    if "_" in token:
        token = token.split("_")[-1]
    return token or "default"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _requirements_signature(requirements_file: Path) -> str:
    if not requirements_file.exists():
        return ""
    stat = requirements_file.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _sync_step_requirements(py: Path, venv_dir: Path, requirements_file: Path) -> None:
    if not requirements_file.exists():
        return
    sig = _requirements_signature(requirements_file)
    stamp = venv_dir / ".requirements.sig"
    old = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
    if old == sig:
        return
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(requirements_file)], check=True)
    stamp.write_text(sig, encoding="utf-8")


def ensure_step_venv(step_name: str, requirements_file: Path) -> Path:
    venv_dir = VENV_ROOT / step_name
    py = _venv_python(venv_dir)
    if py.exists():
        _sync_step_requirements(py, venv_dir, requirements_file)
        return py

    logging.info("[venv] creating %s", venv_dir)
    venv.create(venv_dir, with_pip=True)
    py = _venv_python(venv_dir)
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    _sync_step_requirements(py, venv_dir, requirements_file)
    return py


def run_module_in_step_venv(step_name: str, module: str, args_list: list[str]) -> None:
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
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    if proc.stdout:
        logging.info("[%s stdout]\n%s", step_name, proc.stdout.strip())
    if proc.stderr:
        logging.info("[%s stderr]\n%s", step_name, proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed with code={proc.returncode}")


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


def resolve_visualization_defaults(data_dir: str) -> dict[str, str]:
    root = Path(data_dir)
    return {
        "visualization_results_dir": str(root / "results"),
        "visualization_output_figure": str(root / "results" / "weighted_consistency_vs_accuracy.png"),
        "visualization_output_table_csv": str(root / "results" / "weighted_scores.csv"),
        "visualization_title": "Evaluation: Weighted Consistency vs Weighted Accuracy",
    }


def ensure_parent_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def ensure_alignment_details_dirs(termalign_output: str) -> None:
    p = Path(termalign_output)
    (p.parent.parent / "alignment_details").mkdir(parents=True, exist_ok=True)


def discover_inputs_dirs(task1_root: str | Path) -> list[Path]:
    root = Path(task1_root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("inputs_")], key=lambda p: p.name.lower())


def setup_run_logger(logs_dir: str, command: str) -> Path:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = Path(logs_dir) / f"{command}-{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


def apply_data_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = resolve_default_paths(args.data_dir)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    for key, value in resolve_optional_paths(args.data_dir).items():
        if getattr(args, key, None) is None and Path(value).exists():
            setattr(args, key, value)
    for key, value in resolve_visualization_defaults(args.data_dir).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    batch_input_dir = getattr(args, "bertalign_batch_data_dir", None) or getattr(args, "inputs_dir", None)
    if batch_input_dir:
        suffix = _input_suffix(Path(batch_input_dir).name)
        bucket = f"output_{suffix}"
        root = Path(args.data_dir)
        if getattr(args, "bertalign_batch_output_dir", None) is None:
            args.bertalign_batch_output_dir = str(root / "Task2" / bucket)
        if getattr(args, "termalign_output", None) == defaults["termalign_output"]:
            args.termalign_output = str(root / "Task3" / bucket / "high_confidence" / "all_alignments_high_conf.tsv")
        if getattr(args, "evaluation_output", None) == defaults["evaluation_output"]:
            args.evaluation_output = str(root / "results" / bucket / "evaluation_result.json")
    return args


def add_data_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data")
    p.add_argument("--logs-dir", default="logs")


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
    p.add_argument("--bertalign-max-align", type=int, default=3)
    p.add_argument("--bertalign-top-k", type=int, default=5)
    p.add_argument("--bertalign-win", type=int, default=8)
    p.add_argument("--bertalign-src-lang", default="zh")
    p.add_argument("--bertalign-tgt-lang", default="en")
    p.add_argument("--inputs-dir", default=None)
    p.add_argument("--bertalign-batch-data-dir", default=None)
    p.add_argument("--bertalign-batch-output-dir", default=None)
    p.add_argument("--bertalign-batch-strict", action="store_true")


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
    p.add_argument("--api-model", default=None)
    p.add_argument("--api-prompt-file", default=None)


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-mode", choices=["simple", "batch"], default="batch")
    p.add_argument("--eval-target-txt", default=None)
    p.add_argument("--eval-target-dir", default=None)
    p.add_argument("--eval-report-level", choices=["batch"], default="batch")
    p.add_argument("--eval-metrics", nargs="+", default=["all"])
    p.add_argument("--eval-alpha", type=float, default=0.2)
    p.add_argument("--eval-debug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-debug-sublogs-dir", default=None)


def add_visualization_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--visualization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--visualization-evaluation-files", nargs="*", default=None)
    p.add_argument("--visualization-results-dir", default=None)
    p.add_argument("--visualization-output-figure", default=None)
    p.add_argument("--visualization-output-table-csv", default=None)
    p.add_argument("--visualization-title", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full")
    add_common_args(full)
    add_bertalign_args(full)
    add_termalign_args(full)
    add_eval_args(full)
    add_visualization_args(full)
    full.add_argument("--isolate-venv", action=argparse.BooleanOptionalAction, default=True)

    ba = sub.add_parser("bertalign")
    add_data_dir_arg(ba)
    ba.add_argument("--source-file", default=None)
    ba.add_argument("--target-file", default=None)
    ba.add_argument("--bertalign-output", default=None)
    add_bertalign_args(ba)

    ta = sub.add_parser("termalign")
    add_data_dir_arg(ta)
    ta.add_argument("--bertalign-output", default=None)
    ta.add_argument("--termalign-output", default=None)
    add_termalign_args(ta)

    ev = sub.add_parser("evaluation")
    add_data_dir_arg(ev)
    ev.add_argument("--termalign-output", default=None)
    ev.add_argument("--dictionary-path", default=None)
    ev.add_argument("--evaluation-output", default=None)
    add_eval_args(ev)
    add_visualization_args(ev)
    ev.add_argument("--isolate-venv", action=argparse.BooleanOptionalAction, default=True)

    vz = sub.add_parser("visualization")
    add_data_dir_arg(vz)
    add_visualization_args(vz)
    vz.add_argument("--isolate-venv", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _eval_cli_args(args: argparse.Namespace) -> list[str]:
    return [
        "--termalign-output", args.termalign_output,
        "--dictionary-path", args.dictionary_path,
        "--output-file", args.evaluation_output,
        "--mode", args.eval_mode,
        "--report-level", args.eval_report_level,
        "--alpha", str(args.eval_alpha),
        "--metrics", *args.eval_metrics,
        *( ["--target-txt", args.eval_target_txt] if args.eval_target_txt else []),
        *( ["--target-dir", args.eval_target_dir] if args.eval_target_dir else []),
        *( ["--no-debug"] if not args.eval_debug else []),
        *( ["--debug-sublogs-dir", args.eval_debug_sublogs_dir] if args.eval_debug_sublogs_dir else []),
    ]


def _effective_visualization_files(args: argparse.Namespace) -> list[str] | None:
    if args.visualization_evaluation_files:
        return args.visualization_evaluation_files
    candidate = getattr(args, "evaluation_output", None)
    if candidate and Path(candidate).exists():
        return [candidate]
    return None


def _visualization_cli_args(args: argparse.Namespace) -> list[str]:
    evaluation_files = _effective_visualization_files(args)
    return [
        *( ["--evaluation-files", *evaluation_files] if evaluation_files else []),
        "--results-dir", args.visualization_results_dir,
        "--output-figure", args.visualization_output_figure,
        "--output-table-csv", args.visualization_output_table_csv,
        "--title", args.visualization_title,
    ]


def _run_visualization(args: argparse.Namespace) -> str:
    if args.isolate_venv:
        run_module_in_step_venv("visualization_step", "visualization_step.visualization", _visualization_cli_args(args))
        return args.visualization_output_figure

    from visualization_step.visualization import visualize_evaluation_results

    return visualize_evaluation_results(
        evaluation_files=_effective_visualization_files(args),
        results_dir=args.visualization_results_dir,
        output_figure=args.visualization_output_figure,
        output_table_csv=args.visualization_output_table_csv,
        title=args.visualization_title,
    )


def run_full(args: argparse.Namespace) -> str:
    args = apply_data_defaults(args)
    if args.inputs_dir and not args.bertalign_batch_data_dir:
        args.bertalign_batch_data_dir = args.inputs_dir

    ensure_parent_dirs(args.bertalign_output, args.termalign_output, args.evaluation_output)
    ensure_alignment_details_dirs(args.termalign_output)

    if args.isolate_venv:
        if args.bertalign_batch_data_dir:
            batch_output_dir = args.bertalign_batch_output_dir or str(Path(args.data_dir) / "Task2" / "batch_tsv")
            run_module_in_step_venv("bertalign_step", "bertalign_step.batch_align", [
                "--data-dir", args.bertalign_batch_data_dir,
                "--output-dir", batch_output_dir,
                "--src-lang", args.bertalign_src_lang,
                "--tgt-lang", args.bertalign_tgt_lang,
                "--max-align", str(args.bertalign_max_align),
                "--top-k", str(args.bertalign_top_k),
                "--win", str(args.bertalign_win),
                *( ["--strict"] if args.bertalign_batch_strict else []),
            ])
            ba_out = batch_output_dir
        else:
            run_module_in_step_venv("bertalign_step", "bertalign_step.run_bertalign", [
                "--source-file", args.source_file,
                "--target-file", args.target_file,
                "--output-file", args.bertalign_output,
                "--max-align", str(args.bertalign_max_align),
                "--top-k", str(args.bertalign_top_k),
                "--win", str(args.bertalign_win),
                "--src-lang", args.bertalign_src_lang,
                "--tgt-lang", args.bertalign_tgt_lang,
            ])
            ba_out = args.bertalign_output

        run_module_in_step_venv("termalign_step", "termalign_step.run_termalign", [
            "--bertalign-output", ba_out,
            "--output-file", args.termalign_output,
            "--extraction-mode", args.extraction_mode,
            "--termalign-mode", args.termalign_mode,
            "--min-pair-confidence", str(args.min_pair_confidence),
            "--top-k-pairs", str(args.top_k_pairs),
            "--aligner-model", args.aligner_model,
            "--zh-extractor-model", args.zh_extractor_model,
            "--en-extractor-model", args.en_extractor_model,
            *( ["--api-endpoint", args.api_endpoint] if args.api_endpoint else []),
            *( ["--api-key", args.api_key] if args.api_key else []),
            *( ["--api-model", args.api_model] if args.api_model else []),
            *( ["--api-prompt-file", args.api_prompt_file] if args.api_prompt_file else []),
            *( ["--dict-zh-path", args.dict_zh_path] if args.dict_zh_path else []),
            *( ["--dict-en-path", args.dict_en_path] if args.dict_en_path else []),
            *( ["--skip-bert"] if args.skip_bert else []),
        ])

        run_module_in_step_venv("evaluation_step", "evaluation_step.evaluate_terms", _eval_cli_args(args))
        if args.visualization:
            _run_visualization(args)
        return args.evaluation_output

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

    out = evaluate_terms(
        termalign_output=ta_out,
        dictionary_path=args.dictionary_path,
        output_file=args.evaluation_output,
        mode=args.eval_mode,
        target_txt=args.eval_target_txt,
        target_dir=args.eval_target_dir,
        report_level=args.eval_report_level,
        metrics=args.eval_metrics,
        alpha=args.eval_alpha,
        include_debug=args.eval_debug,
        debug_sublogs_dir=args.eval_debug_sublogs_dir,
    )
    if args.visualization:
        _run_visualization(args)
    return out


def run_full_over_all_inputs(args: argparse.Namespace) -> tuple[list[str], list[tuple[str, str]]]:
    task1_root = Path(args.data_dir) / "Task1"
    inputs_dirs = discover_inputs_dirs(task1_root)
    if not inputs_dirs:
        raise FileNotFoundError(f"No inputs_* folders found under: {task1_root}")

    results: list[str] = []
    failures: list[tuple[str, str]] = []
    for inputs_dir in inputs_dirs:
        if not any(inputs_dir.iterdir()):
            continue
        run_args = deepcopy(args)
        run_args.inputs_dir = str(inputs_dir)
        run_args.bertalign_batch_data_dir = str(inputs_dir)
        try:
            results.append(run_full(run_args))
        except Exception as e:
            failures.append((inputs_dir.name, str(e)))
            logging.exception("Pipeline failed for %s", inputs_dir.name)
    return results, failures


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_run_logger(args.logs_dir, args.command)

    if args.command == "full":
        if args.inputs_dir or args.bertalign_batch_data_dir:
            print(run_full(args))
            return
        results, failures = run_full_over_all_inputs(args)
        for out in results:
            print(out)
        if failures:
            print("FAILED INPUTS:")
            for name, err in failures:
                print(f"{name}: {err}")
        return

    args = apply_data_defaults(args)

    if args.command == "visualization":
        print(_run_visualization(args))
        return

    if args.command == "bertalign":
        from bertalign_step.run_bertalign import run_bertalign
        ensure_parent_dirs(args.bertalign_output)
        out = run_bertalign(
            source_file=args.source_file,
            target_file=args.target_file,
            output_file=args.bertalign_output,
            external_command=args.bertalign_command,
            max_align=args.bertalign_max_align,
            top_k=args.bertalign_top_k,
            win=args.bertalign_win,
            src_lang=args.bertalign_src_lang,
            tgt_lang=args.bertalign_tgt_lang,
        )
        print(out)
        return

    if args.command == "termalign":
        from termalign_step.run_termalign import run_termalign
        ensure_parent_dirs(args.termalign_output)
        ensure_alignment_details_dirs(args.termalign_output)
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
            include_debug=args.eval_debug,
            debug_sublogs_dir=args.eval_debug_sublogs_dir,
        )
        if args.visualization:
            _run_visualization(args)
        print(out)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
