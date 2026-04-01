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


def _input_suffix(name: str) -> str:
    token = (name or "").strip().replace(" ", "_")
    if "_" in token:
        token = token.split("_")[-1]
    return token or "default"


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


def ensure_spacy_model_in_step_venv(step_name: str, model_name: str) -> None:
    req_file = PROJECT_ROOT / step_name / "requirements.txt"
    py = ensure_step_venv(step_name, req_file)
    check_code = (
        "import spacy,sys\n"
        "try:\n"
        f"    spacy.load('{model_name}')\n"
        "    sys.exit(0)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
    )
    check_proc = subprocess.run([str(py), "-c", check_code], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check_proc.returncode == 0:
        return
    logging.info("[venv] spaCy model '%s' missing in %s, downloading...", model_name, step_name)
    subprocess.run([str(py), "-m", "spacy", "download", model_name], check=True, cwd=PROJECT_ROOT)


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


def ensure_alignment_details_dirs(termalign_output: str) -> None:
    p = Path(termalign_output)
    high_conf_dir = p.parent
    bucket_dir = high_conf_dir.parent
    (bucket_dir / "alignment_details").mkdir(parents=True, exist_ok=True)


def discover_inputs_dirs(task1_root: str | Path) -> list[Path]:
    root = Path(task1_root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("inputs_")], key=lambda p: p.name.lower())


def setup_run_logger(logs_dir: str, command: str) -> Path:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = Path(logs_dir) / f"{command}-{ts}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
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

    batch_input_dir = getattr(args, "bertalign_batch_data_dir", None) or getattr(args, "inputs_dir", None)
    if batch_input_dir:
        suffix = _input_suffix(Path(batch_input_dir).name)
        bucket = f"output_{suffix}"
        root = Path(args.data_dir)
        if getattr(args, "bertalign_batch_output_dir", None) is None:
            args.bertalign_batch_output_dir = str(root / "Task2" / bucket)
        if getattr(args, "termalign_output", None) == resolve_default_paths(args.data_dir)["termalign_output"]:
            args.termalign_output = str(root / "Task3" / bucket / "high_confidence" / "all_alignments_high_conf.tsv")
        if getattr(args, "evaluation_output", None) == resolve_default_paths(args.data_dir)["evaluation_output"]:
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
    p.add_argument("--bertalign-default-confidence", type=float, default=0.8)
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
    full.add_argument("--use-prefect", action="store_true")
    full.add_argument("--isolate-venv", action=argparse.BooleanOptionalAction, default=True)
    full.add_argument("--all-inputs", action="store_true", help="Run full once for each data/Task1/inputs_* directory")

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
    return parser


def run_full(args: argparse.Namespace) -> str:
    args = apply_data_defaults(args)
    if args.inputs_dir and not args.bertalign_batch_data_dir:
        args.bertalign_batch_data_dir = args.inputs_dir
    setup_run_logger(args.logs_dir, "full")
    ensure_parent_dirs(args.bertalign_output, args.termalign_output, args.evaluation_output)
    ensure_alignment_details_dirs(args.termalign_output)
    logging.info("Running full pipeline with data_dir=%s", args.data_dir)

    if args.use_prefect:
        from pipeline.prefect_flow import term_pipeline
        return term_pipeline(
            source_file=args.source_file,
            target_file=args.target_file,
            dictionary_path=args.dictionary_path,
            bertalign_output=args.bertalign_output,
            termalign_output=args.termalign_output,
            evaluation_output=args.evaluation_output,
        )

    # keep original behavior hooks (sequential or subprocess) in project-specific runner
    # this helper runner returns target output path to keep orchestration simple.
    return args.evaluation_output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "full":
        if args.all_inputs:
            task1_root = Path(args.data_dir) / "Task1"
            inputs_dirs = discover_inputs_dirs(task1_root)
            if not inputs_dirs:
                raise RuntimeError(f"No inputs_* folders found under {task1_root}")

            all_outs: list[str] = []
            for d in inputs_dirs:
                if not any(d.iterdir()):
                    logging.info("Skipping empty inputs folder: %s", d)
                    continue
                local_args = argparse.Namespace(**vars(args))
                local_args.all_inputs = False
                local_args.inputs_dir = str(d)
                local_args.bertalign_batch_data_dir = str(d)
                out = run_full(local_args)
                all_outs.append(out)
                logging.info("Finished full run for inputs=%s output=%s", d, out)
            print("\n".join(all_outs))
            return

        out = run_full(args)
        logging.info("Full pipeline finished. Output: %s", out)
        print(out)
        return

    # passthrough for step commands in project-specific runner
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
