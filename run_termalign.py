#!/usr/bin/env python3
"""Run termalign step using the provided termalign pipeline scripts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

HF_ZH_MODEL = "owen4512/bert-base-chinese-finance-term-extractor"
HF_EN_MODEL = "owen4512/bert-base-cased-finance-term-extractor"
HF_ALIGN_MODEL = "owen4512/minilm-finance-term-aligner"

DEFAULT_TERM_LIST_DIR = Path(__file__).resolve().parent / "term_list"
DEFAULT_ZH_TERM_LIST = DEFAULT_TERM_LIST_DIR / "zh_terms.txt"
DEFAULT_EN_TERM_LIST = DEFAULT_TERM_LIST_DIR / "en_terms.txt"
STANDALONE_TASK3_ROOT = Path(__file__).resolve().parent / "data" / "Task3"


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts)


def _ensure_embedding_runtime_versions() -> None:
    st_ver = importlib.metadata.version("sentence-transformers")
    tf_ver = importlib.metadata.version("transformers")

    if _parse_version(st_ver) < (5, 2, 2):
        raise RuntimeError("sentence-transformers too old")
    if _parse_version(tf_ver) < (4, 46, 0):
        raise RuntimeError("transformers too old")


def _ensure_hf_models_downloaded() -> tuple[str, str, str]:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    root = Path.home() / ".cache" / "termalign_hf_models"

    resolved = {
        HF_ZH_MODEL: HF_ZH_MODEL,
        HF_EN_MODEL: HF_EN_MODEL,
        HF_ALIGN_MODEL: HF_ALIGN_MODEL,
    }

    for repo_id in (HF_ZH_MODEL, HF_EN_MODEL, HF_ALIGN_MODEL):
        local_dir = root / repo_id.replace("/", "--")
        try:
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            resolved[repo_id] = str(local_dir)
        except Exception:
            continue

    return resolved[HF_ZH_MODEL], resolved[HF_EN_MODEL], resolved[HF_ALIGN_MODEL]


def _jsonl_to_tsv(input_jsonl: Path, output_tsv: Path) -> str:
    import pandas as pd

    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append({"src_text": row["source_segment"], "tgt_text": row["target_segment"]})

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def _prepare_termalign_input(input_path: Path, tmpdir: Path) -> Path:
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            tsv = tmpdir / "input.tsv"
            _jsonl_to_tsv(input_path, tsv)
            return tsv
        return input_path

    out = tmpdir / "batch"
    out.mkdir(parents=True, exist_ok=True)

    for f in input_path.iterdir():
        if f.suffix == ".jsonl":
            _jsonl_to_tsv(f, out / f"{f.stem}.tsv")
        elif f.suffix == ".tsv":
            shutil.copy2(f, out / f.name)

    return out


def _resolve_default_term_lists(dict_zh_path: str | None, dict_en_path: str | None) -> tuple[str | None, str | None]:
    zh = dict_zh_path or (str(DEFAULT_ZH_TERM_LIST) if DEFAULT_ZH_TERM_LIST.exists() else None)
    en = dict_en_path or (str(DEFAULT_EN_TERM_LIST) if DEFAULT_EN_TERM_LIST.exists() else None)
    return zh, en


def _resolve_output_dirs(output_file: Path, output_dir: str | None) -> tuple[Path, Path]:
    # To avoid generating files in unexpected locations, always colocate
    # alignment_details with output_file when output_file is provided.
    if output_file.parent.name == "high_confidence":
        return output_file, output_file.parent.parent / "alignment_details"
    return output_file, output_file.parent


def _suffix_from_task1_input(task1_input_dir: str | None) -> str:
    if not task1_input_dir:
        return "default"
    token = Path(task1_input_dir).name.strip().replace(" ", "_")
    if "_" in token:
        token = token.split("_")[-1]
    return token or "default"


def _resolve_task3_root(pipeline_run: bool, evaluation_pipeline_root: str | None) -> Path:
    if pipeline_run:
        base = Path(evaluation_pipeline_root).resolve() if evaluation_pipeline_root else Path.cwd().resolve()
        return base / "data" / "Task3"
    return STANDALONE_TASK3_ROOT


def _resolve_default_output_file(
    task1_input_dir: str | None,
    pipeline_run: bool,
    evaluation_pipeline_root: str | None,
) -> Path:
    suffix = _suffix_from_task1_input(task1_input_dir)
    task3_root = _resolve_task3_root(pipeline_run, evaluation_pipeline_root)
    return task3_root / f"output_{suffix}" / "high_confidence" / "all_alignments_high_conf.tsv"


def _organize_alignment_outputs(details_dir: Path, min_pair_confidence: float) -> tuple[Path | None, Path | None]:
    import pandas as pd

    generated_names = {
        "all_alignments.tsv",
        "all_alignments_high_conf.tsv",
        "all_allignments_high_conf.tsv",
    }

    all_rows = []
    high_rows = []
    per_file_outputs = 0

    for f in details_dir.rglob("*.tsv"):
        if f.parent == details_dir and f.name in generated_names:
            continue
        if f.parent.name == "high_confidence":
            continue

        try:
            df = pd.read_csv(f, sep="\t")
        except Exception:
            continue
        if df.empty:
            continue

        if "similarity" in df.columns:
            df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
        else:
            df["similarity"] = 0.0

        base = f.stem
        out_all = details_dir / f"{base}_all_alignment.tsv"
        out_high = details_dir / f"{base}_high_conf.tsv"

        df.to_csv(out_all, sep="\t", index=False)
        df_high = df[df["similarity"] >= min_pair_confidence]
        if not df_high.empty:
            df_high.to_csv(out_high, sep="\t", index=False)
            high_rows.append(df_high)

        all_rows.append(df)
        per_file_outputs += 1

    all_path = None
    all_high_path = None
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_path = details_dir / "all_alignments.tsv"
        all_df.to_csv(all_path, sep="\t", index=False)

        if high_rows:
            all_high_df = pd.concat(high_rows, ignore_index=True)
            all_high_path = details_dir / "all_alignments_high_conf.tsv"
            all_high_df.to_csv(all_high_path, sep="\t", index=False)
            # Keep compatibility with requested legacy typo file name.
            all_high_df.to_csv(details_dir / "all_allignments_high_conf.tsv", sep="\t", index=False)

    if per_file_outputs == 0:
        return None, None
    return all_path, all_high_path


def run_termalign(
    bertalign_output: str,
    output_file: str | None = None,
    output_dir: str | None = None,
    extraction_mode: str = "model",
    termalign_mode: str = "hf",
    min_term_confidence: float = 0.5,
    min_pair_confidence: float = 0.5,
    source_lang: str = "zh",
    target_lang: str = "en",
    top_k_pairs: int = 0,
    aligner_model: str = HF_ALIGN_MODEL,
    zh_extractor_model: str = HF_ZH_MODEL,
    en_extractor_model: str = HF_EN_MODEL,
    api_endpoint: str | None = None,
    api_key: str | None = None,
    device: int = -1,
    dict_zh_path: str | None = None,
    dict_en_path: str | None = None,
    skip_bert: bool = False,
    api_model: str | None = None,
    api_prompt_file: str | None = None,
    task1_input_dir: str | None = None,
    pipeline_run: bool = False,
    evaluation_pipeline_root: str | None = None,
) -> str:
    if termalign_mode == "hf":
        _ensure_embedding_runtime_versions()
        zh_extractor_model, en_extractor_model, aligner_model = _ensure_hf_models_downloaded()

    dict_zh_path, dict_en_path = _resolve_default_term_lists(dict_zh_path, dict_en_path)

    input_data = Path(bertalign_output)
    output_path = Path(output_file) if output_file else _resolve_default_output_file(
        task1_input_dir=task1_input_dir,
        pipeline_run=pipeline_run,
        evaluation_pipeline_root=evaluation_pipeline_root,
    )
    high_conf_output, details_dir = _resolve_output_dirs(output_path, output_dir)

    with tempfile.TemporaryDirectory() as tmp:
        from termalign_step.termalign_pipeline.pipeline import run_pipeline

        prepared = _prepare_termalign_input(input_data, Path(tmp))

        details_dir.mkdir(parents=True, exist_ok=True)
        high_conf_output.parent.mkdir(parents=True, exist_ok=True)

        run_pipeline(
            input_path=prepared,
            dict_zh_path=dict_zh_path,
            dict_en_path=dict_en_path,
            bert_model_zh=zh_extractor_model,
            bert_model_en=en_extractor_model,
            embed_model=aligner_model,
            similarity_threshold=min_pair_confidence,
            output_dir=details_dir,
            skip_bert=skip_bert,
        )

        all_path, all_high_path = _organize_alignment_outputs(details_dir, min_pair_confidence)

        src = all_high_path or all_path
        if src is None:
            raise RuntimeError(f"No alignment TSV generated under {details_dir}")
        shutil.copy2(src, high_conf_output)

    return str(high_conf_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TermAlign step")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task1-input-dir", default=None, help="Task1 input folder used to derive outputs_xxx suffix")
    parser.add_argument("--pipeline-run", action="store_true", help="Write outputs under evaluation_pipeline/data/task3")
    parser.add_argument("--evaluation-pipeline-root", default=None, help="Evaluation pipeline root path")
    parser.add_argument("--extraction-mode", choices=["model", "api"], default="model")
    parser.add_argument("--termalign-mode", choices=["api", "local", "hf"], default="hf")
    parser.add_argument("--min-term-confidence", type=float, default=0.5)
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--top-k-pairs", type=int, default=0)
    parser.add_argument("--aligner-model", default=HF_ALIGN_MODEL)
    parser.add_argument("--zh-extractor-model", default=HF_ZH_MODEL)
    parser.add_argument("--en-extractor-model", default=HF_EN_MODEL)
    parser.add_argument("--api-endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dict-zh-path", default=None)
    parser.add_argument("--dict-en-path", default=None)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-prompt-file", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_termalign(
        bertalign_output=args.bertalign_output,
        output_file=args.output_file,
        output_dir=args.output_dir,
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
        device=args.device,
        dict_zh_path=args.dict_zh_path,
        dict_en_path=args.dict_en_path,
        skip_bert=args.skip_bert,
        api_model=args.api_model,
        api_prompt_file=args.api_prompt_file,
        task1_input_dir=args.task1_input_dir,
        pipeline_run=args.pipeline_run,
        evaluation_pipeline_root=args.evaluation_pipeline_root,
    )


if __name__ == "__main__":
    main()
