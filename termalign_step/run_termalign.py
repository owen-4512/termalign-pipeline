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
from typing import Any

HF_ZH_MODEL = "owen4512/bert-base-chinese-finance-term-extractor"
HF_EN_MODEL = "owen4512/bert-base-cased-finance-term-extractor"
HF_ALIGN_MODEL = "owen4512/minilm-finance-term-aligner"
DEFAULT_TERM_LIST_DIR = Path(__file__).resolve().parent / "term_list"
DEFAULT_ZH_TERM_LIST = DEFAULT_TERM_LIST_DIR / "zh_terms.txt"
DEFAULT_EN_TERM_LIST = DEFAULT_TERM_LIST_DIR / "en_terms.txt"


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts)


def _ensure_embedding_runtime_versions() -> None:
    """Fail fast when local runtime is too old for current HF aligner exports."""
    try:
        st_ver = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. "
            "Please run: pip install -U 'sentence-transformers>=5.2.2'"
        ) from exc
    try:
        tf_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. "
            "Please run: pip install -U 'transformers>=4.46.0'"
        ) from exc

    if _parse_version(st_ver) < (5, 2, 2):
        raise RuntimeError(
            f"sentence-transformers=={st_ver} is too old for this HF aligner. "
            "Please upgrade: pip install -U 'sentence-transformers>=5.2.2' 'transformers>=4.46.0'"
        )
    if _parse_version(tf_ver) < (4, 46, 0):
        raise RuntimeError(
            f"transformers=={tf_ver} is too old for this HF aligner. "
            "Please upgrade: pip install -U 'transformers>=4.46.0'"
        )


def _ensure_hf_models_downloaded() -> tuple[str, str, str]:
    """Pre-download required Hugging Face models for hf mode.

    On some Windows environments, creating symlinks in Hugging Face cache can
    fail without admin/developer privileges (WinError 1314). In that case we
    gracefully fall back to repo-id lazy loading.
    """
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    download_root = Path.home() / ".cache" / "termalign_hf_models"
    resolved: dict[str, str] = {
        HF_ZH_MODEL: HF_ZH_MODEL,
        HF_EN_MODEL: HF_EN_MODEL,
        HF_ALIGN_MODEL: HF_ALIGN_MODEL,
    }

    for repo_id in (HF_ZH_MODEL, HF_EN_MODEL, HF_ALIGN_MODEL):
        local_dir = download_root / repo_id.replace("/", "--")
        try:
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                )
            except TypeError:
                # Compatibility fallback.
                snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            resolved[repo_id] = str(local_dir)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "winerror", None) == 1314:
                print(
                    "⚠️ Hugging Face cache symlink permission issue detected on Windows "
                    f"while pre-downloading '{repo_id}'. Falling back to repo-id loading."
                )
                continue
            print(f"⚠️ Failed to pre-download '{repo_id}': {exc}. Falling back to repo-id loading.")
            continue

    return resolved[HF_ZH_MODEL], resolved[HF_EN_MODEL], resolved[HF_ALIGN_MODEL]


def _jsonl_to_tsv(input_jsonl: Path, output_tsv: Path) -> str:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append({"src_text": row["source_segment"], "tgt_text": row["target_segment"]})
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_tsv, sep="\t", index=False)
    return str(output_tsv)


def _prepare_termalign_input(input_path: Path, tmpdir_path: Path) -> Path:
    """Prepare termalign input as TSV file or TSV directory.

    Supports:
    - single JSONL (converted to one TSV),
    - single TSV (used as-is),
    - directory containing JSONL/TSV (normalized into a TSV directory).
    """
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            tsv_path = tmpdir_path / "bertalign_input.tsv"
            _jsonl_to_tsv(input_path, tsv_path)
            return tsv_path
        if input_path.suffix.lower() == ".tsv":
            return input_path
        raise ValueError(f"Unsupported bertalign input file type: {input_path}")

    if input_path.is_dir():
        out_dir = tmpdir_path / "bertalign_batch_tsv"
        out_dir.mkdir(parents=True, exist_ok=True)
        found = False
        for child in sorted(input_path.iterdir()):
            if not child.is_file():
                continue
            suffix = child.suffix.lower()
            if suffix == ".tsv":
                shutil.copy2(child, out_dir / child.name)
                found = True
            elif suffix == ".jsonl":
                _jsonl_to_tsv(child, out_dir / f"{child.stem}.tsv")
                found = True
        if not found:
            raise ValueError(f"No .tsv/.jsonl bertalign files found under directory: {input_path}")
        return out_dir

    raise FileNotFoundError(f"bertalign input not found: {input_path}")


def _resolve_default_term_lists(
    dict_zh_path: str | None,
    dict_en_path: str | None,
) -> tuple[str | None, str | None]:
    zh = dict_zh_path
    en = dict_en_path
    if zh is None and DEFAULT_ZH_TERM_LIST.exists():
        zh = str(DEFAULT_ZH_TERM_LIST)
    if en is None and DEFAULT_EN_TERM_LIST.exists():
        en = str(DEFAULT_EN_TERM_LIST)
    return zh, en


def _alignment_tsv_to_jsonl(align_tsv: Path, output_jsonl: Path, min_similarity: float, top_k_pairs: int) -> str:
    import pandas as pd

    df = pd.read_csv(align_tsv, sep="\t")
    if "similarity" in df.columns:
        df = df[df["similarity"].astype(float) >= min_similarity]
    df = df.sort_values(by="similarity", ascending=False)
    if top_k_pairs > 0:
        df = df.head(top_k_pairs)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            weighted_conf = float(row.get("similarity", 0.0))
            out = {
                "source_term": str(row.get("zh_term", "")),
                "target_term": str(row.get("en_term", "")),
                "model_pair_confidence": weighted_conf,
                "weighted_confidence": weighted_conf,
                "source_sentence": str(row.get("zh_sentence", "")),
                "target_sentence": str(row.get("en_sentence", "")),
                "zh_source": row.get("zh_source"),
                "en_source": row.get("en_source"),
                "zh_confidence": row.get("zh_confidence"),
                "en_confidence": row.get("en_confidence"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return str(output_jsonl)


def _resolve_output_dirs(output_file: Path, output_dir: str | None) -> tuple[Path, Path]:
    """Resolve high-confidence output path and alignment-detail output directory.

    - high-confidence output: `output_file`
    - alignment details:
      - explicit `output_dir` if provided
      - otherwise, if output file is under `.../high_confidence/`, use sibling
        `.../alignment_details/`
      - fallback to `output_file.parent`
    """
    high_conf_output = output_file
    if output_dir:
        details_dir = Path(output_dir)
    elif output_file.parent.name == "high_confidence":
        details_dir = output_file.parent.parent / "alignment_details"
    else:
        details_dir = output_file.parent
    return high_conf_output, details_dir


def run_termalign(
    bertalign_output: str,
    output_file: str,
    output_dir: str | None = None,
    extraction_mode: str = "model",
    termalign_mode: str = "hf",
    min_term_confidence: float = 0.5,
    min_pair_confidence: float = 0.5,
    source_lang: str = "zh",
    target_lang: str = "en",
    top_k_pairs: int = 0,
    aligner_model: str = "owen4512/minilm-finance-term-aligner",
    zh_extractor_model: str = "owen4512/bert-base-chinese-finance-term-extractor",
    en_extractor_model: str = "owen4512/bert-base-cased-finance-term-extractor",
    api_endpoint: str | None = None,
    api_key: str | None = None,
    device: int = -1,
    dict_zh_path: str | None = None,
    dict_en_path: str | None = None,
    skip_bert: bool = False,
    api_model: str | None = None,
    api_prompt_file: str | None = None,
) -> str:
    del min_term_confidence, source_lang, target_lang, device

    mode = termalign_mode
    if extraction_mode == "api":
        mode = "api"
    elif extraction_mode == "model" and termalign_mode == "hf":
        mode = "hf"

    if mode == "api":
        if not api_endpoint:
            raise ValueError("api_endpoint is required when extraction_mode='api'.")
        from termalign_step.run_termalign_api import run_termalign_api
        prompt_text = Path(api_prompt_file).read_text(encoding="utf-8") if api_prompt_file else None
        return run_termalign_api(
            bertalign_output=bertalign_output,
            output_file=output_file,
            api_endpoint=api_endpoint,
            api_key=api_key,
            min_pair_confidence=min_pair_confidence,
            top_k_pairs=top_k_pairs,
            api_model=api_model,
            prompt_text=prompt_text,
        )

    if mode == "hf":
        _ensure_embedding_runtime_versions()
        zh_extractor_model, en_extractor_model, aligner_model = _ensure_hf_models_downloaded()

    dict_zh_path, dict_en_path = _resolve_default_term_lists(dict_zh_path, dict_en_path)

    input_data = Path(bertalign_output)
    output_path = Path(output_file)
    high_conf_output, details_output_dir = _resolve_output_dirs(output_path, output_dir)

    with tempfile.TemporaryDirectory(prefix="termalign-") as tmpdir:
        from termalign_step.termalign_pipeline.pipeline import run_pipeline

        tmpdir_path = Path(tmpdir)
        prepared_input = _prepare_termalign_input(input_data, tmpdir_path)
        details_output_dir.mkdir(parents=True, exist_ok=True)
        high_conf_output.parent.mkdir(parents=True, exist_ok=True)

        run_pipeline(
            input_path=prepared_input,
            dict_zh_path=dict_zh_path,
            dict_en_path=dict_en_path,
            bert_model_zh=zh_extractor_model,
            bert_model_en=en_extractor_model,
            embed_model=aligner_model,
            similarity_threshold=min_pair_confidence,
            output_dir=details_output_dir,
            skip_bert=skip_bert,
        )

        preferred = details_output_dir / "all_alignments_high_conf.tsv"
        if not preferred.exists():
            preferred = details_output_dir / "alignments_high_conf.tsv"
        if not preferred.exists():
            preferred = details_output_dir / "all_alignments.tsv"
        if not preferred.exists():
            preferred = details_output_dir / "alignments.tsv"

        if preferred != high_conf_output:
            shutil.copy2(preferred, high_conf_output)
        return str(high_conf_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TermAlign step")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data" / "outputs"))
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
    parser.add_argument("--api-prompt-file", default=None, help="Read API prompt from txt file")
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
    )


if __name__ == "__main__":
    main()
