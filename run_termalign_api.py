#!/usr/bin/env python3
"""API-based termalign step.

Expected API response schema (per segment pair):
{
  "pairs": [
    {
      "source_term": "...",
      "target_term": "...",
      "confidence": 0.87,
      "source_sentence": "...",   # optional
      "target_sentence": "..."    # optional
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

STANDALONE_TASK3_ROOT = Path(__file__).resolve().parent / "data" / "task3"
TSV_FIELDS = [
    "source_file",
    "zh_term",
    "en_term",
    "similarity",
    "zh_source",
    "en_source",
    "zh_confidence",
    "en_confidence",
    "zh_sentence",
    "en_sentence",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_rows(bertalign_output: Path) -> list[dict[str, Any]]:
    if bertalign_output.is_file():
        if bertalign_output.suffix == ".jsonl":
            return _read_jsonl(bertalign_output)
        if bertalign_output.suffix == ".tsv":
            return _read_tsv(bertalign_output)
        raise ValueError(f"Unsupported input suffix: {bertalign_output}")

    rows: list[dict[str, Any]] = []
    for f in bertalign_output.iterdir():
        if f.suffix not in {".jsonl", ".tsv"}:
            continue
        file_rows = _read_jsonl(f) if f.suffix == ".jsonl" else _read_tsv(f)
        for r in file_rows:
            r.setdefault("source_file", f.stem)
        rows.extend(file_rows)
    return rows


def _resolve_output_dirs(output_file: Path, output_dir: str | None) -> tuple[Path, Path]:
    if output_dir:
        return output_file, Path(output_dir)
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
        return base / "data" / "task3"
    return STANDALONE_TASK3_ROOT


def _resolve_default_output_file(
    task1_input_dir: str | None,
    pipeline_run: bool,
    evaluation_pipeline_root: str | None,
) -> Path:
    suffix = _suffix_from_task1_input(task1_input_dir)
    task3_root = _resolve_task3_root(pipeline_run, evaluation_pipeline_root)
    return task3_root / f"outputs_{suffix}" / "high_confidence" / "all_alignments_high_conf.tsv"


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _to_std_row(item: dict[str, Any], source_file: str, source_sentence: str, target_sentence: str) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "zh_term": item.get("source_term", ""),
        "en_term": item.get("target_term", ""),
        "similarity": item.get("weighted_confidence", 0.0),
        "zh_source": "api",
        "en_source": "api",
        "zh_confidence": "",
        "en_confidence": "",
        "zh_sentence": item.get("source_sentence", source_sentence),
        "en_sentence": item.get("target_sentence", target_sentence),
    }


def run_termalign_api(
    bertalign_output: str,
    output_file: str | None = None,
    api_endpoint: str = "",
    api_key: str | None = None,
    min_pair_confidence: float = 0.5,
    top_k_pairs: int = 0,
    timeout: int = 120,
    api_model: str | None = None,
    prompt_text: str | None = None,
    output_dir: str | None = None,
    task1_input_dir: str | None = None,
    pipeline_run: bool = False,
    evaluation_pipeline_root: str | None = None,
) -> str:
    import requests

    if not api_endpoint:
        raise ValueError("api_endpoint is required")

    rows = _load_rows(Path(bertalign_output))
    resolved_output = Path(output_file) if output_file else _resolve_default_output_file(
        task1_input_dir=task1_input_dir,
        pipeline_run=pipeline_run,
        evaluation_pipeline_root=evaluation_pipeline_root,
    )
    high_conf_output, details_dir = _resolve_output_dirs(resolved_output, output_dir)
    high_conf_output.parent.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    per_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_segment = row.get("source_segment") or row.get("src_text") or ""
        target_segment = row.get("target_segment") or row.get("tgt_text") or ""
        source_file = str(row.get("source_file") or row.get("file") or "__default__")
        align_conf = float(row.get("alignment_confidence", 1.0) or 1.0)

        payload = {
            "source_segment": source_segment,
            "target_segment": target_segment,
            "alignment_confidence": align_conf,
            "min_pair_confidence": min_pair_confidence,
        }
        if api_model:
            payload["model"] = api_model
        if prompt_text:
            payload["prompt"] = prompt_text

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        pairs = data.get("pairs", []) if isinstance(data, dict) else []
        for pair in pairs:
            conf = float(pair.get("confidence", 0.0) or 0.0)
            weighted = conf * align_conf
            if weighted < min_pair_confidence:
                continue
            item = {
                "source_term": pair.get("source_term", ""),
                "target_term": pair.get("target_term", ""),
                "weighted_confidence": weighted,
                "source_sentence": pair.get("source_sentence", source_segment),
                "target_sentence": pair.get("target_sentence", target_segment),
            }
            per_file.setdefault(source_file, []).append(_to_std_row(item, source_file, source_segment, target_segment))

    all_rows: list[dict[str, Any]] = []
    all_high_rows: list[dict[str, Any]] = []
    for source_file, file_rows in per_file.items():
        file_rows.sort(key=lambda x: float(x["similarity"]), reverse=True)
        if top_k_pairs > 0:
            file_rows = file_rows[:top_k_pairs]
        _write_tsv(details_dir / f"{source_file}_all_alignment.tsv", file_rows)

        high_rows = [r for r in file_rows if float(r["similarity"]) >= min_pair_confidence]
        if high_rows:
            _write_tsv(details_dir / f"{source_file}_high_conf.tsv", high_rows)
            all_high_rows.extend(high_rows)

        all_rows.extend(file_rows)

    all_rows.sort(key=lambda x: float(x["similarity"]), reverse=True)
    _write_tsv(details_dir / "all_alignments.tsv", all_rows)

    high_src = details_dir / "all_alignments.tsv"
    if all_high_rows:
        all_high_rows.sort(key=lambda x: float(x["similarity"]), reverse=True)
        high_main = details_dir / "all_alignments_high_conf.tsv"
        _write_tsv(high_main, all_high_rows)
        _write_tsv(details_dir / "all_allignments_high_conf.tsv", all_high_rows)
        high_src = high_main

    shutil.copy2(high_src, high_conf_output)
    return str(high_conf_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="TermAlign API mode")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task1-input-dir", default=None, help="Task1 input folder used to derive outputs_xxx suffix")
    parser.add_argument("--pipeline-run", action="store_true", help="Write outputs under evaluation_pipeline/data/task3")
    parser.add_argument("--evaluation-pipeline-root", default=None, help="Evaluation pipeline root path")
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--top-k-pairs", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--prompt-file", default=None, help="Read API prompt from txt file")
    args = parser.parse_args()
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None

    run_termalign_api(
        bertalign_output=args.bertalign_output,
        output_file=args.output_file,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        min_pair_confidence=args.min_pair_confidence,
        top_k_pairs=args.top_k_pairs,
        timeout=args.timeout,
        api_model=args.api_model,
        prompt_text=prompt_text,
        output_dir=args.output_dir,
        task1_input_dir=args.task1_input_dir,
        pipeline_run=args.pipeline_run,
        evaluation_pipeline_root=args.evaluation_pipeline_root,
    )


if __name__ == "__main__":
    main()
