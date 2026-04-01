#!/usr/bin/env python3
"""Helpers for unified pipeline runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_alignment_details_dirs(termalign_output: str) -> None:
    """
    Ensure flat alignment_details directory exists next to high_confidence output.
    """
    p = Path(termalign_output)
    high_conf_dir = p.parent
    bucket_dir = high_conf_dir.parent
    (bucket_dir / "alignment_details").mkdir(parents=True, exist_ok=True)


def discover_inputs_dirs(task1_root: str | Path) -> list[Path]:
    """
    Discover all Task1 batch folders that match `inputs_*`.
    """
    root = Path(task1_root)
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("inputs_")],
        key=lambda p: p.name.lower(),
    )


def run_full_once_per_inputs(
    *,
    task1_root: str | Path,
    run_full_with_inputs_dir,
    skip_empty: bool = True,
) -> list[str]:
    """
    Run full pipeline once for each `inputs_*` folder under Task1 root.

    `run_full_with_inputs_dir` is a callback receiving one argument: `inputs_dir` (str),
    and should execute one full pipeline run and return an output path string.
    """
    results: list[str] = []
    for d in discover_inputs_dirs(task1_root):
        if skip_empty and not any(d.iterdir()):
            continue
        out = run_full_with_inputs_dir(str(d))
        results.append(out)
    return results

