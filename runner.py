#!/usr/bin/env python3
"""Runner helpers for pipeline output paths."""

from __future__ import annotations

from pathlib import Path


def ensure_alignment_details_dirs(termalign_output: str) -> None:
    """
    Ensure flat alignment_details directory exists next to high_confidence output.
    Never create legacy nested folders:
    - per_file
    - per_file_high_conf
    - all_files

    Examples:
      data/Task3/high_confidence/all_alignments_high_conf.tsv
        -> data/Task3/alignment_details/

      data/Task3/output_gpt/high_confidence/all_alignments_high_conf.tsv
        -> data/Task3/output_gpt/alignment_details/
    """
    p = Path(termalign_output)
    high_conf_dir = p.parent
    bucket_dir = high_conf_dir.parent
    alignment_root = bucket_dir / "alignment_details"
    alignment_root.mkdir(parents=True, exist_ok=True)
