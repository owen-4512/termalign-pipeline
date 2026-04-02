#!/usr/bin/env python3
"""Convenience entrypoint: run full pipeline in API mode for termalign.

Optimized approach:
- Reuse `runner.build_parser()` instead of duplicating dozens of CLI args.
- Force API-mode defaults in one place.
- Keep compatibility with runner's evolving options automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runner import build_parser, run_full, setup_run_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(["full", *argv])

    # Force API mode for termalign stage.
    args.command = "full"
    args.extraction_mode = "api"
    args.termalign_mode = "api"
    args.min_term_confidence = 0.0

    # API mode requires endpoint.
    if not getattr(args, "api_endpoint", None):
        raise SystemExit("--api-endpoint is required for runner_pipeline_api.py")

    # Keep language defaults aligned with API runner behavior.
    if not getattr(args, "source_lang", None):
        args.source_lang = "zh"
    if not getattr(args, "target_lang", None):
        args.target_lang = "en"

    # API mode doesn't use local HF aligner/extractors.
    args.aligner_model = ""
    args.zh_extractor_model = ""
    args.en_extractor_model = ""
    args.device = -1
    args.dict_zh_path = None
    args.dict_en_path = None
    args.skip_bert = False

    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])
    setup_run_logger(args.logs_dir, "full-api")
    out = run_full(args)
    print(out)


if __name__ == "__main__":
    main()
