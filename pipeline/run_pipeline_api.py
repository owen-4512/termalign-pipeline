#!/usr/bin/env python3
"""Convenience entrypoint: run full pipeline in API mode for termalign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.runner import run_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline with termalign API mode")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--dictionary-path", default=None)
    parser.add_argument("--bertalign-output", default=None)
    parser.add_argument("--termalign-output", default=None)
    parser.add_argument("--evaluation-output", default=None)

    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-prompt-file", default=None, help="Prompt txt path")
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--top-k-pairs", type=int, default=0)

    parser.add_argument("--bertalign-command", default=None)
    parser.add_argument("--bertalign-default-confidence", type=float, default=0.8)
    parser.add_argument("--bertalign-max-align", type=int, default=3)
    parser.add_argument("--bertalign-top-k", type=int, default=5)
    parser.add_argument("--bertalign-win", type=int, default=8)
    parser.add_argument("--bertalign-src-lang", default="zh")
    parser.add_argument("--bertalign-tgt-lang", default="en")

    parser.add_argument("--eval-min-confidence", type=float, default=0.0)
    parser.add_argument("--eval-confidence-field", default="weighted_confidence")
    parser.add_argument("--eval-count-field", default=None)
    parser.add_argument("--eval-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--eval-entropy-base", type=float, default=2.0)
    parser.add_argument("--eval-normalize-entropy", action="store_true")
    parser.add_argument("--eval-top-k-variants", type=int, default=None)

    args = parser.parse_args()

    # Reuse run_full contract from runner by setting equivalent fields.
    args.command = "full"
    args.use_prefect = False
    args.extraction_mode = "api"
    args.termalign_mode = "api"
    args.min_term_confidence = 0.0
    args.source_lang = "zh"
    args.target_lang = "en"
    args.aligner_model = ""
    args.zh_extractor_model = ""
    args.en_extractor_model = ""
    args.device = -1
    args.dict_zh_path = None
    args.dict_en_path = None
    args.skip_bert = False

    out = run_full(args)
    print(out)


if __name__ == "__main__":
    main()
