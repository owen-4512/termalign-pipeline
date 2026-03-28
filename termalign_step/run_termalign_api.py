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
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_termalign_api(
    bertalign_output: str,
    output_file: str,
    api_endpoint: str,
    api_key: str | None = None,
    min_pair_confidence: float = 0.5,
    top_k_pairs: int = 0,
    timeout: int = 120,
) -> str:
    import requests

    rows = _read_jsonl(Path(bertalign_output))
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    all_pairs: list[dict[str, Any]] = []
    for row in rows:
        payload = {
            "source_segment": row.get("source_segment", ""),
            "target_segment": row.get("target_segment", ""),
            "alignment_confidence": float(row.get("alignment_confidence", 1.0)),
            "min_pair_confidence": min_pair_confidence,
        }
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        pairs = data.get("pairs", []) if isinstance(data, dict) else []
        for pair in pairs:
            conf = float(pair.get("confidence", 0.0))
            weighted = conf * float(row.get("alignment_confidence", 1.0))
            if weighted < min_pair_confidence:
                continue
            all_pairs.append(
                {
                    "source_term": pair.get("source_term", ""),
                    "target_term": pair.get("target_term", ""),
                    "model_pair_confidence": conf,
                    "weighted_confidence": weighted,
                    "source_sentence": pair.get("source_sentence", row.get("source_segment", "")),
                    "target_sentence": pair.get("target_sentence", row.get("target_segment", "")),
                }
            )

    all_pairs.sort(key=lambda x: x["weighted_confidence"], reverse=True)
    if top_k_pairs > 0:
        all_pairs = all_pairs[:top_k_pairs]

    with out_path.open("w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="TermAlign API mode")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--api-endpoint", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--top-k-pairs", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_termalign_api(
        bertalign_output=args.bertalign_output,
        output_file=args.output_file,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        min_pair_confidence=args.min_pair_confidence,
        top_k_pairs=args.top_k_pairs,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
