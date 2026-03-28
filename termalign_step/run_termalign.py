#!/usr/bin/env python3
"""Term extraction + term alignment with model/api options."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import pipeline


DEFAULT_ALIGNER_MODEL = "owen4512/minilm-finance-term-aligner"
DEFAULT_ZH_EXTRACTOR = "owen4512/bert-base-chinese-finance-term-extractor"
DEFAULT_EN_EXTRACTOR = "owen4512/bert-base-cased-finance-term-extractor"


@dataclass
class Term:
    text: str
    confidence: float


class HFExtractor:
    def __init__(self, zh_model: str, en_model: str, device: int = -1) -> None:
        self.zh_pipe = pipeline("token-classification", model=zh_model, aggregation_strategy="simple", device=device)
        self.en_pipe = pipeline("token-classification", model=en_model, aggregation_strategy="simple", device=device)

    def extract(self, text: str, lang: str, min_conf: float) -> list[Term]:
        pred = self.zh_pipe(text) if lang == "zh" else self.en_pipe(text)
        terms: list[Term] = []
        for item in pred:
            score = float(item.get("score", 0.0))
            if score >= min_conf:
                terms.append(Term(text=item.get("word", "").strip(), confidence=score))
        return [t for t in terms if t.text]


class APIExtractor:
    def __init__(self, endpoint: str, api_key: str | None = None, timeout: int = 60) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def extract(self, text: str, lang: str, min_conf: float) -> list[Term]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"text": text, "lang": lang, "min_confidence": min_conf}
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        items = response.json().get("terms", [])
        return [Term(text=i["text"], confidence=float(i["confidence"])) for i in items if i.get("text")]


class PairScorer:
    def __init__(self, aligner_model: str, device: int = -1) -> None:
        self.pipe = pipeline("text-classification", model=aligner_model, device=device)

    def score(self, source_term: str, target_term: str) -> float:
        out = self.pipe(f"{source_term} [SEP] {target_term}")[0]
        score = float(out.get("score", 0.0))
        label = str(out.get("label", "")).upper()
        return score if "POS" in label or "1" in label else (1 - score)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_termalign(
    bertalign_output: str,
    output_file: str,
    extraction_mode: str = "model",
    min_term_confidence: float = 0.5,
    min_pair_confidence: float = 0.5,
    source_lang: str = "en",
    target_lang: str = "zh",
    top_k_pairs: int = 5,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    zh_extractor_model: str = DEFAULT_ZH_EXTRACTOR,
    en_extractor_model: str = DEFAULT_EN_EXTRACTOR,
    api_endpoint: str | None = None,
    api_key: str | None = None,
    device: int = -1,
) -> str:
    rows = _load_jsonl(Path(bertalign_output))
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if extraction_mode == "api":
        if not api_endpoint:
            raise ValueError("api_endpoint is required when extraction_mode='api'")
        extractor = APIExtractor(endpoint=api_endpoint, api_key=api_key)
    else:
        extractor = HFExtractor(zh_model=zh_extractor_model, en_model=en_extractor_model, device=device)

    scorer = PairScorer(aligner_model=aligner_model, device=device)
    results: list[dict[str, Any]] = []

    for row in rows:
        src = row["source_segment"]
        tgt = row["target_segment"]
        align_conf = float(row.get("alignment_confidence", 1.0))

        src_terms = extractor.extract(src, source_lang, min_term_confidence)
        tgt_terms = extractor.extract(tgt, target_lang, min_term_confidence)

        pair_scores: list[dict[str, Any]] = []
        for s_term, t_term in itertools.product(src_terms, tgt_terms):
            model_conf = scorer.score(s_term.text, t_term.text)
            total_conf = align_conf * s_term.confidence * t_term.confidence * model_conf
            if total_conf >= min_pair_confidence:
                pair_scores.append(
                    {
                        "source_term": s_term.text,
                        "target_term": t_term.text,
                        "alignment_confidence": align_conf,
                        "source_term_confidence": s_term.confidence,
                        "target_term_confidence": t_term.confidence,
                        "model_pair_confidence": model_conf,
                        "weighted_confidence": total_conf,
                        "source_segment": src,
                        "target_segment": tgt,
                    }
                )

        pair_scores.sort(key=lambda x: x["weighted_confidence"], reverse=True)
        results.extend(pair_scores[:top_k_pairs])

    with out_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return str(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TermAlign step")
    parser.add_argument("--bertalign-output", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--extraction-mode", choices=["model", "api"], default="model")
    parser.add_argument("--min-term-confidence", type=float, default=0.5)
    parser.add_argument("--min-pair-confidence", type=float, default=0.5)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--top-k-pairs", type=int, default=5)
    parser.add_argument("--aligner-model", default=DEFAULT_ALIGNER_MODEL)
    parser.add_argument("--zh-extractor-model", default=DEFAULT_ZH_EXTRACTOR)
    parser.add_argument("--en-extractor-model", default=DEFAULT_EN_EXTRACTOR)
    parser.add_argument("--api-endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_termalign(
        bertalign_output=args.bertalign_output,
        output_file=args.output_file,
        extraction_mode=args.extraction_mode,
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
    )


if __name__ == "__main__":
    main()
