from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline

DEFAULT_TERM_LIST_DIR = Path(__file__).resolve().parents[1] / "term_list"
DEFAULT_ZH_TERM_LIST = DEFAULT_TERM_LIST_DIR / "zh_terms.txt"
DEFAULT_EN_TERM_LIST = DEFAULT_TERM_LIST_DIR / "en_terms.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dictionary + BERT term extraction and alignment")
    parser.add_argument("--input", required=True, help="TSV file OR folder containing TSV files")
    parser.add_argument("--dict-zh", help="Chinese dictionary file (optional)")
    parser.add_argument("--dict-en", help="English dictionary file (optional)")
    parser.add_argument("--bert-model-zh", help="Fine-tuned Chinese BERT NER model path")
    parser.add_argument("--bert-model-en", help="Fine-tuned English BERT NER model path")
    parser.add_argument("--embed-model", default="multi-embedding", help="Embedding model for alignment")
    parser.add_argument("--similarity", type=float, default=0.5, help="High-confidence threshold")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT extraction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dict_zh = args.dict_zh if args.dict_zh else (str(DEFAULT_ZH_TERM_LIST) if DEFAULT_ZH_TERM_LIST.exists() else None)
    dict_en = args.dict_en if args.dict_en else (str(DEFAULT_EN_TERM_LIST) if DEFAULT_EN_TERM_LIST.exists() else None)
    run_pipeline(
        input_path=args.input,
        dict_zh_path=dict_zh,
        dict_en_path=dict_en,
        bert_model_zh=args.bert_model_zh,
        bert_model_en=args.bert_model_en,
        embed_model=args.embed_model,
        similarity_threshold=args.similarity,
        output_dir=args.output_dir,
        skip_bert=args.skip_bert,
    )


if __name__ == "__main__":
    main()
