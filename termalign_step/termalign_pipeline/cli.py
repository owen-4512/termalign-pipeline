from __future__ import annotations

import argparse

from .pipeline import run_pipeline


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
    run_pipeline(
        input_path=args.input,
        dict_zh_path=args.dict_zh,
        dict_en_path=args.dict_en,
        bert_model_zh=args.bert_model_zh,
        bert_model_en=args.bert_model_en,
        embed_model=args.embed_model,
        similarity_threshold=args.similarity,
        output_dir=args.output_dir,
        skip_bert=args.skip_bert,
    )


if __name__ == "__main__":
    main()
