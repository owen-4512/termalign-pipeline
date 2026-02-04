from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm

from opencc import OpenCC

from .align import AlignmentResult, Embedder, align_terms
from .extractors import BertTermExtractor, DictionaryExtractor, TermOccurrence
from .io_utils import SentencePair, read_dictionary, read_sentence_pairs, write_tsv


def _dedupe_dict_spans(occurrences: Sequence[TermOccurrence]) -> set[tuple[int, int, str]]:
    return {(occ.start, occ.end, occ.sentence) for occ in occurrences}


def extract_terms(
    pairs: Sequence[SentencePair],
    dict_terms: Sequence[str],
    bert_model: str | None,
    language_label: str,
    skip_bert: bool,
) -> List[TermOccurrence]:
    dict_extractor = DictionaryExtractor(dict_terms)
    bert_extractor = None
    if bert_model and not skip_bert:
        bert_extractor = BertTermExtractor(bert_model)

    all_occurrences: List[TermOccurrence] = []
    for pair in tqdm(pairs, desc=f"extract-{language_label}"):
        sentence = pair.zh if language_label == "zh" else pair.en
        dict_occurrences = dict_extractor.extract(sentence)
        all_occurrences.extend(dict_occurrences)

        dict_spans = _dedupe_dict_spans(dict_occurrences)
        if bert_extractor:
            bert_occurrences = bert_extractor.extract(sentence)
            for occ in bert_occurrences:
                if (occ.start, occ.end, occ.sentence) in dict_spans:
                    continue
                all_occurrences.append(occ)
    return all_occurrences


def build_alignment(
    zh_terms: List[TermOccurrence],
    en_terms: List[TermOccurrence],
    embed_model: str,
) -> List[AlignmentResult]:
    embedder = Embedder(embed_model)
    return align_terms(zh_terms, en_terms, embedder)


def run_pipeline(
    input_path: str | Path,
    dict_zh_path: str | Path,
    dict_en_path: str | Path,
    bert_model_zh: str | None,
    bert_model_en: str | None,
    embed_model: str,
    output_dir: str | Path,
    skip_bert: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converter_t2s = OpenCC("t2s")
    converter_s2t = OpenCC("s2t")

    pairs = read_sentence_pairs(input_path)
    dict_zh = [converter_t2s.convert(term) for term in read_dictionary(dict_zh_path)]
    dict_en = read_dictionary(dict_en_path)

    normalized_pairs = [
        SentencePair(zh=converter_t2s.convert(pair.zh), en=pair.en) for pair in pairs
    ]
    zh_terms = extract_terms(normalized_pairs, dict_zh, bert_model_zh, "zh", skip_bert)
    en_terms = extract_terms(pairs, dict_en, bert_model_en, "en", skip_bert)

    write_tsv(
        output_path / "terms_zh.tsv",
        [
            {
                "term": converter_s2t.convert(term.term),
                "source": term.source,
                "confidence": term.confidence,
                "sentence": converter_s2t.convert(term.sentence),
            }
            for term in zh_terms
        ],
    )
    write_tsv(
        output_path / "terms_en.tsv",
        [
            {
                "term": term.term,
                "source": term.source,
                "confidence": term.confidence,
                "sentence": term.sentence,
            }
            for term in en_terms
        ],
    )

    alignments = build_alignment(zh_terms, en_terms, embed_model)
    write_tsv(
        output_path / "alignments.tsv",
        [
            {
                "zh_term": converter_s2t.convert(alignment.zh_term.term),
                "en_term": alignment.en_term.term,
                "similarity": alignment.similarity,
                "zh_source": alignment.zh_term.source,
                "en_source": alignment.en_term.source,
                "zh_confidence": alignment.zh_term.confidence,
                "en_confidence": alignment.en_term.confidence,
                "zh_sentence": converter_s2t.convert(alignment.zh_term.sentence),
                "en_sentence": alignment.en_term.sentence,
            }
            for alignment in alignments
        ],
    )
