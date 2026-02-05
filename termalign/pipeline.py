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


def _normalize_en_sentence(sentence: str) -> str:
    return " ".join(sentence.split())


def _keep_longest_non_overlapping(terms: List[TermOccurrence]) -> List[TermOccurrence]:
    by_sentence: dict[str, List[TermOccurrence]] = {}
    for term in terms:
        by_sentence.setdefault(term.sentence, []).append(term)

    results: List[TermOccurrence] = []
    for sentence_terms in by_sentence.values():
        sorted_terms = sorted(
            sentence_terms,
            key=lambda item: (-(item.end - item.start), item.start),
        )
        kept: List[TermOccurrence] = []
        for candidate in sorted_terms:
            overlap = False
            for existing in kept:
                if candidate.start < existing.end and candidate.end > existing.start:
                    overlap = True
                    break
            if not overlap:
                kept.append(candidate)
        results.extend(sorted(kept, key=lambda item: item.start))
    return results


def _filter_en_terms(terms: List[TermOccurrence]) -> List[TermOccurrence]:
    filtered = [term for term in terms if len(term.term.strip()) > 2]
    return _keep_longest_non_overlapping(filtered)


def _filter_zh_terms(terms: List[TermOccurrence]) -> List[TermOccurrence]:
    filtered = [term for term in terms if len(term.term.strip()) > 1]
    return _keep_longest_non_overlapping(filtered)


def extract_terms(
    pairs: Sequence[SentencePair],
    dict_terms: Sequence[str] | None,
    bert_model: str | None,
    language_label: str,
    skip_bert: bool,
) -> List[TermOccurrence]:
    dict_extractor = None
    if dict_terms:
        if language_label == "en":
            dict_extractor = DictionaryExtractor(dict_terms, whole_word=True, case_sensitive=False)
        else:
            dict_extractor = DictionaryExtractor(dict_terms)
    bert_extractor = None
    if bert_model and not skip_bert:
        bert_extractor = BertTermExtractor(bert_model)

    all_occurrences: List[TermOccurrence] = []
    for pair in tqdm(pairs, desc=f"extract-{language_label}"):
        sentence = pair.zh if language_label == "zh" else pair.en
        dict_occurrences: List[TermOccurrence] = []
        if dict_extractor:
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
    embedder: Embedder,
) -> List[AlignmentResult]:
    return align_terms(zh_terms, en_terms, embedder)


def _group_terms_by_sentence(terms: List[TermOccurrence]) -> dict[str, List[TermOccurrence]]:
    grouped: dict[str, List[TermOccurrence]] = {}
    for term in terms:
        grouped.setdefault(term.sentence, []).append(term)
    return grouped


def _alignment_rows(alignments: List[AlignmentResult], converter_s2t: OpenCC) -> List[dict]:
    return [
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
    ]


def run_pipeline(
    input_path: str | Path,
    dict_zh_path: str | Path | None,
    dict_en_path: str | Path | None,
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
    dict_zh = None
    if dict_zh_path:
        dict_zh = [converter_t2s.convert(term) for term in read_dictionary(dict_zh_path)]
    dict_en = read_dictionary(dict_en_path) if dict_en_path else None

    normalized_pairs = [
        SentencePair(zh=converter_t2s.convert(pair.zh), en=pair.en) for pair in pairs
    ]
    en_pairs = [SentencePair(zh=pair.zh, en=_normalize_en_sentence(pair.en)) for pair in pairs]

    zh_terms = extract_terms(normalized_pairs, dict_zh, bert_model_zh, "zh", skip_bert)
    en_terms = extract_terms(en_pairs, dict_en, bert_model_en, "en", skip_bert)
    zh_terms = _filter_zh_terms(zh_terms)
    en_terms = _filter_en_terms(en_terms)

    zh_terms_by_sentence = _group_terms_by_sentence(zh_terms)
    en_terms_by_sentence = _group_terms_by_sentence(en_terms)

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

    embedder = Embedder(embed_model)
    alignments: List[AlignmentResult] = []
    for pair in zip(normalized_pairs, en_pairs):
        zh_sentence = pair[0].zh
        en_sentence = pair[1].en
        zh_group = zh_terms_by_sentence.get(zh_sentence, [])
        en_group = en_terms_by_sentence.get(en_sentence, [])
        if not zh_group or not en_group:
            continue
        alignments.extend(build_alignment(zh_group, en_group, embedder))

    rows = _alignment_rows(alignments, converter_s2t)
    write_tsv(output_path / "alignments.tsv", rows)

    filtered_rows = [row for row in rows if float(row["similarity"]) > 0.5]
    write_tsv(output_path / "alignments_high_conf.tsv", filtered_rows)
