from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class TermOccurrence:
    term: str
    source: str
    confidence: float | None
    sentence: str
    start: int
    end: int


class DictionaryExtractor:
    def __init__(self, terms: Sequence[str]) -> None:
        self.terms = [term for term in terms if term]

    def extract(self, sentence: str, source_label: str = "dict") -> List[TermOccurrence]:
        occurrences: List[TermOccurrence] = []
        for term in self.terms:
            for match in re.finditer(re.escape(term), sentence):
                occurrences.append(
                    TermOccurrence(
                        term=term,
                        source=source_label,
                        confidence=None,
                        sentence=sentence,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return occurrences


class BertTermExtractor:
    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if not self.tokenizer.is_fast:
            raise ValueError("BERT term extraction requires a fast tokenizer to access offsets.")
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def extract(self, sentence: str, source_label: str = "bert") -> List[TermOccurrence]:
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        labels = probs.argmax(dim=-1).tolist()
        id2label = self.model.config.id2label

        results: List[TermOccurrence] = []
        current_scores: List[float] = []
        current_start: int | None = None
        current_end: int | None = None

        for idx, (label_id, offset) in enumerate(zip(labels, offsets)):
            start, end = offset
            if start == end:
                if current_scores:
                    results.append(self._flush(sentence, current_scores, current_start, current_end, source_label))
                    current_scores = []
                    current_start = None
                    current_end = None
                continue

            label = id2label[label_id]
            score = probs[idx][label_id].item()
            is_start = label.startswith("B-")
            is_inside = label.startswith("I-")

            if is_start or (is_inside and not current_scores):
                if current_scores:
                    results.append(self._flush(sentence, current_scores, current_start, current_end, source_label))
                current_scores = [score]
                current_start = start
                current_end = end
            elif is_inside and current_scores:
                current_scores.append(score)
                current_end = end
            else:
                if current_scores:
                    results.append(self._flush(sentence, current_scores, current_start, current_end, source_label))
                    current_scores = []
                    current_start = None
                    current_end = None

        if current_scores:
            results.append(self._flush(sentence, current_scores, current_start, current_end, source_label))

        return [result for result in results if result.term]

    def _flush(
        self,
        sentence: str,
        scores: Sequence[float],
        start_index: int | None,
        end_index: int | None,
        source_label: str,
    ) -> TermOccurrence:
        if start_index is None or end_index is None:
            term_text = ""
            start_index = 0
            end_index = 0
        else:
            term_text = sentence[start_index:end_index]
        confidence = float(sum(scores) / max(len(scores), 1))
        return TermOccurrence(
            term=term_text,
            source=source_label,
            confidence=confidence,
            sentence=sentence,
            start=start_index,
            end=end_index,
        )
