from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def extract(self, sentence: str, source_label: str = "bert") -> List[TermOccurrence]:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        token_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        labels = probs.argmax(dim=-1).tolist()

        id2label = self.model.config.id2label
        results: List[TermOccurrence] = []

        current_tokens: List[str] = []
        current_scores: List[float] = []
        current_start: int | None = None

        for idx, (token, label_id) in enumerate(zip(tokens, labels)):
            label = id2label[label_id]
            score = probs[idx][label_id].item()
            is_start = label.startswith("B-")
            is_inside = label.startswith("I-")

            if is_start:
                if current_tokens:
                    results.append(self._flush(sentence, current_tokens, current_scores, current_start))
                current_tokens = [token]
                current_scores = [score]
                current_start = idx
            elif is_inside and current_tokens:
                current_tokens.append(token)
                current_scores.append(score)
            else:
                if current_tokens:
                    results.append(self._flush(sentence, current_tokens, current_scores, current_start))
                    current_tokens = []
                    current_scores = []
                    current_start = None

        if current_tokens:
            results.append(self._flush(sentence, current_tokens, current_scores, current_start))

        return [result for result in results if result.term]

    def _flush(
        self,
        sentence: str,
        tokens: Sequence[str],
        scores: Sequence[float],
        start_index: int | None,
    ) -> TermOccurrence:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        if " " not in sentence:
            text = text.replace(" ", "")
        confidence = float(sum(scores) / max(len(scores), 1))
        start_char, end_char = self._locate_span(sentence, text)
        return TermOccurrence(
            term=text,
            source="bert",
            confidence=confidence,
            sentence=sentence,
            start=start_char,
            end=end_char,
        )

    @staticmethod
    def _locate_span(sentence: str, term: str) -> Tuple[int, int]:
        match = re.search(re.escape(term), sentence)
        if match:
            return match.start(), match.end()
        return 0, len(term)
