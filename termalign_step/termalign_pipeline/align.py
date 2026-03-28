from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .extractors import TermOccurrence


@dataclass
class AlignmentResult:
    zh_term: TermOccurrence
    en_term: TermOccurrence
    similarity: float


class Embedder:
    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.append(pooled[0].cpu().numpy())
        return np.vstack(vectors)


def align_terms(
    zh_terms: List[TermOccurrence],
    en_terms: List[TermOccurrence],
    embedder: Embedder,
) -> List[AlignmentResult]:
    if not zh_terms or not en_terms:
        return []

    zh_embeddings = embedder.encode([term.term for term in zh_terms])
    en_embeddings = embedder.encode([term.term for term in en_terms])

    zh_norm = zh_embeddings / (np.linalg.norm(zh_embeddings, axis=1, keepdims=True) + 1e-8)
    en_norm = en_embeddings / (np.linalg.norm(en_embeddings, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = zh_norm @ en_norm.T

    results: List[AlignmentResult] = []
    for idx, zh_term in enumerate(zh_terms):
        best_idx = int(similarity_matrix[idx].argmax())
        similarity = float(similarity_matrix[idx, best_idx])
        results.append(AlignmentResult(zh_term=zh_term, en_term=en_terms[best_idx], similarity=similarity))
    return results
