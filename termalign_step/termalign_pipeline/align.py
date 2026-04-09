from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._use_sentence_transformer = False
        self._use_hash_fallback = False

        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None

        if self._try_load_transformers(model_name_or_path):
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._use_sentence_transformer = True
            self.sentence_transformer = SentenceTransformer(model_name_or_path, device=self.device)
            return
        except Exception:  # noqa: BLE001
            # Keep trying below with sentence-transformers-style local subpaths.
            self._use_sentence_transformer = False
            self.sentence_transformer = None

        for candidate in self._transformer_candidates(model_name_or_path):
            if self._try_load_transformers(str(candidate)):
                return

        # Final fallback: deterministic lightweight hashing embeddings so the
        # pipeline can continue even when local model/runtime versions are
        # incompatible (e.g. sentence-transformers 3.x vs model exported by 5.x).
        print(
            "⚠️ Failed to load embedding model with all strategies; "
            "falling back to lexical hash embeddings. "
            "For best quality, upgrade sentence-transformers/transformers."
        )
        self._use_hash_fallback = True

    def _try_load_transformers(self, model_name_or_path: str) -> bool:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=False,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            self.model.eval()
            self.model.to(self.device)
            self._use_sentence_transformer = False
            return True
        except Exception:  # noqa: BLE001
            self.tokenizer = None
            self.model = None
            return False

    @staticmethod
    def _transformer_candidates(model_name_or_path: str) -> list[Path]:
        root = Path(model_name_or_path)
        if not root.exists() or not root.is_dir():
            return []
        candidates: list[Path] = []
        direct = root / "0_Transformer"
        if direct.exists():
            candidates.append(direct)
        modules_json = root / "modules.json"
        if modules_json.exists():
            try:
                modules = json.loads(modules_json.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                modules = []
            if isinstance(modules, list):
                for item in modules:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type", ""))
                    item_path = item.get("path")
                    if "Transformer" not in item_type or not item_path:
                        continue
                    candidate = root / str(item_path)
                    if candidate.exists():
                        candidates.append(candidate)
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self._use_sentence_transformer and self.sentence_transformer is not None:
            vectors = self.sentence_transformer.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors)
        if self._use_hash_fallback:
            return self._encode_hash(texts)

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

    @staticmethod
    def _encode_hash(texts: Iterable[str], dim: int = 256) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for token in str(text).lower().split():
                idx = hash(token) % dim
                vec[idx] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        if not vectors:
            return np.zeros((0, dim), dtype=np.float32)
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
