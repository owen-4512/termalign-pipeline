"""Text normalization helpers powered by spaCy."""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language


@lru_cache(maxsize=1)
def _get_nlp() -> Language:
    """Load spaCy English pipeline for tokenization + lemmatization.

    Preferred model is ``en_core_web_sm``. We keep a safe spaCy-only fallback
    for environments where the model isn't installed.
    """
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        nlp = spacy.blank("en")
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        nlp.initialize()
        return nlp


def normalize(text: str) -> str:
    """Normalize text with spaCy tokenization + lemmatization.

    - lowercases text
    - removes punctuation and spaces
    - uses lemma where available, else lowercase token text
    """
    value = (text or "").strip()
    if not value:
        return ""

    doc = _get_nlp()(value)
    tokens: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower() if token.lemma_ else ""
        if not lemma or lemma == "-pron-":
            lemma = token.text.strip().lower()
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)
