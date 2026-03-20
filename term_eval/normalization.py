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
            try:
                nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            except Exception:
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
            lemma = _normalize_inflection(lemma)
            tokens.append(lemma)
    return " ".join(tokens)


def _normalize_inflection(token: str) -> str:
    """General inflection normalization for cases spaCy may miss in fallback mode.

    This keeps the rule generic (plural/participle/past-tense handling) and is
    applied equally to predicted and gold terms through the shared `normalize()`
    path.
    """
    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith("v"):
            return base + "e"
        return base
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 4:
        base = token[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith("iz"):
            return base + "e"
        return base
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token
