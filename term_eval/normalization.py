"""Text normalization helpers."""

from __future__ import annotations

import re

NORM_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Normalize text for tolerant matching and counting."""
    value = text.strip().casefold()
    value = NORM_RE.sub(" ", value)
    return " ".join(value.split())
