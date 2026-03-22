"""Shortest distance penalty for inconsistent term variants."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .data_model import DistanceInputs
from .normalization import normalize


def token_positions_by_variant(tokens: Sequence[str], variants: set[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        ntok = normalize(tok)
        if ntok in variants:
            out[ntok].append(idx)
    return out


def shortest_distance(positions_a: Sequence[int], positions_b: Sequence[int]) -> int:
    i = j = 0
    best = math.inf
    while i < len(positions_a) and j < len(positions_b):
        a, b = positions_a[i], positions_b[j]
        best = min(best, abs(a - b))
        if a < b:
            i += 1
        else:
            j += 1
    return int(best) if best is not math.inf else -1


def compute_shortest_distance_penalty(
    records: Iterable[Mapping[str, Any]], token_map: Mapping[str, DistanceInputs]
) -> float:
    penalties: List[float] = []

    for record in records:
        source_file = str(record.get("source_file", "__default__"))
        inputs = token_map.get(source_file)
        if inputs is None or not inputs.tokens:
            continue

        token_count = len(inputs.tokens)
        extracted_terms = record.get("extracted_terms", {})
        if not isinstance(extracted_terms, Mapping):
            continue

        for _src_term, variants in extracted_terms.items():
            if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
                variants = [str(variants)]
            norm_variants = {normalize(str(v)) for v in variants if normalize(str(v))}
            if len(norm_variants) < 2:
                penalties.append(0.0)
                continue

            pos_map = token_positions_by_variant(inputs.tokens, norm_variants)
            available = [v for v in norm_variants if pos_map.get(v)]
            if len(available) < 2:
                penalties.append(0.0)
                continue

            dmin = math.inf
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    d = shortest_distance(pos_map[available[i]], pos_map[available[j]])
                    if d >= 0:
                        dmin = min(dmin, d)

            if dmin is math.inf:
                penalties.append(0.0)
                continue

            penalty = 1.0 - (float(dmin) / float(token_count))
            penalty = max(0.0, min(1.0, penalty))
            penalties.append(penalty)

    if not penalties:
        return 0.0
    return sum(penalties) / len(penalties)
