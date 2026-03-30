from __future__ import annotations

from typing import Iterable, Mapping


def token_positions_by_variant(tokens: Iterable[str], variants: Iterable[str]) -> dict[str, list[int]]:
    var_set = {v for v in variants}
    out: dict[str, list[int]] = {v: [] for v in var_set}
    for idx, tok in enumerate(tokens):
        if tok in out:
            out[tok].append(idx)
    return out


def shortest_distance(pos_a: list[int], pos_b: list[int]) -> int:
    if not pos_a or not pos_b:
        return -1
    i = j = 0
    best = 10**9
    while i < len(pos_a) and j < len(pos_b):
        best = min(best, abs(pos_a[i] - pos_b[j]))
        if pos_a[i] < pos_b[j]:
            i += 1
        else:
            j += 1
    return best if best < 10**9 else -1
