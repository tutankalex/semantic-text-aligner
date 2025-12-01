#!/usr/bin/env python3
"""
DTW-based sequence alignment using litellm embeddings.

This script aligns two lists of strings (with optional None gaps) by building a
dynamic programming table that mixes cosine distance for matches with a fixed
gap penalty for skips. It can consume either a tuple of two lists or a list of
pre-aligned tuples, and it exposes a small CLI to run against the mixed-case
fixtures used in `tests/fixtures/case_mixed1.py`.
"""

from __future__ import annotations

import argparse
from math import sqrt
from typing import Iterable, Optional, Sequence, Tuple, Union

from litellm import embedding
from loguru import logger

AlignmentInput = Union[
    Tuple[list[Optional[str]], list[Optional[str]]],
    list[Tuple[Optional[str], Optional[str]]],
]


def _cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine distance (1 - cosine similarity) between two vectors."""
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = sqrt(sum(x * x for x in vec_a))
    norm_b = sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = max(min(dot_product / (norm_a * norm_b), 1.0), -1.0)
    return 1.0 - similarity


def _embed_texts(
    texts: Iterable[Optional[str]], model: str
) -> dict[str, Sequence[float]]:
    """Embed unique non-empty strings with litellm and return a lookup map."""
    seen: dict[str, int] = {}
    ordered: list[str] = []
    for item in texts:
        if item is None:
            continue
        if item not in seen:
            seen[item] = len(ordered)
            ordered.append(item)

    if not ordered:
        return {}

    logger.info(f"Requesting embeddings for {len(ordered)} unique items via {model}")
    response = embedding(model=model, input=ordered)
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):  # type: ignore[unreachable]
        data = response.get("data")
    if data is None:
        raise ValueError("litellm embedding response missing 'data' field")
    vectors: list[Sequence[float]] = []
    for record in data:
        # litellm responses may expose attributes or dict-style access
        vec = (
            record.embedding
            if hasattr(record, "embedding")
            else record.get("embedding")
        )
        vectors.append(vec)

    return {text: vectors[idx] for idx, text in enumerate(ordered)}


def _normalize_input(
    input_data: AlignmentInput,
) -> tuple[list[Optional[str]], list[Optional[str]]]:
    """Accept tuple-of-lists or list-of-tuples and return left/right lists."""
    if isinstance(input_data, tuple) and len(input_data) == 2:
        left, right = input_data
        return list(left), list(right)
    if isinstance(input_data, list):
        left: list[Optional[str]] = []
        right: list[Optional[str]] = []
        for pair in input_data:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("List input must contain tuples of length 2")
            left.append(pair[0])
            right.append(pair[1])
        return left, right
    raise ValueError("Input must be either (list, list) or list of tuples")


def _pair_cost(
    left_item: Optional[str],
    right_item: Optional[str],
    embeddings: dict[str, Sequence[float]],
    gap_penalty: float,
) -> float:
    """Cost of aligning two tokens, falling back to gap penalty when missing."""
    if left_item is None and right_item is None:
        return 0.0
    if left_item is None or right_item is None:
        return gap_penalty
    vec_left = embeddings.get(left_item)
    vec_right = embeddings.get(right_item)
    if vec_left is None or vec_right is None:
        return gap_penalty
    return _cosine_distance(vec_left, vec_right)


def align_sequences(
    input_data: AlignmentInput,
    gap_penalty: float = 0.1,
    model: str = "ollama/nomic-embed-text",
) -> list[tuple[Optional[str], Optional[str]]]:
    """Run DTW alignment and return a list of (left, right) tuples."""
    left, right = _normalize_input(input_data)
    embeddings = _embed_texts([*left, *right], model=model)

    m, n = len(left), len(right)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back: list[list[Optional[str]]] = [[None] * (n + 1) for _ in range(m + 1)]
    move: Optional[str] = None

    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
        back[i][0] = "U"
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
        back[0][j] = "L"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_diag = dp[i - 1][j - 1] + _pair_cost(
                left[i - 1], right[j - 1], embeddings, gap_penalty
            )
            cost_up = dp[i - 1][j] + gap_penalty  # gap in right
            cost_left = dp[i][j - 1] + gap_penalty  # gap in left

            best_cost = cost_diag
            move = "D"
            if cost_up < best_cost:
                best_cost = cost_up
                move = "U"
            if cost_left < best_cost:
                best_cost = cost_left
                move = "L"

            dp[i][j] = best_cost
            back[i][j] = move

    aligned: list[tuple[Optional[str], Optional[str]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "D":
            aligned.append((left[i - 1], right[j - 1]))
            i -= 1
            j -= 1
        elif move == "U":
            aligned.append((left[i - 1], None))
            i -= 1
        elif move == "L":
            aligned.append((None, right[j - 1]))
            j -= 1
        else:
            raise RuntimeError(f"Backpointer missing at ({i}, {j})")

    aligned.reverse()
    logger.info(f"DTW alignment complete with total cost {dp[m][n]:.4f}")
    return aligned


def _print_alignment(rows: list[tuple[Optional[str], Optional[str]]]) -> None:
    """Render an alignment table to stdout for quick inspection."""
    max_left = max((len(item or "") for item, _ in rows), default=0)
    for idx, (left, right) in enumerate(rows, start=1):
        left_cell = (left or "").ljust(max_left)
        right_cell = right or ""
        print(f"{idx:3d}. {left_cell} | {right_cell}")


def _print_raw_lists(left: list[Optional[str]], right: list[Optional[str]]) -> None:
    """Print the unaligned input lists side by side."""
    max_left = max((len(item or "") for item in left), default=0)
    print("=== Raw Inputs (unaligned) ===")
    for idx in range(max(len(left), len(right))):
        left_cell = left[idx] if idx < len(left) else None
        right_cell = right[idx] if idx < len(right) else None
        print(f"{idx:3d}. {(left_cell or '').ljust(max_left)} | {right_cell or ''}")
    print("=== End Raw Inputs ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DTW aligner using litellm embeddings (ollama/nomic-embed-text)."
    )
    parser.add_argument(
        "--demo-case-index",
        type=int,
        default=0,
        help="Index into SUBCASES from tests/fixtures/case_mixed1.py (default: 0)",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.8,
        help="Cost for inserting a gap (default: 0.8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/nomic-embed-text",
        help="Embedding model identifier for litellm",
    )
    args = parser.parse_args()

    try:
        from tests.fixtures.case_mixed1 import SUBCASES
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unable to import demo cases: {exc}") from exc

    if not (0 <= args.demo_case_index < len(SUBCASES)):
        raise SystemExit(f"--demo-case-index must be between 0 and {len(SUBCASES) - 1}")

    case = SUBCASES[args.demo_case_index]
    logger.info(f"Running demo case {args.demo_case_index}: {case.name}")
    _print_raw_lists(case.input.left, case.input.right)
    rows = align_sequences(
        (case.input.left, case.input.right),
        gap_penalty=args.gap_penalty,
        model=args.model,
    )
    _print_alignment(rows)


if __name__ == "__main__":
    main()
