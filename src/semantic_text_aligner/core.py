#!/usr/bin/env python3
"""
Public entry point for aligning two string lists, with optional chunking + stitching.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from . import aligner
from .stitcher import stitch_all_chunks

AlignmentInput = Union[
    Tuple[list[Optional[str]], list[Optional[str]]],
    list[Tuple[Optional[str], Optional[str]]],
]


def _chunk_pairs(
    left: list[Optional[str]],
    right: list[Optional[str]],
    chunk_size: int,
    overlap_size: int,
) -> Iterable[tuple[list[Optional[str]], list[Optional[str]]]]:
    """Yield paired slices of left/right lists with the requested overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative")

    step = chunk_size - overlap_size
    if step <= 0:
        step = 1

    max_len = max(len(left), len(right))
    start = 0
    while start < max_len:
        end = start + chunk_size
        yield left[start:end], right[start:end]
        if end >= max_len:
            break
        start += step


def align_string_lists(
    input_data: AlignmentInput,
    chunk_size: Optional[int] = None,
    overlap_size: Optional[int] = None,
    gap_penalty: float = 0.1,
    model: str = "ollama/nomic-embed-text",
) -> list[tuple[Optional[str], Optional[str]]]:
    """
    Align two lists of strings, optionally in overlapping chunks for memory efficiency.

    - If chunk_size is None, a single full alignment is run.
    - If overlap_size is None (and chunking is enabled), overlap defaults to min(4, chunk_size // 2).
    """
    left, right = aligner._normalize_input(input_data)

    if chunk_size is None:
        return aligner.align_sequences((left, right), gap_penalty=gap_penalty, model=model)

    if overlap_size is None:
        overlap_size = min(4, chunk_size // 2)

    chunks: list[list[tuple[Optional[str], Optional[str]]]] = []
    for left_slice, right_slice in _chunk_pairs(left, right, chunk_size, overlap_size):
        aligned_chunk = aligner.align_sequences(
            (left_slice, right_slice), gap_penalty=gap_penalty, model=model
        )
        chunks.append(aligned_chunk)

    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks[0]

    return stitch_all_chunks(chunks, overlap_size=overlap_size)


def _load_lines(path: str) -> list[str]:
    """Load newline-delimited strings from a file, trimming outer whitespace."""
    return Path(path).read_text(encoding="utf-8").strip().splitlines()


def _print_alignment(rows: list[tuple[Optional[str], Optional[str]]]) -> None:
    """Render an alignment table to stdout for quick inspection."""
    max_left = max((len(item or "") for item, _ in rows), default=0)
    for idx, (left, right) in enumerate(rows, start=1):
        left_cell = (left or "").ljust(max_left)
        right_cell = right or ""
        print(f"{idx:3d}. {left_cell} | {right_cell}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align two newline-delimited files, optionally chunked and stitched."
    )
    parser.add_argument("file_left", type=str, help="Path to left-side lines")
    parser.add_argument("file_right", type=str, help="Path to right-side lines")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for alignment (default: no chunking)",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=None,
        help="Overlap size for stitching (default: min(4, chunk_size//2) when chunking)",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.1,
        help="Gap penalty for DTW alignment (default: 0.1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/nomic-embed-text",
        help="Embedding model identifier for litellm",
    )
    args = parser.parse_args()

    left_lines = _load_lines(args.file_left)
    right_lines = _load_lines(args.file_right)

    rows = align_string_lists(
        (left_lines, right_lines),
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        gap_penalty=args.gap_penalty,
        model=args.model,
    )
    _print_alignment(rows)


if __name__ == "__main__":
    main()
