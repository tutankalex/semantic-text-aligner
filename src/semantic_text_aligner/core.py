#!/usr/bin/env python3
"""
Public entry point for aligning two string lists, with optional chunking + stitching.
"""

from __future__ import annotations

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
