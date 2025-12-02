from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple

from loguru import logger

AlignedRow = Tuple[Optional[str], Optional[str]]


def _compatible_and_merge(a: AlignedRow, b: AlignedRow) -> Optional[AlignedRow]:
    """Return merged row if rows are compatible, otherwise None."""
    left_a, right_a = a
    left_b, right_b = b
    left_conflict = left_a is not None and left_b is not None and left_a != left_b
    right_conflict = right_a is not None and right_b is not None and right_a != right_b
    if left_conflict or right_conflict:
        return None

    shared_token = (
        (left_a is not None and left_a == left_b)
        or (right_a is not None and right_a == right_b)
    )
    if not shared_token:
        return None

    merged_left = left_a if left_a is not None else left_b
    merged_right = right_a if right_a is not None else right_b
    return (merged_left, merged_right)


def _find_canonical_anchor(rows: list[AlignedRow]) -> Optional[int]:
    """Return the index of the first row where both elements are non-None."""
    for idx, (left, right) in enumerate(rows):
        if left is not None and right is not None:
            return idx
    return None


def _matching_overlap_length(
    tail_window: list[AlignedRow],
    new_chunk: list[AlignedRow],
    tail_idx: int,
    head_idx: int,
) -> tuple[int, list[AlignedRow]]:
    """How many consecutive rows are compatible starting at the anchor positions.

    Compatibility allows upgrading gap rows when the other chunk supplies the
    missing side, as long as there are no conflicts.
    """

    merged_rows: list[AlignedRow] = []
    match_len = 0
    while (
        tail_idx + match_len < len(tail_window)
        and head_idx + match_len < len(new_chunk)
    ):
        merged = _compatible_and_merge(
            tail_window[tail_idx + match_len], new_chunk[head_idx + match_len]
        )
        if merged is None:
            break
        merged_rows.append(merged)
        match_len += 1

    return match_len, merged_rows


def _trailing_permutable_block_len(window: list[AlignedRow]) -> int:
    """Length of the trailing permutable (gap-only) block."""
    length = 0
    for left, right in reversed(window):
        if (left is None) != (right is None):
            length += 1
        else:
            break
    return length


def _stitch_core(
    prev_tail: list[AlignedRow],
    new_chunk: list[AlignedRow],
    overlap_size: int,
) -> tuple[list[AlignedRow], int, bool, bool]:
    """Shared core used by both public stitchers.

    Returns:
    - stitched_rows: the combined overlap suffix plus any new rows.
    - trim_from_tail: how many rows should be removed from the accumulator tail
      before appending stitched_rows.
    - anchor_found: True when the canonical anchor exists in both windows.
    - anchor_present_in_head: True when the canonical anchor exists in head_window.
    """
    W_max = 2 * overlap_size
    tail_window = prev_tail[-W_max:] if W_max > 0 else []
    head_window = new_chunk[:W_max] if W_max > 0 else []

    anchor_idx_head = _find_canonical_anchor(head_window)
    if anchor_idx_head is None:
        return list(new_chunk), 0, False, False

    anchor_row = head_window[anchor_idx_head]

    def _find_anchor_in_tail() -> Optional[int]:
        try:
            return tail_window.index(anchor_row)
        except ValueError:
            for idx, row in enumerate(tail_window):
                if _compatible_and_merge(row, anchor_row) == anchor_row:
                    return idx
            return None

    anchor_idx_tail = _find_anchor_in_tail()
    if anchor_idx_tail is None:
        return list(new_chunk), 0, False, True

    tail_prefix = tail_window[:anchor_idx_tail]
    head_prefix = head_window[:anchor_idx_head]

    updated_tail_window = list(tail_window)
    consumed_head = [False] * len(head_prefix)
    earliest_rewrite_idx: Optional[int] = None

    for h_idx, head_row in enumerate(head_prefix):
        for t_idx in range(anchor_idx_tail, -1, -1):
            merged = _compatible_and_merge(updated_tail_window[t_idx], head_row)
            if merged is None:
                continue
            consumed_head[h_idx] = True
            if merged != updated_tail_window[t_idx]:
                updated_tail_window[t_idx] = merged
                earliest_rewrite_idx = (
                    t_idx
                    if earliest_rewrite_idx is None
                    else min(earliest_rewrite_idx, t_idx)
                )
            break

    leftover_head_rows = [
        row for used, row in zip(consumed_head, head_prefix) if not used
    ]

    rewrite_idx = (
        earliest_rewrite_idx if earliest_rewrite_idx is not None else anchor_idx_tail
    )
    start_in_prev = len(prev_tail) - len(tail_window) + rewrite_idx
    overlap_len, merged_overlap = _matching_overlap_length(
        updated_tail_window, new_chunk, anchor_idx_tail, anchor_idx_head
    )

    trim_from_tail = len(prev_tail) - start_in_prev
    rewritten_prefix = updated_tail_window[rewrite_idx:anchor_idx_tail]
    stitched_rows = (
        leftover_head_rows
        + rewritten_prefix
        + merged_overlap
        + list(new_chunk[anchor_idx_head + overlap_len :])
    )
    return stitched_rows, trim_from_tail, True, True


def stitch_two_chunks(
    prev_tail: list[AlignedRow],
    new_chunk: list[AlignedRow],
    overlap_size: int,
) -> list[AlignedRow]:
    """
    Stitch a new chunk onto a tail window, returning the rows to append.

    The returned rows reflect the canonical anchor alignment and any overlap
    that can safely be trimmed away.
    """
    stitched_rows, _, _, _ = _stitch_core(prev_tail, new_chunk, overlap_size)
    return stitched_rows


def stitch_all_chunks(
    chunks: Iterable[list[AlignedRow]],
    overlap_size: int,
) -> list[AlignedRow]:
    """
    Stitch an iterable of aligned chunks into a single alignment.

    The accumulator is trimmed only within the tail window, keeping memory
    bounded while preserving earlier rows.
    """
    iterator = iter(chunks)
    try:
        first_chunk = list(next(iterator))
    except StopIteration:
        return []

    accumulator = first_chunk
    W_max = 2 * overlap_size
    chunk_idx = 0

    for chunk in iterator:
        chunk_idx += 1
        chunk_list = list(chunk)
        tail_window = accumulator[-W_max:] if W_max > 0 else []

        head_window = chunk_list[:W_max] if W_max > 0 else []
        print(
            f"\n--- Before stitch: Accumulator tail (last {min(len(accumulator), W_max)} rows) ---"
        )
        _print_alignment(tail_window)
        print(
            f"\n--- Before stitch: New chunk head (first {min(len(chunk_list), W_max)} rows) ---"
        )
        _print_alignment(head_window)

        stitched_rows, trim_from_tail, anchor_found, anchor_in_head = _stitch_core(
            tail_window, chunk_list, overlap_size
        )

        print(f"\n--- After stitch: Rows to append ({len(stitched_rows)} rows) ---")
        _print_alignment(stitched_rows)

        if anchor_found and trim_from_tail:
            accumulator = accumulator[:-trim_from_tail]
        elif not anchor_found and anchor_in_head:
            permutable_len = _trailing_permutable_block_len(tail_window)
            trim_len = min(permutable_len, overlap_size)
            if trim_len:
                accumulator = accumulator[:-trim_len]

        accumulator.extend(stitched_rows)

    return accumulator


def _print_alignment(rows: list[AlignedRow]) -> None:
    """Render an alignment table to stdout for quick inspection."""
    max_left = max((len(item or "") for item, _ in rows), default=0)
    for idx, (left, right) in enumerate(rows, start=1):
        left_cell = (left or "").ljust(max_left)
        right_cell = right or ""
        print(f"{idx:3d}. {left_cell} | {right_cell}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch multiple aligned chunks together using anchor-based alignment."
    )
    parser.add_argument(
        "--case-indices",
        type=lambda x: [int(i) for i in x.split(",") if i.strip()],
        default=[0, 1, 2],
        help="Comma-separated list of case indices from tests/fixtures/case_mixed1.py (default: 0,1,2)",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=2,
        help="Overlap size for stitching (default: 2)",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.1,
        help="Cost for inserting a gap in alignment (default: 0.1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/nomic-embed-text",
        help="Embedding model identifier for litellm (default: ollama/nomic-embed-text)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: DEBUG)",
    )
    args = parser.parse_args()

    # Configure loguru logger
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{level: <8}</level> | {message}",
        level=args.log_level,
        colorize=True,
    )

    try:
        try:
            from .core import align_sequences
        except ImportError:
            from semantic_text_aligner.core import align_sequences
        from tests.fixtures.case_mixed1 import SUBCASES
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unable to import required modules: {exc}") from exc

    # Validate case indices
    for idx in args.case_indices:
        if not (0 <= idx < len(SUBCASES)):
            raise SystemExit(
                f"Case index {idx} is out of range. Must be between 0 and {len(SUBCASES) - 1}"
            )

    logger.info(
        f"Stitching {len(args.case_indices)} chunks with overlap_size={args.overlap_size}"
    )
    logger.info(f"Using cases: {args.case_indices}")

    # Align each case individually
    alignments = []
    for idx in args.case_indices:
        case = SUBCASES[idx]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Aligning case {idx}: {case.name}")
        logger.info(f"{'=' * 80}")
        logger.debug(f"Left input: {case.input.left}")
        logger.debug(f"Right input: {case.input.right}")

        aligned = align_sequences(
            (case.input.left, case.input.right),
            gap_penalty=args.gap_penalty,
            model=args.model,
        )
        alignments.append(aligned)
        logger.info(f"Case {idx} alignment complete: {len(aligned)} rows")
        print(f"\n=== Chunk {idx} Alignment ({case.name}) ===")
        _print_alignment(aligned)
        print()

    # Stitch all chunks together
    logger.info(f"\n{'=' * 80}")
    logger.info("Starting stitch_all_chunks")
    logger.info(f"{'=' * 80}")

    stitched_result = stitch_all_chunks(alignments, overlap_size=args.overlap_size)

    logger.info(f"\n{'=' * 80}")
    logger.info("Stitching complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Final stitched result: {len(stitched_result)} rows")
    print("\n=== FINAL STITCHED RESULT ===")
    _print_alignment(stitched_result)
    print()


if __name__ == "__main__":
    main()
