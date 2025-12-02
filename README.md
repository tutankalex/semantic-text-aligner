Semantic Text Aligner
=====================

This project aligns two lists of strings using embedding-based dynamic time warping (DTW) and stitches chunked alignment outputs back into a single global alignment. It returns a `list[tuple[str | None, str | None]]` where each row pairs items from the left/right lists or `None` when a gap is inserted.

What it solves
--------------
- **Sequence alignment**: `core.py` embeds tokens with `litellm` and runs a DTW-style dynamic program with a gap penalty to align arbitrary lists of strings.
- **Chunk stitching**: `stitcher.py` reconstructs a full alignment from overlapping chunks. It uses a canonical anchor (first non-None/non-None row) to synchronize overlaps, merges compatible gap blocks, and upgrades gap-only rows when the corresponding token appears in the new chunk.
- Handles degenerate regions where gap-only rows can appear in different orders across chunks, normalizing them via anchor-based overlap matching.

Key concepts
------------
- **Aligned row**: `(left_item, right_item)` where each element is `str | None`.
- **Chunk**: List of aligned rows for a span of the inputs, extracted with an overlap window.
- **Overlap window**: Size `O`; maximum search window `W_max = 2 * O`.
- **Canonical anchor**: First row in a new chunk with both sides non-`None`; used to re-synchronize.
- **Permutable gap block**: Consecutive rows with exactly one `None`; these may reorder between chunks but must preserve column order.

APIs
----
- One-shot or chunked alignment:  
  `semantic_text_aligner.core.align_string_lists(input_data, chunk_size=None, overlap_size=None, gap_penalty=0.1, model="ollama/nomic-embed-text")`  
  - `chunk_size=None` runs a single full alignment.  
  - When chunked, `overlap_size=None` defaults to `min(4, chunk_size // 2)` and stitched output is returned.
- Low-level alignment (no stitching): `semantic_text_aligner.aligner.align_sequences(...)`
- Stitching helpers: `stitch_two_chunks` (returns rows to append) and `stitch_all_chunks`.

How stitching works
-------------------
1) Take `tail_window = prev_tail[-W_max:]` and `head_window = new_chunk[:W_max]` where `W_max = 2 * overlap_size`.
2) Find canonical anchor in `head_window`. If none, append `new_chunk` as-is.
3) Locate a compatible anchor in `tail_window` (exact match or mergeable gap/full rows).
4) Merge compatible overlap rows, upgrading gaps when the other chunk supplies the missing token, stopping on conflicts.
5) Rewrite any affected tail prefix, drop duplicate head rows already absorbed, trim the accumulator tail as needed, and append the stitched rows.

CLI usage
---------
Quick one-off alignment from the shell:
```
python - <<'PY'
from semantic_text_aligner.core import align_string_lists
left = ["a", "b", "c"]
right = ["a", "c", "d"]
print(align_string_lists((left, right)))
PY
```

Stitch multiple pre-aligned chunks (uses fixtures from `tests/fixtures/case_mixed1.py`):
```
python -m semantic_text_aligner.stitcher --case-indices 0,1,2 --overlap-size 2
```

Programmatic example
--------------------
```python
from semantic_text_aligner.core import align_string_lists

left = ["dog", "pizza", "house", "balloon"]
right = ["cat", "mouse", "pizza pie", "home"]

# Full alignment in one shot
full = align_string_lists((left, right))

# Or align in overlapping chunks for memory control
chunked = align_string_lists((left, right), chunk_size=3, overlap_size=None)
```
```

Testing
-------
Run the unit suite (covers the four canonical GOAL1 cases plus overlap/gap upgrades and edge cases):
```
python -m unittest discover -s tests -p 'test*.py'
```

Development notes
-----------------
- Stitching is append-oriented: `stitch_two_chunks` returns only the rows to append; callers manage the accumulator.
- The gap-upgrade logic treats compatible rows as equal if one side is `None` and the other provides the same token; conflicts stop overlap growth.
- Overlap trimming is bounded to the tail window to keep memory use predictable.
