import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from semantic_text_aligner.stitcher import stitch_all_chunks  # noqa: E402
from semantic_text_aligner.stitcher import stitch_two_chunks


class StitcherExamplesTest(unittest.TestCase):
    def test_trivial_no_overlap(self) -> None:
        chunk_a = [
            ("a", "x"),
            ("b", "y"),
        ]
        chunk_b = [("c", "z")]

        result = stitch_all_chunks([chunk_a, chunk_b], overlap_size=0)

        self.assertEqual(
            result,
            [
                ("a", "x"),
                ("b", "y"),
                ("c", "z"),
            ],
        )

    def test_simple_edge_stitch(self) -> None:
        chunk_a = [
            ("a", "x"),
            ("b", "y"),
            ("c", "z"),
            ("d", None),
        ]
        chunk_b = [
            ("c", "z"),
            ("d", None),
            ("e", "w"),
        ]

        result = stitch_all_chunks([chunk_a, chunk_b], overlap_size=2)

        self.assertEqual(
            result,
            [
                ("a", "x"),
                ("b", "y"),
                ("c", "z"),
                ("d", None),
                ("e", "w"),
            ],
        )

    def test_gap_block_permutation(self) -> None:
        chunk_1 = [
            ("dog", None),
            (None, "cat"),
            ("pizza", None),
            (None, "mouse"),
        ]
        chunk_2 = [
            (None, "mouse"),
            ("pizza", "pizza pie"),
            ("house", None),
        ]

        result = stitch_all_chunks([chunk_1, chunk_2], overlap_size=2)

        self.assertEqual(
            result,
            [
                ("dog", None),
                (None, "cat"),
                (None, "mouse"),
                ("pizza", "pizza pie"),
                ("house", None),
            ],
        )

    def test_complex_gap_block_permutation(self) -> None:
        chunk_a = [
            ("a", None),
            ("b", None),
            (None, "x"),
            (None, "y"),
            ("c", "z"),
            ("d", "w"),
        ]
        chunk_b = [
            (None, "y"),
            (None, "x"),
            ("a", None),
            ("b", None),
            ("c", "z"),
            ("d", "w"),
            ("e", None),
        ]

        result = stitch_all_chunks([chunk_a, chunk_b], overlap_size=3)

        self.assertEqual(
            result,
            [
                ("a", None),
                ("b", None),
                (None, "x"),
                (None, "y"),
                ("c", "z"),
                ("d", "w"),
                ("e", None),
            ],
        )

    def test_no_canonical_anchor(self) -> None:
        tail = [
            ("a", None),
            (None, "b"),
        ]
        chunk = [
            (None, "c"),
            (None, "d"),
        ]

        stitched = stitch_two_chunks(tail, chunk, overlap_size=1)
        self.assertEqual(stitched, chunk)

    def test_overlap_upgrades_gap_row(self) -> None:
        chunk_a = [
            (
                "create a list of emergency contacts and distribute it to family members",
                "create a list of emergency contacts",
            ),
            (None, "distribute it to family members"),
            (
                "share stories and memories with family members",
                "share stories and memories",
            ),
            ("set boundaries for personal time", None),
        ]
        chunk_b = [
            (None, "distribute it to family members"),
            (
                "share stories and memories with family members",
                "share stories and memories",
            ),
            ("set boundaries for personal time", "set boundaries for personal time"),
            (
                "share and teach family recipes to the next generation",
                "share and teach family recipes",
            ),
            ("read a chapter from a book", None),
        ]

        stitched = stitch_all_chunks([chunk_a, chunk_b], overlap_size=2)

        self.assertEqual(
            stitched,
            [
                (
                    "create a list of emergency contacts and distribute it to family members",
                    "create a list of emergency contacts",
                ),
                (None, "distribute it to family members"),
                (
                    "share stories and memories with family members",
                    "share stories and memories",
                ),
                (
                    "set boundaries for personal time",
                    "set boundaries for personal time",
                ),
                (
                    "share and teach family recipes to the next generation",
                    "share and teach family recipes",
                ),
                ("read a chapter from a book", None),
            ],
        )

    def test_permutable_gap_block_with_upgrade(self) -> None:
        chunk_a = [
            ("X", "y"),
            ("A", None),
            (None, "B"),
            ("C", None),
        ]
        chunk_b = [
            (None, "B"),
            ("A", None),
            ("C", "c"),
            ("Z", "w"),
        ]

        stitched = stitch_all_chunks([chunk_a, chunk_b], overlap_size=2)

        self.assertEqual(
            stitched,
            [
                ("X", "y"),
                ("A", None),
                (None, "B"),
                ("C", "c"),
                ("Z", "w"),
            ],
        )

    def test_head_gap_row_absorbed_by_full_tail_row(self) -> None:
        chunk_a = [
            (
                "create a list of emergency contacts and distribute it to family members",
                "create a list of emergency contacts",
            ),
            (None, "distribute it to family members"),
            (
                "share stories and memories with family members",
                "share stories and memories",
            ),
            ("set boundaries for personal time", "set boundaries for personal time"),
            (
                "share and teach family recipes to the next generation",
                "share and teach family recipes",
            ),
            ("read a chapter from a book", None),
        ]
        chunk_b = [
            (None, "set boundaries for personal time"),
            (
                "share and teach family recipes to the next generation",
                "share and teach family recipes",
            ),
            ("read a chapter from a book", "read a chapter from a book"),
            ("plan and cook a healthy dinner", "plan and cook a healthy dinner"),
        ]

        stitched = stitch_all_chunks([chunk_a, chunk_b], overlap_size=2)

        self.assertEqual(
            stitched,
            [
                (
                    "create a list of emergency contacts and distribute it to family members",
                    "create a list of emergency contacts",
                ),
                (None, "distribute it to family members"),
                (
                    "share stories and memories with family members",
                    "share stories and memories",
                ),
                (
                    "set boundaries for personal time",
                    "set boundaries for personal time",
                ),
                (
                    "share and teach family recipes to the next generation",
                    "share and teach family recipes",
                ),
                ("read a chapter from a book", "read a chapter from a book"),
                ("plan and cook a healthy dinner", "plan and cook a healthy dinner"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
