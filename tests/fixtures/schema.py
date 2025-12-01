"""
Shared structures for alignment fixtures.

Each fixture is defined in its own module as a single AlignmentCase instance
named CASE for easy discovery/import in tests.
"""

import csv
import io
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlignmentIO:
    """Input or expected alignment payload.

    Can be initialized with:
    - Explicit left/right: AlignmentIO(left=[...], right=[...])
    - CSV string: AlignmentIO(text="left1,right1\nleft2,right2\n...")
    """

    left: list[Optional[str]]
    right: list[Optional[str]]

    def __init__(
        self,
        left_or_text: str | list[Optional[str]] | None = None,
        right: list[Optional[str]] | None = None,
    ):
        """Initialize from explicit left/right or CSV text string.

        Args:
            left_or_text: List of left-side strings (or None for gaps) or CSV string with format "left,right\nleft,right\n..."
            right: List of right-side strings (or None for gaps)
        """
        if isinstance(left_or_text, str):
            # Parse CSV string
            reader = csv.reader(io.StringIO(left_or_text))
            parsed_left = []
            parsed_right = []
            for row in reader:
                if len(row) >= 2:
                    parsed_left.append(row[0].strip() if row[0].strip() else None)
                    parsed_right.append(row[1].strip() if row[1].strip() else None)
                elif len(row) == 1:
                    # Single column - treat as left, right is None
                    parsed_left.append(row[0].strip() if row[0].strip() else None)
                    parsed_right.append(None)
            self.left = parsed_left
            self.right = parsed_right
        elif isinstance(left_or_text, list) and right is not None:
            self.left = left_or_text
            self.right = right
        else:
            raise ValueError(
                "Must provide either 'text' (CSV string) or both 'left' and 'right' lists"
            )


@dataclass(frozen=True)
class AlignmentCase:
    """Single alignment scenario with inputs and expected aligned outputs."""

    name: str
    description: str
    input: AlignmentIO
    expected: AlignmentIO
    model: str = "ollama/gemma3:12b"
