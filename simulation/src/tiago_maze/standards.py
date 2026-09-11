"""The four fixed 40-corner mazes used by the recorded experiments.

Historical reports show these layouts under seeds 11--14.  Test numbers were
counterbalanced between subjects, so a report's TEST1 is not a maze identity;
the seed is.  The routes are frozen here so future generator edits cannot
silently change the experimental standard.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardMaze:
    number: int
    seed: int
    route: str

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple("LEFT" if letter == "L" else "RIGHT" for letter in self.route)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(1 if letter == "L" else 2 for letter in self.route)


STANDARD_MAZES = {
    1: StandardMaze(1, 11, "LRLRLRRLRLRLLLRLRRLRLLRRLRRLLRLRLLRRLRLL"),
    2: StandardMaze(2, 12, "RLRRLRLRRLRRLRLLLRLRLRLLRLRRLRLRLRLLRLRL"),
    3: StandardMaze(3, 13, "RRLRRLLRLRLLRLRRLLRLLRLRLRLLRRRLLLRRLRLR"),
    4: StandardMaze(4, 14, "LRLRLRRRLLRLLRLRLRLLRLRLRLRRLRLLRRLRLRRL"),
}


def get_standard_maze(number: int) -> StandardMaze:
    try:
        return STANDARD_MAZES[int(number)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("standard maze must be 1, 2, 3, or 4") from error
