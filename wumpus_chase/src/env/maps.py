from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class MapLayout:
    size: int
    obstacles: FrozenSet[Tuple[int, int]]
    wumpus: Tuple[int, int]
    treasure: Tuple[int, int]


def default_layout() -> MapLayout:
    size = 7
    obstacles = frozenset({
        (2, 2),
        (2, 3),
        (3, 2),
        (4, 4),
        (1, 4),
    })
    wumpus = (5, 5)
    treasure = (1, 5)
    return MapLayout(
        size=size,
        obstacles=obstacles,
        wumpus=wumpus,
        treasure=treasure,
    )
