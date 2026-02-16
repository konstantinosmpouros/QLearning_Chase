"""
Extended map layouts for Wumpus Chase with pits and breeze.

Classic Wumpus World elements:
- Wumpus: Death if stepped on. Adjacent cells have "stench".
- Pits: Death if stepped on. Adjacent cells have "breeze".
- Treasure: Win condition.
- Obstacles: Impassable cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple, Set


@dataclass(frozen=True)
class ExtendedMapLayout:
    """
    Extended map layout including pits for breeze perception.
    
    Attributes:
        size: Grid dimension (size x size)
        obstacles: Impassable cells
        wumpus: Wumpus location (stench in adjacent cells)
        pits: Pit locations (breeze in adjacent cells)
        treasure: Goal location
    """
    size: int
    obstacles: FrozenSet[Tuple[int, int]]
    wumpus: Tuple[int, int]
    pits: FrozenSet[Tuple[int, int]]
    treasure: Tuple[int, int]
    
    def get_adjacent_cells(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all valid adjacent cells (4-connectivity)."""
        x, y = pos
        adjacent = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                adjacent.append((nx, ny))
        return set(adjacent)
    
    def get_breeze_cells(self) -> FrozenSet[Tuple[int, int]]:
        """Get all cells where breeze is perceived (adjacent to pits)."""
        breeze_cells = set()
        for pit in self.pits:
            breeze_cells.update(self.get_adjacent_cells(pit))
        # Remove pit cells themselves and obstacles
        breeze_cells -= self.pits
        breeze_cells -= self.obstacles
        return frozenset(breeze_cells)
    
    def get_stench_cells(self) -> FrozenSet[Tuple[int, int]]:
        """Get all cells where stench is perceived (adjacent to wumpus)."""
        stench_cells = self.get_adjacent_cells(self.wumpus)
        stench_cells -= self.obstacles
        stench_cells.discard(self.wumpus)
        return frozenset(stench_cells)
    
    def is_safe_cell(self, pos: Tuple[int, int]) -> bool:
        """Check if a cell is safe (not pit, wumpus, or obstacle)."""
        return (
            pos not in self.pits and 
            pos != self.wumpus and 
            pos not in self.obstacles
        )
    
    def get_free_cells(self) -> FrozenSet[Tuple[int, int]]:
        """Get all traversable cells (excluding obstacles, wumpus, pits)."""
        free = set()
        for x in range(self.size):
            for y in range(self.size):
                pos = (x, y)
                if self.is_safe_cell(pos) and pos != self.treasure:
                    free.add(pos)
        return frozenset(free)


def default_extended_layout() -> ExtendedMapLayout:
    """
    Default 7x7 layout with wumpus, pits, and treasure.
    
    Layout visualization (0-indexed, x increases downward, y increases rightward):
    
        0   1   2   3   4   5   6
      +---+---+---+---+---+---+---+
    0 |   |   |   |   |   |   |   |
      +---+---+---+---+---+---+---+
    1 |   |   |   |   | # |   T   |  T = Treasure
      +---+---+---+---+---+---+---+
    2 |   |   | # | # |   |   |   |  # = Obstacle
      +---+---+---+---+---+---+---+
    3 |   |   | # |   | P |   |   |  P = Pit
      +---+---+---+---+---+---+---+
    4 |   |   |   |   | # |   |   |  W = Wumpus
      +---+---+---+---+---+---+---+
    5 |   | P |   |   |   | W |   |
      +---+---+---+---+---+---+---+
    6 |   |   |   |   |   |   |   |
      +---+---+---+---+---+---+---+
    
    Breeze cells: adjacent to pits (3,3), (3,5), (5,0), (5,2), (4,1), (6,1)
    Stench cells: adjacent to wumpus (4,5), (5,4), (5,6), (6,5)
    """
    size = 7
    obstacles = frozenset({
        (2, 2),
        (2, 3),
        (3, 2),
        (4, 4),
        (1, 4),
    })
    wumpus = (5, 5)
    pits = frozenset({
        (3, 4),  # Pit near center
        (5, 1),  # Pit on left side
    })
    treasure = (1, 5)
    
    return ExtendedMapLayout(
        size=size,
        obstacles=obstacles,
        wumpus=wumpus,
        pits=pits,
        treasure=treasure,
    )


def small_layout() -> ExtendedMapLayout:
    """
    Smaller 5x5 layout for faster training/testing.
    
        0   1   2   3   4
      +---+---+---+---+---+
    0 |   |   |   |   |   |
      +---+---+---+---+---+
    1 |   | # |   |   | T |  T = Treasure
      +---+---+---+---+---+
    2 |   | # | P |   |   |  P = Pit
      +---+---+---+---+---+
    3 |   |   |   | W |   |  W = Wumpus
      +---+---+---+---+---+
    4 |   |   |   |   |   |
      +---+---+---+---+---+
    """
    size = 5
    obstacles = frozenset({
        (1, 1),
        (2, 1),
    })
    wumpus = (3, 3)
    pits = frozenset({
        (2, 2),
    })
    treasure = (1, 4)
    
    return ExtendedMapLayout(
        size=size,
        obstacles=obstacles,
        wumpus=wumpus,
        pits=pits,
        treasure=treasure,
    )


def dangerous_layout() -> ExtendedMapLayout:
    """
    More dangerous 7x7 layout with multiple pits.
    """
    size = 7
    obstacles = frozenset({
        (3, 3),
    })
    wumpus = (5, 5)
    pits = frozenset({
        (1, 2),
        (2, 5),
        (4, 1),
        (5, 3),
    })
    treasure = (0, 6)
    
    return ExtendedMapLayout(
        size=size,
        obstacles=obstacles,
        wumpus=wumpus,
        pits=pits,
        treasure=treasure,
    )


def visualize_layout(layout: ExtendedMapLayout) -> str:
    """Create ASCII visualization of the layout."""
    breeze_cells = layout.get_breeze_cells()
    stench_cells = layout.get_stench_cells()
    
    lines = []
    header = "    " + "   ".join(str(i) for i in range(layout.size))
    lines.append(header)
    lines.append("  +" + "---+" * layout.size)
    
    for x in range(layout.size):
        row = f"{x} |"
        for y in range(layout.size):
            pos = (x, y)
            if pos == layout.wumpus:
                cell = " W "
            elif pos == layout.treasure:
                cell = " T "
            elif pos in layout.pits:
                cell = " P "
            elif pos in layout.obstacles:
                cell = " # "
            elif pos in breeze_cells and pos in stench_cells:
                cell = "BS "  # Both breeze and stench
            elif pos in breeze_cells:
                cell = " b "  # Breeze
            elif pos in stench_cells:
                cell = " s "  # Stench
            else:
                cell = "   "
            row += cell + "|"
        lines.append(row)
        lines.append("  +" + "---+" * layout.size)
    
    legend = "\nLegend: W=Wumpus, T=Treasure, P=Pit, #=Obstacle, b=breeze, s=stench, BS=both"
    lines.append(legend)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test and visualize layouts
    print("=== Default Extended Layout ===")
    layout = default_extended_layout()
    print(visualize_layout(layout))
    print(f"\nBreeze cells: {sorted(layout.get_breeze_cells())}")
    print(f"Stench cells: {sorted(layout.get_stench_cells())}")
    print(f"Free cells: {len(layout.get_free_cells())}")
    
    print("\n\n=== Small Layout ===")
    small = small_layout()
    print(visualize_layout(small))
    
    print("\n\n=== Dangerous Layout ===")
    dangerous = dangerous_layout()
    print(visualize_layout(dangerous))
