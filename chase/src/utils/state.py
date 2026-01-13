"""
State encoding utilities.
"""

from typing import Tuple


def encode_state(s: Tuple[int, int, int, int], size: int = 5) -> int:
    """Encode 4‑tuple state to integer index for tabular arrays."""
    cx, cy, rx, ry = s
    c_id = cx * size + cy
    r_id = rx * size + ry
    return c_id * (size * size) + r_id
