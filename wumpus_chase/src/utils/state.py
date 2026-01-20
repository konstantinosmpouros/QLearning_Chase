"""State encoding utilities."""

from typing import Tuple


def encode_state(s: Tuple[int, int, int, int], size: int) -> int:
    ax, ay, bx, by = s
    a_id = ax * size + ay
    b_id = bx * size + by
    return a_id * (size * size) + b_id


def decode_state(idx: int, size: int) -> Tuple[int, int, int, int]:
    a_id, b_id = divmod(idx, size * size)
    ax, ay = divmod(a_id, size)
    bx, by = divmod(b_id, size)
    return (ax, ay, bx, by)
