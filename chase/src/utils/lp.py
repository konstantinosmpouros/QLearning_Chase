"""
Linear program helper for maximin strategy.
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from env import A


def solve_row_player_maximin(Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve the row player's (catcher) maximin mixed strategy for zero‑sum payoff Q.
    We formulate a linear program:
      max v
      s.t. sum_a pi[a] Q[a,b] >= v for all b
           sum_a pi[a] = 1
           pi[a] >= 0
    Returns (pi, v).
    If LP fails, returns uniform mixed strategy.
    """
    assert Q.shape == (A, A)

    # Minimize negative v (i.e. maximize v).
    c = np.zeros(A + 1)
    c[-1] = -1.0

    # Inequalities: -∑_a pi[a] Q[a,b] + v <= 0  for each b
    A_ub_list: List[List[float]] = []
    b_ub_list: List[float] = []
    for b in range(A):
        row = np.zeros(A + 1)
        row[:A] = -Q[:, b]
        row[-1] = 1.0
        A_ub_list.append(row.tolist())
        b_ub_list.append(0.0)

    A_eq = np.zeros((1, A + 1))
    A_eq[0, :A] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * A + [(None, None)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub_list),
        b_ub=np.array(b_ub_list),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        pi = np.ones(A) / A
        v = float(np.min(pi @ Q))
        return pi, v

    x = res.x
    pi = np.clip(x[:A], 0.0, 1.0)
    pi /= (pi.sum() + 1e-12)
    v = float(x[-1])
    return pi, v
