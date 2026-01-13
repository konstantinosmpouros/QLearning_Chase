"""
Excel logging utilities for the chase environment.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


@dataclass
class ExcelLogger:
    """
    Lightweight Excel logger that stores every step for each training run.
    Data is kept in memory and flushed to an xlsx file at the end of main.
    """

    path: Path | str = "results/chase_run.xlsx"
    sheets: Dict[str, List[Dict[str, object]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        base = Path(__file__).resolve().parent.parent.parent
        path = Path(self.path)
        if not path.is_absolute():
            path = base / path
        # Ensure parent directory exists (e.g., results/)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log_step(
        self,
        *,
        run_label: str,
        env: Any,
        p_fail: float,
        episode: int,
        step_idx: int,
        state: tuple,
        action_c: int,
        action_r: int,
        next_state: tuple,
        reward: float,
        done: bool,
        info: Dict,
        sid: int,
        sid_next: int,
        epsilon: float | None = None,
        value_estimate: float | None = None,
        extra: Dict | None = None,
        actions: Sequence[str] | None = None,
    ) -> None:
        action_c_name = (
            actions[action_c] if actions is not None and action_c < len(actions) else None
        )
        action_r_name = (
            actions[action_r] if actions is not None and action_r < len(actions) else None
        )
        record: Dict[str, object] = {
            "run": run_label,
            "p_fail": p_fail,
            "env_size": getattr(env, "size", None),
            "env_t_max": getattr(env, "t_max", None),
            "env_step_penalty": getattr(env, "step_penalty", None),
            "env_seed": getattr(env, "seed", None),
            "episode": episode,
            "step": step_idx,
            "env_t": info.get("t"),
            "sid": sid,
            "sid_next": sid_next,
            "catcher_x": state[0],
            "catcher_y": state[1],
            "runner_x": state[2],
            "runner_y": state[3],
            "catcher_action_id": action_c,
            "runner_action_id": action_r,
            "catcher_action": action_c_name,
            "runner_action": action_r_name,
            "catcher_x_next": next_state[0],
            "catcher_y_next": next_state[1],
            "runner_x_next": next_state[2],
            "runner_y_next": next_state[3],
            "reward": reward,
            "capture": info.get("capture", False),
            "done": done,
            "c_fail": info.get("c_fail"),
            "r_fail": info.get("r_fail"),
            "value_estimate": value_estimate,
            "epsilon": epsilon,
        }
        if extra:
            record.update(extra)
        self.sheets[run_label].append(record)

    def save(self) -> None:
        if not self.sheets:
            print("[ExcelLogger] No rows to write.")
            return
        with pd.ExcelWriter(self.path, engine="openpyxl") as writer:
            for sheet_name, rows in self.sheets.items():
                safe_sheet = (sheet_name or "run")[:31]
                pd.DataFrame(rows).to_excel(writer, sheet_name=safe_sheet, index=False)
        total_rows = sum(len(rows) for rows in self.sheets.values())
        print(f"[ExcelLogger] Wrote {total_rows} rows to {self.path}")
