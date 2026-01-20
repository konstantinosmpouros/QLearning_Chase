from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from utils import encode_state


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "run_label",
            "phase",
            "agent_a",
            "agent_b",
            "episode",
            "step",
            "size",
            "p_fail",
            "t_max",
            "step_penalty",
            "layout_treasure_x",
            "layout_treasure_y",
            "layout_wumpus_x",
            "layout_wumpus_y",
            "layout_obstacles",
            "state_ax",
            "state_ay",
            "state_bx",
            "state_by",
            "action_a",
            "action_b",
            "action_a_name",
            "action_b_name",
            "next_state_ax",
            "next_state_ay",
            "next_state_bx",
            "next_state_by",
            "reward",
            "done",
            "outcome",
            "capture",
            "a_dead",
            "b_dead",
            "a_treasure",
            "b_treasure",
            "a_fail",
            "b_fail",
            "sid",
            "sid_next",
        ]
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def _obstacles_str(self, obstacles: Iterable[tuple[int, int]]) -> str:
        return ";".join(f"{x},{y}" for x, y in sorted(obstacles))

    def log_step(
        self,
        *,
        run_label: str,
        phase: str,
        env,
        episode: int,
        step_idx: int,
        state,
        action_a: int,
        action_b: int,
        next_state,
        reward: float,
        done: bool,
        info: dict,
        actions: Sequence[str] | None = None,
        agent_a: str | None = None,
        agent_b: str | None = None,
        sid: int | None = None,
        sid_next: int | None = None,
    ) -> None:
        if sid is None:
            sid = encode_state(state, env.size)
        if sid_next is None:
            sid_next = encode_state(next_state, env.size)

        if actions is None:
            action_a_name = str(action_a)
            action_b_name = str(action_b)
        else:
            action_a_name = actions[action_a]
            action_b_name = actions[action_b]

        layout = env.layout
        row = {
            "run_label": run_label,
            "phase": phase,
            "agent_a": agent_a or "",
            "agent_b": agent_b or "",
            "episode": episode,
            "step": step_idx,
            "size": env.size,
            "p_fail": env.p_fail,
            "t_max": env.t_max,
            "step_penalty": env.step_penalty,
            "layout_treasure_x": layout.treasure[0],
            "layout_treasure_y": layout.treasure[1],
            "layout_wumpus_x": layout.wumpus[0],
            "layout_wumpus_y": layout.wumpus[1],
            "layout_obstacles": self._obstacles_str(layout.obstacles),
            "state_ax": state[0],
            "state_ay": state[1],
            "state_bx": state[2],
            "state_by": state[3],
            "action_a": action_a,
            "action_b": action_b,
            "action_a_name": action_a_name,
            "action_b_name": action_b_name,
            "next_state_ax": next_state[0],
            "next_state_ay": next_state[1],
            "next_state_bx": next_state[2],
            "next_state_by": next_state[3],
            "reward": reward,
            "done": done,
            "outcome": info.get("outcome"),
            "capture": info.get("capture"),
            "a_dead": info.get("a_dead"),
            "b_dead": info.get("b_dead"),
            "a_treasure": info.get("a_treasure"),
            "b_treasure": info.get("b_treasure"),
            "a_fail": info.get("a_fail"),
            "b_fail": info.get("b_fail"),
            "sid": sid,
            "sid_next": sid_next,
        }
        self._writer.writerow(row)
        self._file.flush()

    def save(self) -> None:
        self._file.flush()
        self._file.close()
