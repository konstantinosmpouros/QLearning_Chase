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
            "role_a",
            "role_b",
            "episode",
            "step",
            "size",
            "p_fail",
            "t_max",
            "step_penalty",
            "outcome_reward",
            "obstacle_penalty",
            "chase_dist_reward",
            "treasure_dist_reward",
            "layout_treasure_x",
            "layout_treasure_y",
            "layout_wumpus_x",
            "layout_wumpus_y",
            "layout_obstacles",
            "layout_pits",
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
            "reward_a",
            "reward_b",
            "done",
            "outcome",
            "winner_role",
            "capture",
            "a_dead",
            "b_dead",
            "a_treasure",
            "b_treasure",
            "a_fail",
            "b_fail",
            "a_hit_obstacle",
            "b_hit_obstacle",
            "a_bumped",
            "b_bumped",
            "a_in_pit",
            "b_in_pit",
            "a_met_wumpus",
            "b_met_wumpus",
            "a_breeze",
            "a_stench",
            "b_breeze",
            "b_stench",
            "dist_ab_before",
            "dist_ab_after",
            "dist_ab_delta",
            "a_treasure_before",
            "a_treasure_after",
            "b_treasure_before",
            "b_treasure_after",
            "a_treasure_dist",
            "b_treasure_dist",
            "reward_outcome",
            "reward_step",
            "reward_hazard",
            "reward_bump",
            "reward_perception",
            "reward_obstacle",
            "reward_chase",
            "reward_treasure",
            "sid",
            "sid_next",
            "epsilon",
            "pi_a",
            "pi_b",
            "value_a",
            "value_b",
        ]
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def _obstacles_str(self, obstacles: Iterable[tuple[int, int]]) -> str:
        return ";".join(f"{x},{y}" for x, y in sorted(obstacles))

    def _points_str(self, points: Iterable[tuple[int, int]]) -> str:
        return ";".join(f"{x},{y}" for x, y in sorted(points))

    def _manhattan(self, p1, p2) -> int:
        if not isinstance(p1, tuple) or not isinstance(p2, tuple):
            return 0
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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
        **extra,
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
        state_ax = state[0] if isinstance(state, (tuple, list)) and len(state) == 4 else -1
        state_ay = state[1] if isinstance(state, (tuple, list)) and len(state) == 4 else -1
        state_bx = state[2] if isinstance(state, (tuple, list)) and len(state) == 4 else -1
        state_by = state[3] if isinstance(state, (tuple, list)) and len(state) == 4 else -1
        next_state_ax = next_state[0] if isinstance(next_state, (tuple, list)) and len(next_state) == 4 else -1
        next_state_ay = next_state[1] if isinstance(next_state, (tuple, list)) and len(next_state) == 4 else -1
        next_state_bx = next_state[2] if isinstance(next_state, (tuple, list)) and len(next_state) == 4 else -1
        next_state_by = next_state[3] if isinstance(next_state, (tuple, list)) and len(next_state) == 4 else -1

        a_old = info.get("a_old")
        b_old = info.get("b_old")
        a_new = info.get("a_new")
        b_new = info.get("b_new")

        dist_ab_before = info.get("dist_ab_before", self._manhattan(a_old, b_old))
        dist_ab_after = info.get("dist_ab_after", self._manhattan(a_new, b_new))
        dist_ab_delta = info.get("dist_ab_delta", dist_ab_after - dist_ab_before)

        a_treasure_before = info.get(
            "a_treasure_before",
            self._manhattan(a_old, layout.treasure),
        )
        a_treasure_after = info.get(
            "a_treasure_after",
            self._manhattan(a_new, layout.treasure),
        )
        b_treasure_before = info.get(
            "b_treasure_before",
            self._manhattan(b_old, layout.treasure),
        )
        b_treasure_after = info.get(
            "b_treasure_after",
            self._manhattan(b_new, layout.treasure),
        )

        reward_step = info.get("reward_step", 0.0)
        reward_bump = info.get("reward_bump", 0.0)
        reward_obstacle = info.get("reward_obstacle", reward_bump)
        row = {
            "run_label": run_label,
            "phase": phase,
            "agent_a": agent_a or "",
            "agent_b": agent_b or "",
            "role_a": getattr(env, "role_a", "A"),
            "role_b": getattr(env, "role_b", "B"),
            "episode": episode,
            "step": step_idx,
            "size": env.size,
            "p_fail": env.p_fail,
            "t_max": env.t_max,
            "step_penalty": env.step_penalty,
            "outcome_reward": getattr(env, "outcome_reward", None),
            "obstacle_penalty": getattr(env, "obstacle_penalty", None),
            "chase_dist_reward": getattr(env, "chase_dist_reward", None),
            "treasure_dist_reward": getattr(env, "treasure_dist_reward", None),
            "layout_treasure_x": layout.treasure[0],
            "layout_treasure_y": layout.treasure[1],
            "layout_wumpus_x": layout.wumpus[0],
            "layout_wumpus_y": layout.wumpus[1],
            "layout_obstacles": self._obstacles_str(layout.obstacles),
            "layout_pits": self._points_str(getattr(layout, "pits", [])),
            "state_ax": state_ax,
            "state_ay": state_ay,
            "state_bx": state_bx,
            "state_by": state_by,
            "action_a": action_a,
            "action_b": action_b,
            "action_a_name": action_a_name,
            "action_b_name": action_b_name,
            "next_state_ax": next_state_ax,
            "next_state_ay": next_state_ay,
            "next_state_bx": next_state_bx,
            "next_state_by": next_state_by,
            "reward": reward,
            "reward_a": info.get("reward_a", reward),
            "reward_b": info.get("reward_b", -reward),
            "done": bool(done),
            "outcome": info.get("outcome") or "IN_PROGRESS",
            "winner_role": info.get("winner_role", ""),
            "capture": bool(info.get("capture", False)),
            "a_dead": bool(info.get("a_dead", False)),
            "b_dead": bool(info.get("b_dead", False)),
            "a_treasure": bool(info.get("a_treasure", False)),
            "b_treasure": bool(info.get("b_treasure", False)),
            "a_fail": bool(info.get("a_fail", False)),
            "b_fail": bool(info.get("b_fail", False)),
            "a_hit_obstacle": bool(info.get("a_hit_obstacle", info.get("a_bumped", False))),
            "b_hit_obstacle": bool(info.get("b_hit_obstacle", info.get("b_bumped", False))),
            "a_bumped": bool(info.get("a_bumped", False)),
            "b_bumped": bool(info.get("b_bumped", False)),
            "a_in_pit": bool(info.get("a_in_pit", False)),
            "b_in_pit": bool(info.get("b_in_pit", False)),
            "a_met_wumpus": bool(info.get("a_met_wumpus", False)),
            "b_met_wumpus": bool(info.get("b_met_wumpus", False)),
            "a_breeze": bool(info.get("a_breeze", False)),
            "a_stench": bool(info.get("a_stench", False)),
            "b_breeze": bool(info.get("b_breeze", False)),
            "b_stench": bool(info.get("b_stench", False)),
            "dist_ab_before": dist_ab_before,
            "dist_ab_after": dist_ab_after,
            "dist_ab_delta": dist_ab_delta,
            "a_treasure_before": a_treasure_before,
            "a_treasure_after": a_treasure_after,
            "b_treasure_before": b_treasure_before,
            "b_treasure_after": b_treasure_after,
            "a_treasure_dist": info.get("a_treasure_dist", a_treasure_after),
            "b_treasure_dist": info.get("b_treasure_dist", b_treasure_after),
            "reward_outcome": info.get("reward_outcome", 0.0),
            "reward_step": reward_step,
            "reward_hazard": info.get("reward_hazard", 0.0),
            "reward_bump": reward_bump,
            "reward_perception": info.get("reward_perception", 0.0),
            "reward_obstacle": reward_obstacle,
            "reward_chase": info.get("reward_chase", 0.0),
            "reward_treasure": info.get("reward_treasure", 0.0),
            "sid": sid,
            "sid_next": sid_next,
            "epsilon": 0.0,
            "pi_a": "[]",
            "pi_b": "[]",
            "value_a": 0.0,
            "value_b": 0.0,
        }
        for key, value in extra.items():
            if key in row:
                row[key] = value
        self._writer.writerow(row)
        self._file.flush()

    def save(self) -> None:
        self._file.flush()
        self._file.close()
