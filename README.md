# Project Q-Catch

Multi-Agent Reinforcement Learning in adversarial gridworld environments.

**Authors:** Christos Karatsalos (mtn2510) · Konstantinos Mpouros (mtn2517)  
**Supervisor:** Prof. George Vouros  
**Course:** Intelligent Agents & Multi-Agentic Systems — MSc in Artificial Intelligence  
**Institution:** University of Piraeus & NCSR "DEMOKRITOS", Greece  

---

## Overview

Q-Catch implements and compares Q-learning-based methods for two-player pursuit–evasion games. Two scenarios of increasing complexity are provided:

| Scenario | Grid | Agents | Hazards | Extra mechanics |
|---|---|---|---|---|
| **Base Chase** | 5 × 5 | Catcher vs Runner | — | Simultaneous moves, stochastic action failure |
| **Wumpus Chase** | 7 × 7 | Chaser vs Runner (fixed roles) | Pits, Wumpus | Treasure goal, breeze/stench percepts, no-draw terminal priority |

Both scenarios use simultaneous actions, ε-greedy exploration with linear decay, and a shaped reward signal combining outcome, step, distance, and hazard terms.

### Learning algorithms

| Algorithm | Type | Scenario |
|---|---|---|
| **Fictitious Play (FP)** | Belief-based 1-step lookahead | Chase, Wumpus |
| **Minimax-Q** | Tabular zero-sum RL + LP security policy | Chase, Wumpus |
| **Nash-Q** | General-sum tabular RL + Nash equilibrium | Wumpus |
| **Dyna-Q / Dyna-Q+** | Model-based planning (+ exploration bonus) | Wumpus |
| **DQN** | Deep Q-Network (optional, requires PyTorch) | Wumpus |

---

## Repository structure

```
QLearning-Chase/
├── requirements.txt              # Shared dependencies (Chase)
├── chase/                        # Scenario 1 – Base Chase (5×5)
│   ├── src/
│   │   ├── env/tag_env.py        # TagEnv: grid, actions, capture rules
│   │   ├── agents/               # FPAgent, MinimaxQAgent
│   │   ├── train/                # Training loops per matchup
│   │   ├── logger/               # ExcelLogger (step-level .xlsx output)
│   │   ├── utils/                # LP solver, state encoding
│   │   └── chase.py              # Entry point
│   ├── notebook/                 # Jupyter analysis (train + eval)
│   └── results/                  # Excel logs (train / eval)
│
└── wumpus_chase/                 # Scenario 2 – Wumpus Chase (7×7)
    ├── src/
    │   ├── env/
    │   │   ├── wumpus_env_extended.py   # WumpusChaseEnvExtended
    │   │   └── maps_extended.py         # Layouts (small / default / dangerous)
    │   ├── agents/               # FP, MinimaxQ, NashQ, DynaQ, DQN
    │   ├── train/                # Training loops per method
    │   ├── cli/                  # Argparse CLI (train.py, eval_all.py)
    │   ├── logger/               # CSVLogger (step-level .csv output)
    │   ├── utils/                # LP solver, state encoding
    │   └── wumpus_chase_extended.py  # Entry point
    ├── notebook/                 # Jupyter analysis + exported figures
    ├── results/                  # Evaluation report
    └── results_dqn_tmp/          # DQN-specific logs
```

---

## Quick start

### 1. Install dependencies

```bash
# Core stack (both scenarios)
pip install -r requirements.txt

# Wumpus Chase extras (Nash-Q solver, optional DQN)
pip install -r wumpus_chase/requirements.txt
```

DQN training additionally requires **PyTorch ≥ 2.1**. If PyTorch is not installed the Wumpus CLI will skip DQN matchups automatically.

### 2. Run Base Chase

```bash
cd chase/src
python chase.py
```

This trains all four matchups (FP vs FP, Minimax-Q self-play, Minimax-Q vs FP, Minimax-Q vs Minimax-Q) for 10 000 episodes each at `p_fail ∈ {0.10, 0.20}`. Step-level logs are written to `chase/results/`.

### 3. Run Wumpus Chase

```bash
cd wumpus_chase/src
python wumpus_chase_extended.py [OPTIONS]
```

#### CLI options

| Flag | Default | Description |
|---|---|---|
| `--matchup` | `all` | Single matchup or `all` |
| `--suite` | `quick` | `quick` (self-play only) or `full` (all cross-play) |
| `--episodes` | `3000` | Training episodes per matchup |
| `--eval-every` | `100` | Evaluation checkpoint interval |
| `--eval-episodes` | `20` | Greedy episodes per checkpoint |
| `--layout` | `small` | Map layout: `small` (5×5), `default` (7×7), `dangerous` (7×7) |
| `--p-fail` | `0.10` | Action failure probability |
| `--t-max` | auto | Max steps per episode (30 for 5×5, 40 for 7×7) |
| `--planning-steps` | `10` | Simulated planning steps for Dyna-Q/Q+ |
| `--dqn-type` | `dueling_double` | DQN variant: `standard`, `double`, `dueling`, `dueling_double` |
| `--seed` | `42` | Base random seed |
| `--results-dir` | `results/` | Output directory for CSV logs |
| `--quiet` | — | Suppress per-episode console output |

#### Examples

```bash
# Quick self-play baselines on the default 7×7 map
python wumpus_chase_extended.py --suite quick --layout default --episodes 5000

# Full cross-play tournament
python wumpus_chase_extended.py --suite full --layout default --episodes 10000

# Single matchup
python wumpus_chase_extended.py --matchup dynaq_plus_selfplay --episodes 8000

# DQN self-play (requires PyTorch)
python wumpus_chase_extended.py --matchup dqn_vs_dqn --dqn-type dueling_double --episodes 5000
```

### 4. Analyse results

Jupyter notebooks are provided for both scenarios:

```bash
# Chase analysis
jupyter notebook chase/notebook/analytics_eval.ipynb

# Wumpus Chase analysis
jupyter notebook wumpus_chase/notebook/analytics_eval.ipynb
```

The notebooks load the step-level logs and produce rolling performance curves, win-rate summaries, terminal-reason breakdowns, spatial heatmaps, distance diagnostics, and reward decompositions.

---

## Environment details

### Base Chase (5 × 5)

```
    0   1   2   3   4
  +---+---+---+---+---+
0 |   |   |   |   |   |    A = Catcher (Agent A)
  +---+---+---+---+---+    B = Runner  (Agent B)
1 |   | A |   |   |   |
  +---+---+---+---+---+    Actions: STAY, UP, DOWN, LEFT, RIGHT
2 |   |   |   |   |   |    p_fail = 0.10 per action
  +---+---+---+---+---+    Capture: same-cell or cross-path swap
3 |   |   |   | B |   |    Termination: capture or t_max = 30
  +---+---+---+---+---+
4 |   |   |   |   |   |
  +---+---+---+---+---+
```

State: `(x_A, y_A, x_B, y_B)` → 625 joint states. Reward: capture ±10, step penalty −0.01 (Catcher only), distance shaping ±0.02.

### Wumpus Chase (7 × 7, default extended layout)

```
    0   1   2   3   4   5   6
  +---+---+---+---+---+---+---+
0 |   |   |   |   |   |   |   |    # = Obstacle
  +---+---+---+---+---+---+---+    P = Pit (lethal)
1 |   |   |   |   | # | T |   |    W = Wumpus (lethal)
  +---+---+---+---+---+---+---+    T = Treasure (goal)
2 |   |   | # | # | b |   |   |    b = Breeze (adjacent to pit)
  +---+---+---+---+---+---+---+    s = Stench (adjacent to Wumpus)
3 |   |   | # | b | P | b |   |
  +---+---+---+---+---+---+---+
4 |   | b |   |   | # | s |   |    Terminal priority (no draws):
  +---+---+---+---+---+---+---+      1. Capture → A_WIN
5 | b | P | b |   | s | W | s |      2. Treasure → first to reach wins
  +---+---+---+---+---+---+---+      3. Hazard → victim loses
6 |   | b |   |   |   | s |   |      4. Timeout (t ≥ 40) → B_WIN
  +---+---+---+---+---+---+---+
```

State: `(x_A, y_A, x_B, y_B)` → 2 401 joint states. Agents observe breeze/stench/glitter/bump. Reward includes outcome (±10), hazard penalties (−10), perception penalties, chase-distance shaping, and treasure-distance shaping.

---

## Key experimental results

### Base Chase

| Matchup | Capture rate | Avg steps (captured) |
|---|---|---|
| Minimax-Q vs Minimax-Q | Highest | Lowest (best) |
| FP vs FP | High | Mid-range |
| Minimax-Q self-play | High | Mid-range |
| Minimax-Q vs FP | ≈ 0 (timeout-dominated) | — |

Increasing `p_fail` from 0.1 to 0.2 preserves the relative ranking. Spatial analysis shows the Catcher adopts a center-control strategy while the Runner favours perimeter evasion.

### Wumpus Chase (win rates by role)

| Agent | Runner win rate | Chaser win rate | Best role |
|---|---|---|---|
| **DynaQ+** | **0.707** | 0.439 | Runner |
| **DynaQ** | 0.654 | **0.594** | Chaser |
| DQN | 0.652 | 0.348 | Runner |
| NashQ | 0.567 | 0.497 | Runner |
| FP | 0.536 | 0.425 | Runner |
| MinimaxQ | 0.279 | 0.464 | Chaser |

DynaQ is the strongest Chaser (only method with positive average terminal reward at +1.814). DynaQ+ is the strongest Runner (highest win rate and terminal reward at +4.104). Among hazard endings, pits account for ~92% of Chaser deaths and ~77% of Runner deaths.

---

## Shared hyperparameters

| Parameter | Chase | Wumpus Chase |
|---|---|---|
| Discount factor (γ) | 0.95 | 0.95 |
| Learning rate (α) | 0.10 | 0.10 |
| ε-greedy start → end | 0.20 → 0.05 | 0.30 → 0.05 |
| Evaluation episodes | 10 / checkpoint | 20 / checkpoint |
| `p_fail` | 0.10 | 0.10 |
| Dyna planning steps | — | 10 |

---

## References

1. Puterman, M. L. (1994). *Markov Decision Processes*. Wiley.
2. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
3. Shapley, L. S. (1953). Stochastic games. *PNAS*, 39(10), 1095–1100.
4. Littman, M. L. (1994). Markov Games as a Framework for Multi-Agent RL. *ICML*.
5. Hu, J. & Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games. *JMLR*, 4, 1039–1069.
6. Ng, A. Y., Harada, D. & Russell, S. (1999). Policy invariance under reward transformations. *ICML*.
7. Watkins, C. J. C. H. & Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3-4), 279–292.
8. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.

---

## License

Academic project — University of Piraeus & NCSR "DEMOKRITOS", February 2026.
