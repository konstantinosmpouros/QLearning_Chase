# Wumpus Chase (Extended, Role-Driven)

Επέκταση του Wumpus Chase με:
- hazards (`pits`, `wumpus`)
- perceptions (`breeze`, `stench`)
- multi-agent training (FP, MinimaxQ, NashQ, DynaQ, DynaQ+, optional DQN)
- **fixed role gameplay** (`A=Chaser`, `B=Runner`)

## Core Scenario

Οι πράκτορες είναι μέσα στο ίδιο Wumpus περιβάλλον, αλλά με ασύμμετρους στόχους:

- **Chaser (A)**: να πιάσει τον Runner ή να πάρει το treasure
- **Runner (B)**: να αποφύγει τον Chaser και να πάρει το treasure

## Current Game Rules (No Draw)

Outcome priority per step:
1. Capture (same cell or cross-path) -> `A_WIN` (Chaser)
2. Treasure reached -> whoever reaches it wins
3. Hazards (pit/wumpus) -> if Runner dies then `A_WIN`, else `B_WIN`
4. Timeout -> `B_WIN` (Runner)

## Reward Design (Role-Driven, Non Zero-Sum)

Το περιβάλλον δίνει ξεχωριστά rewards:
- `reward_a` για Chaser
- `reward_b` για Runner

Και περιλαμβάνει:
- terminal outcome rewards
- role-specific distance shaping (chase/evade)
- treasure progress
- hazard, bump, perception penalties

## Run Training + Evaluation

```bash
python3 wumpus_chase/src/cli/train.py --matchup all --episodes 3000 --eval-every 100 --eval-episodes 20
```

Or via entrypoint wrapper:

```bash
python3 wumpus_chase/src/wumpus_chase_extended.py --matchup all
```

## Unified CLI

Βασικές επιλογές:
- `--matchup`
- `--episodes`
- `--eval-every`
- `--eval-episodes`
- `--suite [quick|full]`
- `--layout [small|default|dangerous]`
- `--seed`
- `--results-dir`
- `--quiet`

## Available Matchups

Core families:
- FP, MinimaxQ, NashQ, DynaQ
- DynaQ+ (selfplay and vs FP/MinimaxQ/NashQ/DynaQ)

Notes:
- `--matchup all --suite full` runs all currently supported matchups.
- If DQN dependencies are available, `dqn_vs_dqn` is also appended in full suite mode.

## Console / Logs

Το CLI summary εμφανίζει:
- `Win Rate` (A / Chaser win rate)
- `RunnerWin` (B / Runner win rate)
- `Avg Steps`
- `Avg Return`

CSV logs:
- `wumpus_chase/results/wumpus_extended_train.csv`
- `wumpus_chase/results/wumpus_extended_eval.csv`

The CSV logger includes role-aware columns such as:
- `role_a`, `role_b`, `winner_role`
- `reward_a`, `reward_b`

Evaluation is executed during training at each `--eval-every` interval.

## Quick Commands

```bash
# Full quick sweep
python3 wumpus_chase/src/cli/train.py --matchup all --suite quick

# Full suite
python3 wumpus_chase/src/cli/train.py --matchup all --suite full

# Single matchup
python3 wumpus_chase/src/cli/train.py --matchup nashq_vs_dynaq --episodes 1000 --eval-every 100 --eval-episodes 10 --layout small
```

## Architecture

```text
wumpus_chase/src/
├── env/
│   ├── maps_extended.py
│   └── wumpus_env_extended.py
├── agents/
│   ├── fp_agent.py
│   ├── minimax_q_agent.py
│   ├── nash_q_agent.py
│   ├── dyna_q_agent.py
│   └── dqn_agent.py
├── train/
│   ├── fp_training.py
│   ├── minimaxq_training.py
│   ├── nashq_training.py
│   ├── dynaq_training.py
│   └── dqn_training.py
├── cli/
│   └── train.py
└── logger/
    └── csv_logger.py
```
