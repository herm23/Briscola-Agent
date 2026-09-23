# Briscola RL Agent

Project for the Reinforcement Learning course 2025/26 (University of Padua):
a reinforcement learning agent (DQN) that plays two-player **Briscola** against
the provided rule-based `ScriptedAIAgent`, compared with a hand-written
heuristic baseline that counts cards.

📄 **Report:** [`RL_project__Briscola_.pdf`](RL_project__Briscola_.pdf) covers
the baseline, the choice of algorithm, the learning curves and the discussion.

The code lives in the `brimarl_masked/` package. The name cannot be changed
because every internal import uses `brimarl_masked.*`. The original project
README, which describes each module, is in
[`brimarl_masked/README.md`](brimarl_masked/README.md).

## Results

Seed 0, 1000 games against each opponent:

| Agent | Win rate vs `ScriptedAIAgent` | Win rate vs `RandomAgent` |
|-------|:-----------------------------:|:-------------------------:|
| Heuristic baseline (`HeuristicAgent`, card counting) | 59.9% | 83.9% |
| **DQN** (best checkpoint) | **52.3%** | 79.1% |

The DQN agent beats the scripted opponent but stays below the card-counting
baseline. The [report](RL_project__Briscola_.pdf) also covers an A2C agent and an alternative reward
("own points"); both did worse than the standard DQN.

## Setup

```bash
conda create -n RL python=3.12 -y
conda activate RL
pip install tensorflow numpy matplotlib tqdm
```

## Usage

Always run commands from the **repository root** (the folder that contains
`brimarl_masked/`). Run scripts as modules:

```bash
# heuristic baseline vs RandomAgent (smoke test, no learning)
python -m brimarl_masked.main.main

# training
python -m brimarl_masked.main.train_best_response              # DQN, standard reward
python -m brimarl_masked.main.train_best_response --reward own # DQN, "own points" reward
python -m brimarl_masked.main.train_a2c                        # A2C, standard reward
python -m brimarl_masked.main.train_a2c --reward own           # A2C, "own points" reward

# evaluation (1000 games vs ScriptedAIAgent + 1000 vs RandomAgent)
python evaluate.py
python evaluate.py --agent a2c --reward own --num_games 500
```

`evaluate.py` loads its weights from `models_savings/2/<Agent>/best/`, or from
the folder given with `--weights`. Training writes its checkpoints to
`models_savings/`.

You can also run a file directly:
`PYTHONPATH=. python brimarl_masked/main/train_best_response.py`.

Note: the `scritps` folder really is spelled that way. The typo comes from the
original code and the imports depend on it.
