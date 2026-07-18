"""Generates the learning-curve figures of the report: learning_curve.pdf
(main DQN run) and learning_curve_bonus.pdf (2 algorithms x 2 rewards),
from the win rates stored during training.

Usage (from the repository root):
    python brimarl_masked/submit/make_plots.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN_RATES_PATH = "models_savings/2/DeepQAgent/win_rates.txt"
OUT_PATH = os.path.join(os.path.dirname(__file__), "learning_curve.pdf")

# bonus experiments: (label, win_rates path, color)
BONUS_RUNS = [
    ("DQN, standard", "models_savings/2/DeepQAgent/win_rates.txt", "#1f77b4"),
    ("DQN, own points", "models_savings/2/DeepQAgent_own/win_rates.txt", "#ff7f0e"),
    ("A2C, standard", "models_savings/2/ACAgentQuick/win_rates.txt", "#2ca02c"),
    ("A2C, own points", "models_savings/2/ACAgentQuick_own/win_rates.txt", "#d62728"),
]
OUT_PATH_BONUS = os.path.join(os.path.dirname(__file__), "learning_curve_bonus.pdf")
RANDOM_VS_SCRIPTED = 0.212    # luck floor: RandomAgent, 1000 games, seed 0

EVALUATE_EVERY = 250          # epochs between two evaluations during training
BASELINE_VS_SCRIPTED = 0.599  # HeuristicAgent, 1000 games, seed 0
BASELINE_VS_RANDOM = 0.839    # HeuristicAgent, 1000 games, seed 0
EPS_MIN_EPOCH = 4800          # epsilon reaches its minimum (0.1) at 80% of training

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    with open(WIN_RATES_PATH) as f:
        win_rates = json.load(f)
    epochs = [(i + 1) * EVALUATE_EVERY for i in range(len(win_rates))]
    vs_scripted = [w[0] for w in win_rates]
    vs_random = [w[1] for w in win_rates]

    fig, ax = plt.subplots(figsize=(3.25, 2.6))

    ax.plot(epochs, vs_scripted, color="#1f77b4", marker="o", markersize=2.5,
            linewidth=1.2, label="DQN vs ScriptedAIAgent")
    ax.plot(epochs, vs_random, color="#2ca02c", marker="o", markersize=2.5,
            linewidth=1.2, alpha=0.8, label="DQN vs RandomAgent")

    ax.axhline(BASELINE_VS_SCRIPTED, color="#1f77b4", linestyle="--",
               linewidth=1., label="HeuristicAgent vs Scripted (0.599)")
    ax.axhline(BASELINE_VS_RANDOM, color="#2ca02c", linestyle="--",
               linewidth=1., alpha=0.8, label="HeuristicAgent vs Random (0.839)")

    ax.axvline(EPS_MIN_EPOCH, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate(r"$\varepsilon = 0.1$", xy=(EPS_MIN_EPOCH, 0.06),
                xytext=(EPS_MIN_EPOCH - 1450, 0.06), fontsize=8, color="gray")

    ax.set_xlabel("Training epoch (1 game each)")
    ax.set_ylabel("Win rate")
    ax.set_xlim(0, 6100)
    ax.set_ylim(0., 1.)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)

    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


def main_bonus():
    fig, ax = plt.subplots(figsize=(3.25, 2.6))

    for label, path, color in BONUS_RUNS:
        with open(path) as f:
            win_rates = json.load(f)
        epochs = [(i + 1) * EVALUATE_EVERY for i in range(len(win_rates))]
        ax.plot(epochs, [w[0] for w in win_rates], color=color, marker="o",
                markersize=2, linewidth=1.1, label=label)

    ax.axhline(BASELINE_VS_SCRIPTED, color="gray", linestyle="--",
               linewidth=1., label="HeuristicAgent (0.599)")
    ax.axhline(RANDOM_VS_SCRIPTED, color="gray", linestyle=":",
               linewidth=0.8, label="RandomAgent (0.212)")

    ax.set_xlabel("Training epoch (1 game each)")
    ax.set_ylabel("Win rate vs ScriptedAIAgent")
    ax.set_xlim(0, 6100)
    ax.set_ylim(0., 1.)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=2)

    fig.savefig(OUT_PATH_BONUS, bbox_inches="tight")
    print(f"Saved {OUT_PATH_BONUS}")


if __name__ == "__main__":
    main()
    main_bonus()
