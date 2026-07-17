"""Evaluation entry point: loads the trained DQN weights from memory and
evaluates the agent against the ScriptedAIAgent (and the RandomAgent as a
sanity check), printing the resulting win rates.

Usage (from the repository root):
    python evaluate.py [--num_games 1000] [--weights <dir>]
"""
import argparse
import random

import numpy as np
import tensorflow as tf

from brimarl_masked.environment.environment import BriscolaGame, BriscolaLogger
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.scritps.evaluate import evaluate

WEIGHTS_DIR = "models_savings/2/DeepQAgent/best"


def main(num_games: int, weights_dir: str):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = BriscolaGame(2, logger, win_extra_points=0)

    # greedy agent: epsilon 0 and training=False disable any exploration
    agent = DeepQAgent(epsilon=0., minimum_epsilon=0., epsilon_decay=0., training=False)
    agent.load_model(weights_dir)
    print(f"Loaded weights from {weights_dir}")

    print(f"\nEvaluating vs ScriptedAIAgent ({num_games} games)")
    evaluate(game, [agent, ScriptedAIAgent()], num_games)

    print(f"\nEvaluating vs RandomAgent ({num_games} games)")
    evaluate(game, [agent, RandomAgent()], num_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_games", type=int, default=1000,
                        help="number of evaluation games against each opponent")
    parser.add_argument("--weights", type=str, default=WEIGHTS_DIR,
                        help="directory containing the saved agent weights")
    args = parser.parse_args()
    main(args.num_games, args.weights)
