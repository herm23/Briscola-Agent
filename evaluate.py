"""Evaluation entry point: loads the trained DQN weights from memory and
evaluates the agent against the ScriptedAIAgent (and the RandomAgent as a
sanity check), printing the resulting win rates.

Usage (from the repository root):
    python evaluate.py [--num_games 1000] [--weights <dir>]

Bonus experiments (Section "Bonus experiments" of the report): the same
protocol can evaluate the A2C agent and/or the agents trained with the
"own points" reward variant, selecting the corresponding best checkpoint:
    python evaluate.py --agent a2c
    python evaluate.py --reward own
    python evaluate.py --agent a2c --reward own
The default (no flags) evaluates the submitted DQN agent and is unchanged.
"""
import argparse
import random

import numpy as np
import tensorflow as tf

from brimarl_masked.environment.environment import BriscolaGame, BriscolaLogger
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.scritps.evaluate import evaluate

# best checkpoint of each training run (see report); the submission default
# is ("dqn", "standard") and its weights are untouched
WEIGHTS_DIRS = {
    ("dqn", "standard"): "models_savings/2/DeepQAgent/best",
    ("dqn", "own"): "models_savings/2/DeepQAgent_own/best",
    ("a2c", "standard"): "models_savings/2/ACAgentQuick/best",
    ("a2c", "own"): "models_savings/2/ACAgentQuick_own/best",
}


def main(num_games: int, weights_dir: str, agent_type: str = "dqn"):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = BriscolaGame(2, logger, win_extra_points=0)

    if agent_type == "dqn":
        # greedy agent: epsilon 0 and training=False disable any exploration
        agent = DeepQAgent(epsilon=0., minimum_epsilon=0., epsilon_decay=0., training=False)
    else:
        # training=False makes the actor play the mode of its masked softmax
        agent = ACAgentQuick(training=False)
    agent.load_model(weights_dir)
    print(f"Loaded {agent_type} weights from {weights_dir}")

    print(f"\nEvaluating vs ScriptedAIAgent ({num_games} games)")
    evaluate(game, [agent, ScriptedAIAgent()], num_games)

    print(f"\nEvaluating vs RandomAgent ({num_games} games)")
    evaluate(game, [agent, RandomAgent()], num_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_games", type=int, default=1000,
                        help="number of evaluation games against each opponent")
    parser.add_argument("--agent", choices=["dqn", "a2c"], default="dqn",
                        help="agent to evaluate (default: the submitted DQN)")
    parser.add_argument("--reward", choices=["standard", "own"], default="standard",
                        help="reward variant the agent was trained with "
                             "(selects the default weights directory)")
    parser.add_argument("--weights", type=str, default=None,
                        help="directory containing the saved agent weights "
                             "(overrides the --agent/--reward default)")
    args = parser.parse_args()
    weights = args.weights or WEIGHTS_DIRS[(args.agent, args.reward)]
    main(args.num_games, weights, args.agent)
