import argparse
import random

import numpy as np
import tensorflow as tf

from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
from brimarl_masked.environment.reward_variants import BriscolaGameOwnPoints
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.a2c import A2CAlgorithm
from brimarl_masked.scritps.training import *


def main(episodes=6000, evaluate_every=250, num_evaluation=500, reward="standard"):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game_class = BriscolaGame if reward == "standard" else BriscolaGameOwnPoints
    game = game_class(2, logger, win_extra_points=0)

    # no epsilon here: the actor samples from its own (masked) softmax, so
    # exploration is intrinsic to the policy and annealed by learning itself
    agent = ACAgentQuick(training=True)

    training = TrainingScripted(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=A2CAlgorithm(
            num_players=2,
            discount=1.0,               # short episodic games, no need to discount
            num_learning_per_epoch=1,   # on-policy: one pass over the fresh batch
            min_samples=128,            # learn every ~6-7 games of transitions
            entropy_beta=0.01,          # mild exploration bonus on the policy
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        save_dir="models_savings/2/ACAgentQuick" + ("" if reward == "standard" else "_own"),
    )
    training.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["standard", "own"], default="standard",
                        help="standard: trick points, +/- winner/loser; own: winner +points, loser 0")
    args = parser.parse_args()
    main(reward=args.reward)
