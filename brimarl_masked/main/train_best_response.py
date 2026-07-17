import argparse
import random

import numpy as np
import tensorflow as tf

from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
from brimarl_masked.environment.reward_variants import BriscolaGameOwnPoints
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.algorithms.dqn import QLearningAlgorithm
from brimarl_masked.scritps.training import *


class TrainingScriptedExploring(TrainingScripted):
    """TrainingScripted collects games with agent.clone(), whose default is
    training=False: the DeepQAgent would always act greedily and never
    explore. This subclass keeps epsilon-greedy exploration on during the
    data collection (2-player game only)."""

    def simulate_and_store(self):
        states, actions, masks, rewards, dones = play_episode(
            self.game, [self.agent.clone(training=True), ScriptedAIAgent()], train=True
        )
        self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])


def main(episodes=6000, evaluate_every=250, num_evaluation=500, reward="standard"):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game_class = BriscolaGame if reward == "standard" else BriscolaGameOwnPoints
    game = game_class(2, logger, win_extra_points=0)

    agent = DeepQAgent(
        epsilon=1.0,                            # start fully exploratory
        minimum_epsilon=0.1,
        epsilon_decay=0.9 / (0.8 * episodes),   # reach 0.1 at ~80% of training
    )

    training = TrainingScriptedExploring(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=QLearningAlgorithm(
            num_players=2,
            batch_size=128,
            discount=1.0,               # short episodic games, no need to discount
            replace_every=250,          # target net refresh, in training iterations
            num_learning_per_epoch=2,
            replay_memory_capacity=10000,   # ~500 games of transitions
            # loss_fn=tf.keras.losses.Huber(delta=10.),  # alternative to the MSE default
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        save_dir="models_savings/2/DeepQAgent" + ("" if reward == "standard" else "_own"),
    )
    training.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["standard", "own"], default="standard",
                        help="standard: trick points, +/- winner/loser; own: winner +points, loser 0")
    args = parser.parse_args()
    main(reward=args.reward)
