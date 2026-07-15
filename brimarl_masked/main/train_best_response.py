from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.dqn import QLearningAlgorithm
from brimarl_masked.scritps.training import *


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    # for DQN agents, replace the appropriate code with the corresponding classes for the actor critic
    agent = DeepQAgent(...)

    episodes = ...
    evaluate_every = ...
    num_evaluation = ...
    training = TrainingScripted(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=QLearningAlgorithm(
            ...
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
    )
    training.train()


if __name__ == "__main__":
    main()