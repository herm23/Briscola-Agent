import random

from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.environment.environment import BriscolaGame
from brimarl_masked.environment.utils import BriscolaLogger

if __name__ == "__main__":
    # Smoke test: RandomAgent vs ScriptedAIAgent, no learning involved.
    # Verifies that the environment runs end to end.
    random.seed(0)

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)
    agents = [RandomAgent(), ScriptedAIAgent()]

    num_games = 1000
    wins = [0, 0]
    for _ in range(num_games):
        winner_id, winner_points = play_episode(game, agents, train=False)
        wins[winner_id] += 1

    print(f"Games played: {num_games}")
    for agent, num_wins in zip(agents, wins):
        print(f"{agent.name}: {num_wins} wins ({100 * num_wins / num_games:.1f}%)")
