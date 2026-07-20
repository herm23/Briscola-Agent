import time
from statistics import mean

from tqdm.auto import tqdm

from brimarl_masked.environment.emulate import play_episode


def evaluate(game, agents, num_evaluations):
    """Play num_evaluations games and report statistics."""
    total_wins = [0] * len(agents)
    points_history = [[0] for _ in range(len(agents))]
    print()
    time.sleep(0.1)
    for _ in tqdm(range(num_evaluations), desc="\tEval", position=0, smoothing=0.):
        game_winner_id, winner_points = play_episode(game, agents, train=False)
        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id or player.id == game.get_teammate(game_winner_id):
                total_wins[player.id] += 1

    print(f"\tTotal wins: {total_wins}.")
    for i in range(len(agents)):
        print(f"\t{agents[i].name} {i} won {total_wins[i]/num_evaluations:.2%} with an average of {mean(points_history[i]):.2f} points.")

    return total_wins, points_history
