"""Reward-function variants of BriscolaGame (bonus experiments).

The provided environment is left untouched: variants subclass BriscolaGame
and override get_rewards_from_step only.
"""
from brimarl_masked.environment.environment import BriscolaGame


class BriscolaGameOwnPoints(BriscolaGame):
    """Reward = points captured by the trick winner; the loser gets 0.

    With discount 1 the return of a game is exactly the agent's final score
    (0..120), instead of the point differential (-120..120) of the standard
    reward: the agent is trained to maximize its own points, with no explicit
    incentive to deny points to the opponent.
    """

    def get_rewards_from_step(self):
        winner_player_id, points = self.evaluate_step()

        rewards = {}
        for player_id in self.get_players_order():
            is_winner = player_id in [winner_player_id, self.get_teammate(winner_player_id)]
            rewards[player_id] = points if is_winner else 0
        return rewards
