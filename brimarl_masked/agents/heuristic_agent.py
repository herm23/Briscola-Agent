import brimarl_masked.environment.environment as brisc
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer


class HeuristicAgent(brisc.Agent):
    """Rule-based baseline agent (no learning). Designed for the 2-player game.

    It improves on ScriptedAIAgent with two ingredients it lacks:
    1. card counting: it remembers every card that has appeared on the table,
       so at any time it knows which briscole and carichi are still unseen;
    2. expected-value decisions: capture thresholds scale with the value of
       the briscola being spent, and leads use worst-case reasoning on the
       cards the opponent may still hold (a lead is "safe" only if no unseen
       card can beat it).
    """

    def __init__(self):
        super().__init__("HeuristicAgent")
        self.name = "HeuristicAgent"
        self.seen_ids = set()

    def state(self, game: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        """Accumulate the ids of all cards that have touched the table."""
        for card in game.played_cards:
            self.seen_ids.add(card.id)
        return []

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        hand = player.hand
        briscola_seed = game.briscola.seed
        table = game.played_cards
        points_on_table = sum(card.points for card in table)

        if table:
            # --- playing second ---
            winning = [
                (i, card) for i, card in enumerate(hand)
                if all(brisc.scoring(briscola_seed, played, card) for played in table)
            ]
            if winning:
                # if capturing closes the game (> 60 points), always do it,
                # trying the cheapest sufficient card first
                for i, card in sorted(
                    winning, key=lambda ic: (ic[1].seed == briscola_seed, ic[1].strength)
                ):
                    if player.points + points_on_table + card.points > 60:
                        return i, None, None

                non_briscola_winners = [
                    (i, card) for i, card in winning if card.seed != briscola_seed
                ]
                if non_briscola_winners:
                    # capturing without briscola is free: secure the most points
                    i, _ = max(non_briscola_winners, key=lambda ic: ic[1].points)
                    return i, None, None

                # capture is only possible with a briscola: worth it only if the
                # table pays for the value of the briscola spent
                i, cheapest = min(winning, key=lambda ic: ic[1].strength)
                if points_on_table >= self._capture_threshold(cheapest):
                    return i, None, None

            # cannot or should not win the trick: give away as little as possible
            return self._safest_discard(hand, briscola_seed), None, None

        # --- playing first ---
        # bank points: lead the most valuable card the opponent cannot beat
        safe_cards = [card for card in hand if self._is_safe_lead(game, hand, card)]
        if safe_cards:
            best = max(safe_cards, key=lambda card: card.points)
            if best.points >= 3:
                return hand.index(best), None, None

        # otherwise lead the cheapest card (lisci first, briscole preserved)
        return self._safest_discard(hand, briscola_seed), None, None

    def _unseen_cards(self, game: BriscolaGame, hand):
        """Cards that may still be in the opponent's hand (or in the deck):
        everything except what was seen on the table, my own hand and the
        revealed briscola while it still sits under the deck."""
        hand_ids = {card.id for card in hand}
        under_deck = game.deck.briscola
        return [
            card for card in game.deck.deck
            if card.id not in self.seen_ids
            and card.id not in hand_ids
            and not (under_deck is not None and card.id == under_deck.id)
        ]

    def _is_safe_lead(self, game: BriscolaGame, hand, card):
        """True if no card the opponent may hold beats this card when led."""
        return not any(
            brisc.scoring(game.briscola.seed, card, other)
            for other in self._unseen_cards(game, hand)
        )

    @staticmethod
    def _capture_threshold(briscola_card):
        """Minimum points on the table that justify spending this briscola."""
        if briscola_card.points == 0:
            return 3   # liscio di briscola (2, 4, 5, 6, 7)
        if briscola_card.points <= 4:
            return 6   # fante, cavallo, re di briscola
        return 10      # asso o tre di briscola

    @staticmethod
    def _safest_discard(hand, briscola_seed):
        """Index of the card that gives away the least: fewest points first,
        then prefer non-briscola, then lowest strength."""
        return min(
            range(len(hand)),
            key=lambda i: (
                hand[i].points,
                hand[i].seed == briscola_seed,
                hand[i].strength,
            ),
        )

    def update(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass

    def reset(self):
        self.seen_ids = set()

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass

    def clone(self, training=False):
        return HeuristicAgent()
