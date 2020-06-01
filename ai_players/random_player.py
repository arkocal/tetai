import random

from ai_players.base import AIPlayer

class RandomAIPlayer(AIPlayer):

    def score_field(self, field):
        return random.random()
