import pickle

from ai_players.base import AIPlayer
import ai_players.features as features

def feature_vector(field):
    return features.heights(field) + features.nr_holes(field)


class SklearnMLPAIPlayer(AIPlayer):

    def __init__(self, mechanics, model_path):
        super(SklearnMLPAIPlayer, self).__init__(mechanics)
        with open(model_path, "rb") as infile:
            self.mlp = pickle.load(infile)

    def score_field(self, field):
        return self.mlp.predict([feature_vector(field)])[0]
