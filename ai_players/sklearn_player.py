import pickle

from sklearn.neural_network import MLPRegressor

from ai_players.base import AIPlayer
import ai_players.features as features

def feature_vector(field):
    return features.heights(field) + features.nr_holes(field)


class SklearnMLPAIPlayer(AIPlayer):

    def __init__(self, mechanics, model_path):
        super(SklearnMLPAIPlayer, self).__init__(mechanics)
        if model_path is not None:
            with open(model_path, "rb") as infile:
                self.mlp = pickle.load(infile)
        else:
            self.mlp = MLPRegressor(hidden_layer_sizes=(200, 80, 80),
                                    max_iter=20, verbose=True)

    def score_field(self, field):
        return self.mlp.predict([feature_vector(field)])[0]

    def train(self, training_data):
        X = [feature_vector(d[0]) for d in training_data]
        y = [float(d[1]) for d in training_data]
        self.mlp.partial_fit(X, y)

    def dump(self, path):
        with open(path, "wb") as outfile:
            pickle.dump(self.mlp, outfile)
