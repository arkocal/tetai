import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progress.bar import Bar

from ai_players.base import AIPlayer
import ai_players.features as features

FEATURE_VEC_LEN = 16
LAYER_1 = 200
LAYER_2 = 80
LAYER_3 = 80

class DeepBoy(nn.Module):

    def __init__(self, num_features):
        super(DeepBoy, self).__init__()
        self.linear0 = nn.Linear(num_features, LAYER_1)
        self.linear1 = nn.Linear(LAYER_1, LAYER_2)
        self.linear2 = nn.Linear(LAYER_2, LAYER_3)
        self.linear3 = nn.Linear(LAYER_3, 1)

    def forward(self, feature_vec):
        out = F.relu(self.linear0(feature_vec))
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out

def make_feature_vec(field):
    return torch.tensor(features.heights(field)+features.nr_holes(field),
                        dtype=torch.float)

class TorchAIPlayer(AIPlayer):

    def __init__(self, mechanics, model_state_dict_path=None):
        super(TorchAIPlayer, self).__init__(mechanics)
        self.model = DeepBoy(num_features=16)
        if model_state_dict_path is not None:
            self.model.load_state_dict(torch.load(model_state_dict_path))

    def score_field(self, field):
        with torch.no_grad():
            model_score = self.model(make_feature_vec(field))
            return model_score.item()

    def train(self, training_data):
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        losses = []
        self.model.train()
        X = [make_feature_vec(d[0]) for d in training_data]
        y = [torch.tensor([float(d[1])], dtype=torch.float) for d in training_data]
        for feature_vec, target_score in Bar("Training batch", max=len(X)).iter(zip(X, y)):
            optimizer.zero_grad()
            model_score = self.model(feature_vec)
            loss = loss_function(model_score, target_score)
            loss.backward()
            optimizer.step()

    def dump(self, path):
        torch.save(self.model.state_dict(), path)
