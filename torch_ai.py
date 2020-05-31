import argparse
from progress.bar import Bar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import deserialize_field, feature_vector

FEATURE_VEC_LEN = 16
LAYER_1 = 200
LAYER_2 = 80
LAYER_3 = 80
MODEL_PATH = "models/experimental/comb12_iter_0"

torch.set_num_threads(3)

parser = argparse.ArgumentParser(description='Automatically learn tetris')
parser.add_argument('--start_from', dest='model_start_path')
parser.add_argument('--dump', dest='model_dump_path', required=True)
parser.add_argument('--niter', dest='niter', type=int, default=1)
args = parser.parse_args()

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


def make_feature_vec(serialized_field):
    field = deserialize_field(serialized_field)
    return torch.tensor(feature_vector(field), dtype=torch.float)

if __name__=="__main__":
    X = []
    y = []
    with open("/tmp/scores", "r") as infile:
        for line in Bar("Reading input").iter(infile):
            serialized_field, score = line.split()
            X.append(make_feature_vec(serialized_field))
            y.append(torch.tensor([float(score)], dtype=torch.float))
    print("Finished reading")

    loss_function = nn.MSELoss()
    model = DeepBoy(num_features=FEATURE_VEC_LEN)
    if args.model_start_path:
        print("Using existing model")
        model.load_state_dict(torch.load(args.model_start_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
#    with torch.autograd.detect_anomaly():
    if True:
        model.train()
        for epoch in Bar("Training").iter(range(args.niter)):
            for feature_vec, target_score in zip(X, y):
#                print(feature_vec, target_score)
                optimizer.zero_grad()
                model_score = model(feature_vec)
                loss = loss_function(model_score, target_score)
                loss.backward()
                optimizer.step()
            print(" ", loss)
            losses.append(loss)
            torch.save(model.state_dict(), args.model_dump_path + "_iter_" + str(epoch))
    torch.save(model.state_dict(), args.model_dump_path)
    print(losses)
    print("Dumped to", args.model_dump_path)
