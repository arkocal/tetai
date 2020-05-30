import argparse
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.utils.random import sample_without_replacement

#from progress.bar import Bar

from common import deserialize_field, feature_vector
#from rules import RuleSet, place_piece, Piece, ROTATIONS, Config, STANDARD_RULESET
#import utils


parser = argparse.ArgumentParser(description='Automatically learn tetris')

parser.add_argument('--start_from', dest='mlp_start_path')
parser.add_argument('--dump', dest='mlp_dump_path', required=True)
parser.add_argument('--scores', dest='score_path', required=True)
parser.add_argument('--niter', dest='niter', type=int)
args = parser.parse_args()

if args.mlp_start_path:
    mlp = pickle.load(open(args.mlp_start_path, "rb"))
    print("Starting from old net")
else:
    print("Creating new net")
#    mlp = MLPRegressor(hidden_layer_sizes=(200, 80, 80), max_iter=20, verbose=True)
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=20, verbose=True)

X = []
y = []

with open(args.score_path, "r") as infile:
    for line in infile:
        serialized_field, rating = line.split()
        field = deserialize_field(serialized_field)
        features = feature_vector(field)
        X.append(features)
        y.append(float(rating))

    print("Starting fit")
    if args.niter:
        for _ in range(args.niter):
            mlp.partial_fit(X, y)
    else:
         mlp.fit(X, y)

with open(args.mlp_dump_path, "wb") as ofile:
    pickle.dump(mlp, ofile)

# 394 -> deep boy 1
