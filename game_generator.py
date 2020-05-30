"""Example call:
shuf -n NR_GAMES | python game_generator.py --model my_pickled_model.
"""
import argparse
import pickle

from common import score_field_with_features, make_move_with_model, piece_stream
from common import serialize_field, deserialize_field
from rules import is_valid, STANDARD_RULESET, place_piece

import numpy as np
from progress.bar import Bar

#from utils import dump_field_rating, piece_stream, dump_feature_rating
#import utils
#from rules import STANDARD_RULESET, is_valid, place_piece
#from moves import make_move_with
class GameOverScore:
    worst_score = 0

    def __float__(self):
        return GameOverScore.worst_score

def load_model(path):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)

def read_fields_from_stdin(nr_of_fields):
    fields = []
    for _ in Bar("Reading fields").iter(range(nr_of_fields)):
        fields.append(deserialize_field(input()))
    return fields

def dump_scores(new_scores, dump_path):
    with open(dump_path, "a") as dump_file:
        for field, score in new_scores:
            dump_file.write(f"{serialize_field(field)} {score}\n")

def score_future(ai_player, input_field, nr_moves, nr_replay):
    current_score = score_field_with_features(ai_player, input_field)
    new_scores = []
    for game_round in range(nr_replay):
        field = input_field
        pieces = piece_stream()
        for nr_pieces, piece in enumerate(pieces):
            if nr_pieces == nr_moves:
                new_scores.append(score_field_with_features(ai_player, field))
                break
            if not is_valid(field, piece, STANDARD_RULESET.start_config, STANDARD_RULESET):
                new_scores.append(GameOverScore())
                break
            config, moves = make_move_with_model(ai_player, field, piece)
            field = place_piece(field, piece, config, STANDARD_RULESET)
    return(input_field, current_score, new_scores)


def aggregate_future_scores(future_score_lists):
    for field, old_score, new_scores in future_score_lists:
        GameOverScore.worst_score = min(min(float(s) for s in new_scores), GameOverScore.worst_score)
    new_scores_aggregated = [sum(float(s) for s in ns)/len(ns) for _, _, ns in future_score_lists]
    normalization_factor = (sum(os for _, os, _ in future_score_lists) /
                            sum(new_scores_aggregated))
    return [(field, new_score*normalization_factor) for (field, _, _), new_score in
            zip(future_score_lists, new_scores_aggregated)]


parser = argparse.ArgumentParser(description='Generate rated games by playing')
parser.add_argument('--model', dest='model_path', required=True,
                    help="Pickled model to play with.")
# TODO date should be dumped into a database to avoid redundancy
parser.add_argument('--dump', dest='score_dump_path', required=True,
                    help="File path to dump (field, score) into")
parser.add_argument('--nr_moves', dest='nr_moves', default=5,
                    help="How many moves should be played to create new score.")
parser.add_argument('--nr_replays', dest='nr_replays', default=5,
                    help="How often should a field be replayed to create new score.")
parser.add_argument('--nr_fields', dest='nr_fields', default=2, type=int,
                    help="Number of fields to rescore.")

args = parser.parse_args()

ai_player = load_model(args.model_path)
fields = read_fields_from_stdin(args.nr_fields)

future_score_lists = []
for field in Bar("Playing").iter(fields):
    future_score_lists .append(score_future(ai_player, field,
                                            args.nr_moves, args.nr_replays))
new_scores = aggregate_future_scores(future_score_lists)
dump_scores(new_scores, args.score_dump_path)
  #




"""ruleset = STANDARD_RULESET

class GameOver: # Used as mathematical type
    pass


def rate_field_features_mlp(mlp, field, ruleset):
    features = utils.heights(field) + utils.nr_holes(field)
    return mlp.predict([features])[0]

def rate_future(mlp, field, ruleset, nr_moves, pieces=None):
    def rate_function(field, ruleset):
        return rate_field_features_mlp(mlp, field, ruleset)

    if pieces is None:
        pieces = piece_stream(ruleset)
    for nr_pieces, piece in enumerate(pieces):
        if nr_pieces == nr_moves:
            break
        if not is_valid(field, piece, ruleset.start_config, ruleset):
            return GameOver
        config, moves = make_move_with(field, piece, ruleset, rate_function=rate_function)
        field = place_piece(field, piece, config, ruleset)
    return rate_field_features_mlp(mlp, field, ruleset)

def create_intuition_ratings(mlp, fields, ruleset, nr_moves):
    #ir = [(field,
    #       rate_field_features_mlp(mlp, field, ruleset),
    #       rate_future(mlp, field, ruleset, nr_moves)) for field in fields]
    ir = []
    for field in Bar("Playing").iter(fields):
        ir.append((field,
                   rate_field_features_mlp(mlp, field, ruleset),
                   rate_future(mlp, field, ruleset, nr_moves)))
    old_scores = [os for _, os, ns in ir]
    new_scores = [ns for _, os, ns in ir]
    worst_new_score = min(n for n in new_scores if isinstance(n, float)) # This could be problematic
    sum_new_scores = 0
    for ns in new_scores:
        if not isinstance(ns, float):
            ns = worst_new_score
        sum_new_scores += ns
    normalization_factor = sum(old_scores) / sum_new_scores
    return [(field, (new_score * normalization_factor if isinstance(new_score, float) else worst_new_score))
             for field, _, new_score in ir]


mlp_path = "mlp_feature_intuition_net1"
mlp = pickle.load(open(mlp_path, "rb"))
with open("headr", "r") as infile:
    flat_fields = []
    for line in infile:
        flat_fields.append([int(i) for i in line.split()[:-1]])
    fields = [np.array(field_as_array, dtype=bool).reshape(10,22) for
              field_as_array in flat_fields]
    print("Read everything")
    for field, rating in create_intuition_ratings(mlp, fields, ruleset, 5):
        utils.dump_field_rating(field, rating, "rating_intuition_n5_1")
"""
