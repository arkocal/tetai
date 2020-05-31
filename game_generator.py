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
    future_score_lists.append(score_future(ai_player, field,
                                            args.nr_moves, args.nr_replays))
new_scores = aggregate_future_scores(future_score_lists)
dump_scores(new_scores, args.score_dump_path)
