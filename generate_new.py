import argparse
import random

from progress.bar import Bar

from mechanics import Mechanics
import utils

parser = argparse.ArgumentParser(description='Rescore fields')
parser.add_argument('--ai', dest='ai', choices=["random", "sklearn_mlp", "torch"],
                    required=True,
                    help='AI player that should play')
parser.add_argument('--ai-data', dest='ai_data',
                    help='Saved data for ai, such as trained models')
parser.add_argument('--score', dest='score', choices=[None, "future", "neighbor"],
                    help='Rescoring strategy for games.')
parser.add_argument('--future-len', dest='future_len', type=int, default=5,
                    help='How many moves to look for in the future, only active if score is "future"')
parser.add_argument('--future-replay', dest='future_replay', type=int, default=5,
                    help='How many random games in the future, only active if score is "future"')
parser.add_argument('--dump', dest="dump_path", required=True)
parser.add_argument('--fields', dest="fields", required=True,
                    help="File containing list of start fields for rescoring")
parser.add_argument('--nr-fields', dest="nr_fields", type=int, default=10_000,
                    help="How many fields should be rescored")
#parser.add_argument('--nr-games', dest="nr_games", default=-1,
#                    help="Number of games to play, -1 means indefinetly.")
args = parser.parse_args()
tetris = Mechanics()
ai_player = utils.get_ai_player(args.ai, tetris, args.ai_data)

class GameOverScore:
    worst_score = 0

    def __float__(self):
        return GameOverScore.worst_score

def score_future(ai_player, input_field, nr_moves, nr_replay):
    current_score = ai_player.score_field(input_field)
    new_scores = []
    for game_round in range(nr_replay):
        field = input_field.copy()
        pieces = ai_player.mechanics.piece_stream()
        for nr_pieces, piece in enumerate(pieces):
            if nr_pieces == nr_moves:
                new_scores.append(ai_player.score_field(field))
                break
            if not ai_player.mechanics.can_place_piece(field, piece, ai_player.mechanics.start_placement):
                new_scores.append(GameOverScore())
                break
            placement, _ = ai_player.choose_placement(field, piece)
            field = ai_player.mechanics.place_piece(field, piece, placement)
    return(input_field, current_score, new_scores)

def aggregate_future_scores(future_score_lists):
    for field, old_score, new_scores in future_score_lists:
        GameOverScore.worst_score = min(min(float(s) for s in new_scores), GameOverScore.worst_score)
    new_scores_aggregated = [sum(float(s) for s in ns)/len(ns) for _, _, ns in future_score_lists]
    normalization_factor = (sum(os for _, os, _ in future_score_lists) /
                            sum(new_scores_aggregated))
    return [(field, new_score*normalization_factor) for (field, _, _), new_score in
            zip(future_score_lists, new_scores_aggregated)]

def score_neighbors(ai_player, input_field):
    current_score = ai_player.score_field(input_field)
    new_scores = []
    for piece in ai_player.mechanics.piece_types:
        if not ai_player.mechanics.can_place_piece(input_field, piece, ai_player.mechanics.start_placement):
            new_scores.append(GameOverScore())
        else:
            placement, _ = ai_player.choose_placement(input_field, piece)
            field = ai_player.mechanics.place_piece(input_field, piece, placement)
            new_scores.append(ai_player.score_field(field))
    return(input_field, current_score, new_scores)


with open(args.fields, "r") as infile:
    fields = [utils.deserialize_field(line.split()[0]) for line in infile]
    fields = random.sample(fields, args.nr_fields)
    print("Loaded fields")

if args.score in ["future", "neighbor"]:
    score_lists = []
    for field in Bar("Playing", suffix='%(index)d/%(max)d %(eta)ds').iter(fields):
        if args.score == "future":
            score_lists.append(score_future(ai_player, field,
                                                   args.future_len, args.future_replay))
        elif args.score == "neighbor":
            score_lists.append(score_neighbors(ai_player, field))
    new_scores = aggregate_future_scores(score_lists)



with open(args.dump_path, "a") as dump_file:
    for field, score in new_scores:
        dump_file.write(f"{utils.serialize_field(field)} {score}\n")
