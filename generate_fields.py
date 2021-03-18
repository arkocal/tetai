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
parser.add_argument('--dump', dest="dump_path", required=True)
parser.add_argument('--nr-fields', dest="nr_fields", type=int, default=10_000,
                    help="How many fields should be created")

args = parser.parse_args()
tetris = Mechanics()
ai_player = utils.get_ai_player(args.ai, tetris, args.ai_data)
field = tetris.get_empty_field()
for nr_pieces, piece in Bar("Playing", max=args.nr_fields, suffix='%(index)d/%(max)d %(eta)ds').iter(enumerate(tetris.piece_stream())):
    if nr_pieces == args.nr_fields:
        break
    if not tetris.can_place_piece(field, piece, tetris.start_placement):
        field = tetris.get_empty_field()
    placement, control_action_path = ai_player.choose_placement(field, piece)
    field = tetris.place_piece(field, piece, placement)
    utils.dump_field(field, args.dump_path)
