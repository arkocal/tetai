import argparse
import time

from mechanics import Mechanics
import utils

parser = argparse.ArgumentParser(description='Show a tetris game')
parser.add_argument('--ai', dest='ai', choices=["random", "sklearn_mlp", "torch"],
                    required=True,
                    help='AI player that should play')
parser.add_argument('--move-time', dest='move_time', type=int,
                    default=500,
                    help='Move time in ms, negative value for prompt')
parser.add_argument('--show', dest='show', choices=["field", "avg_pieces"],
                    default="field",
                    help='Show end results of the field')
parser.add_argument('--ai-data', dest='ai_data',
                    help='Saved data for ai, such as trained models')
args = parser.parse_args()

tetris = Mechanics()
ai_player = utils.get_ai_player(args.ai, tetris, args.ai_data)
field = tetris.get_empty_field()
total_games = 0
for nr_pieces, piece in enumerate(tetris.piece_stream()):
    if args.show == "field":
        if args.move_time > 0:
            time.sleep(args.move_time / 1000)
        else:
            input()
        utils.print_field(field, clear=True)
        if not tetris.can_place_piece(field, piece, tetris.start_placement):
            print("Game over")
            exit()
    elif args.show == "avg_pieces":
        if not tetris.can_place_piece(field, piece, tetris.start_placement):
            field = tetris.get_empty_field()
            total_games += 1
            print(nr_pieces/total_games, "in", total_games)
    placement, control_action_path = ai_player.choose_placement(field, piece)
    field = tetris.place_piece(field, piece, placement)
