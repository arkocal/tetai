import argparse
import time

from ai_players import RandomAIPlayer, SklearnMLPAIPlayer, TorchAIPlayer
from mechanics import Mechanics
import utils

parser = argparse.ArgumentParser(description='Show a tetris game')
parser.add_argument('--ai', dest='ai', choices=["random", "sklearn_mlp", "torch"],
                    required=True,
                    help='AI player that should play')
parser.add_argument('--move-time', dest='move_time', type=int,
                    default=500,
                    help='Move time in ms, negative value for prompt')
parser.add_argument('--ai-data', dest='ai_data',
                    help='Saved data for ai, such as trained models')

args = parser.parse_args()

tetris = Mechanics()
if args.ai == "random":
    ai_player = RandomAIPlayer(tetris)
elif args.ai == "sklearn_mlp":
    ai_player = SklearnMLPAIPlayer(tetris, args.ai_data)
elif args.ai == "torch":
    ai_player = TorchAIPlayer(tetris, args.ai_data)

field = tetris.get_empty_field()

for piece in tetris.piece_stream():
    utils.print_field(field, clear=True)
    if not tetris.can_place_piece(field, piece, tetris.start_placement):
        print("Game over")
        exit()
    placement, control_action_path = ai_player.choose_placement(field, piece)
    field = tetris.place_piece(field, piece, placement)
    if args.move_time > 0:
        time.sleep(args.move_time / 1000)
    else:
        input()
