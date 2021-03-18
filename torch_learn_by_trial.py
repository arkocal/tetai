#
# Step 1. pick a field
# Step 2. pick 2 random moves
# Step 3. rate moves
# Step 4. play for NR_MOVES, re-evaluate
# Step 5. train by swapping ORIGINAL EVALUATIONS if worse > better

import random
import time

from ai_players import TorchAIPlayer
import utils
from mechanics import Mechanics

nes_tetris = Mechanics()

NR_MOVES = 10
EPOCHS = 1_000_000
MIN_SCORE = -10*10 # should suffice
GAMMA = 0.01
ALPHA = 0.01
class GameOver(Exception):
    pass


def max_height(field):
    max_height = 0
    for x in range(len(field)):
        for y in range(len(field[0])):
            if field[x][y]:
                max_height = max(max_height, y)
    return max_height

def play(ai_player, field, nr_moves):
    for nr_pieces, piece in enumerate(ai_player.mechanics.piece_types):
        if nr_pieces == nr_moves:
            return field, ai_player.score_field(field)
        if not ai_player.mechanics.can_place_piece(field, piece, ai_player.mechanics.start_placement):
            raise GameOver
        placement, _ = ai_player.choose_placement(field, piece)
        field = ai_player.mechanics.place_piece(field, piece, placement)
    return field, ai_player.score_field(field)

def fg_from_file(path):
    with open(path) as field_file:
        real_fields = [line.split()[0] for line in field_file]
    def fg(): #EXISTING
        return utils.deserialize_field(random.choice(real_fields))
    return fg

field_generator_0 = fg_from_file("fields/fields")
ai_player = TorchAIPlayer(nes_tetris)
change_mind = 0
epoch_start = time.time()
for i in range(EPOCHS):
    if i and i%100 == 0:
        print(i, change_mind, time.time()-epoch_start)
        epoch_start = time.time()
        change_mind = 0
        ai_player.dump("models/experimental/trial")
    field = field_generator_0()
    piece = random.choice(nes_tetris.piece_types)
    placements = nes_tetris.get_valid_end_placements(field, piece)
    if not placements:
        continue
    p1, _ = random.choice(placements)
    p2, _ = random.choice(placements)
    field_1 = nes_tetris.place_piece(field, piece, p1)
    field_2 = nes_tetris.place_piece(field, piece, p2)
    score_1 = ai_player.score_field(field_1)
    score_2 = ai_player.score_field(field_2)
    nr_moves = random.randint(5, 15)
    try:
        future_field_1, future_score_1 = play(ai_player, field_1, nr_moves)
        height_1 = max_height(future_field_1)
    except GameOver:
        future_score_1 = MIN_SCORE
        height_1 = 25
    try:
        future_field_2, future_score_2 = play(ai_player, field_2, nr_moves)
        height_2 = max_height(future_field_2)
    except GameOver:
        future_score_2 = MIN_SCORE
        height_2 = 25

    if (score_1 > score_2 and height_1 > height_2):
        change_mind += 1
        score_1_new = (score_1 + GAMMA*score_2)/(1+GAMMA)
        score_2_new = (score_2 + GAMMA*score_1)/(1+GAMMA)
    elif (score_2 > score_1 and height_2 > height_1):
        change_mind += 1
        score_1_new = (score_1 + GAMMA*score_2)/(1+GAMMA)
        score_2_new = (score_2 + GAMMA*score_1)/(1+GAMMA)
    elif score_1 > score_2:
        diff = score_1 - score_2
        score_1_new = score_1 + ALPHA*diff
        score_2_new = score_2 - ALPHA*diff
    elif score_2 > score_1:
        diff = score_2 - score_1
        score_1_new = score_1 - ALPHA*diff
        score_2_new = score_2 + ALPHA*diff
    ai_player.train([(field_1, score_1_new), (field_2, score_2_new)])


ai_player.dump("models/experimental/trial")
