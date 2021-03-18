import random
import time
import multiprocessing

from ai_players import TorchAIPlayer, RuleBasedAIPlayer
import utils
from mechanics import Mechanics

nes_tetris = Mechanics()

EPOCHS = 1_000_000
NR_MOVES = 5
MIN_SCORE = -100_000_000_000_000
GAME_OVER_PENALTY = 0

class GameOver(Exception):
    pass

def play(ai_player, field, nr_moves):
    for nr_pieces, piece in enumerate(ai_player.mechanics.piece_types):
        if nr_pieces == nr_moves:
            return field, ai_player.score_field(field)
        if not ai_player.mechanics.can_place_piece(field, piece, ai_player.mechanics.start_placement):
            raise GameOver
        placement, _ = ai_player.choose_placement(field, piece)
        field = ai_player.mechanics.place_piece(field, piece, placement)
    return field, ai_player.score_field(field)


def max_height(field):
    max_height = 0
    for x in range(len(field)):
        for y in range(len(field[0])):
            if field[x][y]:
                max_height = max(max_height, y)
    return max_height

def fg_from_file(path):
    with open(path) as field_file:
        real_fields = [line.split()[0] for line in field_file]
    def fg(): #EXISTING
        return utils.deserialize_field(random.choice(real_fields))
    return fg

# RANDOM
def field_generator_0():
    field = "".join(["".join([random.choice(["0", "1"]) for _ in range(10)])+"0"*12 for _ in range(10)])
    return utils.deserialize_field(field)


field_generator_1 = fg_from_file("fields/fields")
# RANDOM HIGH
def field_generator_2():
    field = "".join(["".join([random.choice(["0", "1"]) for _ in range(14)])+"0"*8 for _ in range(10)])
    return utils.deserialize_field(field)

# RANDOM LOW
def field_generator_4():
    field = "".join(["".join([random.choice(["0", "1"]) for _ in range(4)])+"0"*18 for _ in range(10)])
    return utils.deserialize_field(field)

field_generator_3 = fg_from_file("fields/torch_fields")
field_generator_5 = fg_from_file("scores/scores_iter_2")

generators = [field_generator_0, field_generator_1, field_generator_2, field_generator_3]
N = len(generators)

def play_wrap(t):
    player, x, nr_moves = t
    try:
        field, score = play(player, x, nr_moves)
        return (player, score, field, max_height(field))
    except GameOver:
        return None

rule_based_agents = [RuleBasedAIPlayer(nes_tetris)]
gen_agent_pairs = [(gen, TorchAIPlayer(nes_tetris, f"models/agent_{i}")) for i, gen in enumerate(generators)]
agents = [a for g, a in gen_agent_pairs]
for i, agent in enumerate(agents + rule_based_agents):
    agent.tag = i
print(gen_agent_pairs)
print(agents)
worst_score = 0
normalization_factor = 0
inconsistence = 0
epoch_start_time = time.time()
pool = multiprocessing.Pool()
for i in range(EPOCHS):
    if i and i%100==0:
        print("EPOCH:", i, "inconsistence:", inconsistence, "time:", time.time()-epoch_start_time)
        epoch_start_time = time.time()
        for j, agent in enumerate(agents):
            agent.dump(f"models/agent_{j}")
        inconsistence = 0
    generator, agent = random.choice(gen_agent_pairs)
    x = generator()
    initial_score = agent.score_field(x)
#    best_player, best_score = None, MIN_SCORE
    scores = []
    own_score = None
    scores = [s for s in pool.map(play_wrap, [(player, x, NR_MOVES) for player in agents+rule_based_agents])
              if s is not None]
    for player, score, field, height in scores:
        if agent.tag==player.tag:
            own_score = score
#    for player in agents + rule_based_agents:
#        try:
#            field, score = play(player, x, NR_MOVES)
#            scores.append((score, field, max_height(field)))
#            if agent==player:
#                own_score = score
#        except GameOver:
#            continue
    if not scores: # All game over, nothing to do
        print("All failed")
        continue
    bplayer, bscore, bfield, bheight = max(scores, key=lambda x:x[2])
    new_score = agent.score_field(bfield)
#    current_factor = initial_score / bscore
#    normalization_factor = (normalization_factor*i+current_factor)/(i+1)
#    normalized = bscore*normalization_factor
    if own_score is None or new_score > own_score:
        agent.train([(x, new_score)])
    else:
        inconsistence += 1
