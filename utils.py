import itertools
import os

from ai_players import RandomAIPlayer, SklearnMLPAIPlayer, TorchAIPlayer

def print_field(field, active_cos=None, clear=False):
    if clear:
        os.system("clear")
    if active_cos == None:
        active_cos = []
    width = len(field)
    height = len(field[0])
    for y in range(height-1, -1, -1):
        for x in range(width):
            if (x,y) in active_cos:
                print("X", end="")
            elif field[x][y]:
                print("█", end="")
            else:
                print("_", end="")
        print()


def serialize_field(field):
    return "".join("1" if i else "0" for i in itertools.chain.from_iterable(field))

def deserialize_field(serialized_field, field_height=22):
    field = []
    for i in range(0, len(serialized_field), field_height):
        serialized_column = serialized_field[i:i+field_height]
        field.append([j=="1" for j in serialized_column])
    return field

def get_ai_player(name, mechanics, data=None):
    if name == "random":
        ai_player = RandomAIPlayer(mechanics)
    elif name == "sklearn_mlp":
        ai_player = SklearnMLPAIPlayer(mechanics, data)
    elif name == "torch":
        ai_player = TorchAIPlayer(mechanics, data)
    return ai_player

dump_cache = []
CACHE_SIZE = 500
def dump_field(field, path, score=None):
    global dump_cache
    score_str = "" if score is None else str(score)
    dump_cache.append(serialize_field(field) + " " + score_str)
    if len(dump_cache) >= CACHE_SIZE:
        with open(path, "a") as outfile:
            for line in dump_cache:
                outfile.write(line + "\n")
        dump_cache = []
