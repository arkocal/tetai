import itertools
import os

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

def deserialize_field(serialized_field, field_height=10):
    field = []
    for i in range(0, len(serialized_field), field_height):
        serialized_column = serialized_field[i:i+field_height]
        field.append([j=="1" for j in serialized_column])
