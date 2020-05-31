from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import itertools

class Piece(Enum):
    I = 1
    O = 2
    L = 3
    J = 4
    S = 5
    Z = 6
    T = 7

Config = namedtuple("Config", ["x", "y", "rotation"])
RuleSet = namedtuple("RuleSet", ["pieces", "field_width", "field_height",
                                 "start_config", "rotations"])

class ControlAction(Enum):
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    ROTATION = 4

START_CONFIG = Config(5, 19, 0)
ROTATIONS = {Piece.I: [[(-2, 0), (-1, 0), (0, 0), (1, 0)],
                       [(0, 2), (0, 1), (0, 0), (0, -1)]],
             Piece.O: [[(0, 0), (0, -1), (-1,0), (-1,-1)]],
             Piece.L: [[(-1, -1), (-1, 0), (0, 0), (1, 0)],
                       [(-1, 1), (0,1), (0, 0), (0, -1)],
                       [(-1, 0), (0, 0), (1, 0), (1, 1)],
                       [(0,1), (0, 0), (0, -1), (1, -1)]],
             Piece.J: [[(-1, 0), (0, 0), (1, 0), (1, -1)],
                       [(-1, -1), (0, -1), (0, 0), (0,1)],
                       [(-1, 1), (-1, 0), (0, 0), (1, 0)],
                       [(0, -1), (0, 0), (0, 1), (1, 1)]],
             Piece.S: [[(-1, -1), (0, -1), (0, 0), (1, 0)],
                       [(0, 1), (0, 0), (1, 0), (1, -1)]],
             Piece.Z: [[(-1, 0), (0, 0), (0, -1), (1, -1)],
                       [(0, -1), (0, 0), (1, 0), (1, 1)]],
             Piece.T: [[(0, 0), (-1, 0), (1, 0), (0, -1)],
                       [(0, 0), (-1, 0), (0, -1), (0,1)],
                       [(0, 0), (-1, 0), (1, 0), (0,1)],
                       [(0, 0), (1, 0), (0, -1), (0,1)]]
             }

STANDARD_RULESET = RuleSet(pieces=tuple(p for p in Piece),
                           field_width=10,
                           field_height=22,
                           start_config=START_CONFIG,
                           rotations=ROTATIONS)

def get_piece_tile_co(piece, config, ruleset=STANDARD_RULESET):
    rotation = ruleset.rotations[piece][config.rotation]
    return [(config.x + x_offset, config.y + y_offset) for x_offset, y_offset in rotation]

def is_valid(field, piece, config, ruleset=STANDARD_RULESET):
    """Return if config is valid for board with active piece."""
    rotation = ruleset.rotations[piece][config.rotation]
    piece_tiles = [(config.x + x_offset, config.y + y_offset) for x_offset, y_offset in rotation]
    for x, y in piece_tiles:
        cos_valid = (0 <= x < ruleset.field_width and 0 <= y < ruleset.field_height)
        if not cos_valid or field[x][y]:
            return False
    return True

def place_piece(field, piece, config=START_CONFIG, ruleset=STANDARD_RULESET):
    """Take a valid config and return the field with the placed config"""
    piece_tiles = get_piece_tile_co(piece, config, ruleset)
    transposed_field = [[field[x][y] or (x,y) in piece_tiles
                         for x in range(ruleset.field_width)]
                        for y in range(ruleset.field_height)]
    # Burn lines
    transposed_field = [line for line in transposed_field if not all(line)]
    while len(transposed_field) < ruleset.field_height:
        transposed_field.append([False for _ in range(ruleset.field_width)])
    # Transpose back
    return tuple(tuple(transposed_field[y][x]
                       for y in range(ruleset.field_height))
                 for x in range(ruleset.field_width))
