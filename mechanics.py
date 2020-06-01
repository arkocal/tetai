from collections import namedtuple
from enum import Enum
import random

class Piece(Enum):
    I = 1
    O = 2
    L = 3
    J = 4
    S = 5
    Z = 6
    T = 7

class ControlAction(Enum):
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    ROTATION = 4

Placement = namedtuple("Placement", ["x", "y", "rotation"])

STANDARD_ROTATIONS = {Piece.I: [[(-2, 0), (-1, 0), (0, 0), (1, 0)],
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

class Mechanics:

    def __init__(self, field_width=10, field_height=22,
                 piece_rotations=STANDARD_ROTATIONS,
                 start_placement=None):
        self.field_width = field_width
        self.field_height = field_height
        self.piece_rotations = piece_rotations
        if start_placement is None:
            start_placement = Placement(x=field_width//2, y=field_height-3,
                                        rotation=0)
        self.start_placement = start_placement


    def get_empty_field(self):
        return [[False for y in range(self.field_height)]
                for x in range(self.field_width)]

    def get_piece_tile_co(self, piece, placement):
        rotation = self.piece_rotations[piece][placement.rotation]
        return [(placement.x + x_offset, placement.y + y_offset)
                for x_offset, y_offset in rotation]


    def can_place_piece(self, field, piece, placement):
        """Return if placement is valid with active piece."""
        def can_place_tile_at(x, y):
            are_cos_valid = (0 <= x < self.field_width and
                             0 <= y < self.field_height)
            return are_cos_valid and not field[x][y]

        return all(can_place_tile_at(x, y)
                   for x, y in self.get_piece_tile_co(piece, placement))

    def place_piece(self, field, piece, placement=None):
        """Take a valid placement and return the field with the placed piece.

        If placement is None, the piece will be place with
        the initial placement."""
        if placement is None:
            placement = self.start_placement
        piece_tiles = self.get_piece_tile_co(piece, placement)
        transposed_field = [[field[x][y] or (x,y) in piece_tiles
                             for x in range(self.field_width)]
                            for y in range(self.field_height)]
        # Burn lines
        transposed_field = [line for line in transposed_field if not all(line)]
        while len(transposed_field) < self.field_height:
            transposed_field.append([False for _ in range(self.field_width)])
        # Transpose back
        return [[transposed_field[y][x]
                 for y in range(self.field_height)]
                for x in range(self.field_width)]

    @property
    def piece_types(self):
        return list(self.piece_rotations.keys())

    def piece_stream(self):
        while True:
            yield random.choice(list(self.piece_types))

    def get_valid_end_placements(self, field, piece, initial_placement=None):
        # Algorithm summary: Placement stack keeps a unique growing
        # list of valid placements. For each placement, there are four
        # neighbours that can be reached with a single ControlAction.
        # The valid neighbours are added to the stack.
        # Basically, this is a WFS of accessible placements.
        if initial_placement is None:
            initial_placement = self.start_placement
        # List of 2-tuples placement/ control action path
        placement_stack = [(initial_placement, [])]
        placement_stack_set = {initial_placement}
        end_placements = []
        for placement, control_action_path in placement_stack:
            rotation = (Placement(placement.x, placement.y,
                                  (placement.rotation+1)%len(self.piece_rotations[piece])),
                        control_action_path + [ControlAction.ROTATION])
            right = (Placement(placement.x+1, placement.y, placement.rotation),
                     control_action_path + [ControlAction.RIGHT])
            left = (Placement(placement.x-1, placement.y, placement.rotation),
                    control_action_path + [ControlAction.LEFT])
            down = (Placement(placement.x, placement.y-1, placement.rotation),
                    control_action_path + [ControlAction.DOWN])
            for next_placement, next_control_action_path in [right, left, rotation, down]:
                if next_placement in placement_stack_set:
                    continue
                if self.can_place_piece(field, piece, next_placement):
                    placement_stack.append((next_placement, next_control_action_path))
                    placement_stack_set.add(next_placement)
            down_placement = down[0]
            if not self.can_place_piece(field, piece, down_placement):
                end_placements.append((placement, control_action_path))
        return end_placements

Config = namedtuple("Config", ["x", "y", "rotation"])
RuleSet = namedtuple("RuleSet", ["pieces", "field_width", "field_height",
                                 "start_config", "rotations"])
