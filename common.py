import random

import numpy as np

from rules import STANDARD_RULESET, ROTATIONS, place_piece, is_valid, Config, ControlAction

def serialize_field(field):
    return "".join([str(i) for i in np.array(field, dtype=int).ravel()])

def deserialize_field(serialized_field):
    return np.array([i=="1" for i in serialized_field], dtype=bool).reshape(10, 22)

def piece_stream(ruleset=STANDARD_RULESET):
    while True:
        yield random.choice(ruleset.pieces)

def list_possible_end_configs(field, piece, ruleset=STANDARD_RULESET, initial_config=None):
    if initial_config is None:
        initial_config = ruleset.start_config
    # List of 2-tuples config/path
    config_stack = [(initial_config, [])]
    config_stack_set = {initial_config}
    end_configs = []
    for config, path in config_stack:
        rotation = Config(config.x, config.y,
                          (config.rotation+1)%len(ROTATIONS[piece])), path + [ControlAction.ROTATION]
        right = Config(config.x+1, config.y, config.rotation), path + [ControlAction.RIGHT]
        left = Config(config.x-1, config.y, config.rotation), path + [ControlAction.LEFT]
        down = Config(config.x, config.y-1, config.rotation), path + [ControlAction.DOWN]
        for next_conf, movement in [right, left, rotation, down]:
            if next_conf not in config_stack_set and is_valid(field, piece, next_conf, ruleset):
                config_stack.append((next_conf, movement))
                config_stack_set.add(next_conf)
        if not is_valid(field, piece, down[0]):
            end_configs.append((config, path))
    return end_configs


def heights(field):
    width = len(field)
    height = len(field[0])
    heights = [0 for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if field[x][y]:
                heights[x] = y+1
    return heights

def nr_holes(field):
    """Return 6 (selected arbitrary) list of holes,
    each element describing the height."""
    ARBITRARY_DEPTH = 6
    width = len(field)
    height = len(field[0]) # Count tuckable holes too, much easier!
    nr_holes = [0 for _ in range(ARBITRARY_DEPTH)]
    for x in range(width):
        depth = 0
        for y in range(height):
            if field[x][y]:
                if depth:
                    nr_holes[min(depth-1, ARBITRARY_DEPTH-1)] += 1
                depth = 0
            else:
                depth += 1
    return nr_holes

def feature_vector(field):
    return heights(field) + nr_holes(field)

def score_field_with_features(model, field):
    features = heights(field) + nr_holes(field)
    return(model.predict([features])[0])

def make_move_with(field, piece, ruleset=STANDARD_RULESET, initial_config=None,
                   rate_function=None):
    def rate_conf(conf_and_movement):
        conf, movement = conf_and_movement
        return rate_function(place_piece(field, piece, conf, ruleset), ruleset)
    possible_configs = list_possible_end_configs(field, piece, ruleset, initial_config)
    next_conf = max(possible_configs, key=rate_conf)
    return next_conf

def make_move_with_model(model, field, piece, initial_config=None):
    def rate_function(field, ruleset=None):
        return score_field_with_features(model, field)
    return make_move_with(field, piece, initial_config=initial_config,
                          rate_function=rate_function)
