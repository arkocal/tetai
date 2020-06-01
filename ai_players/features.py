"""Features to be used by other models"""

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
