class AIPlayer:
    """Main AI Player class,
    implementations should subclass this."""

    def __init__(self, mechanics):
        self.mechanics = mechanics

    def score_field(self, field):
        """Give a score for each field, this is used by 'choose_placement',
        and must be implemented if that method is not overwritten."""
        raise NotImplementedError

    def choose_placement(self, field, piece, initial_placement=None,
                         next_pieces=None):
        """Return a placement and list of ControlActions
        that reach the end placement."""
        return max((placement_and_path for placement_and_path in self.mechanics.get_valid_end_placements(field, piece, initial_placement)),
                   key=lambda x: self.score_field(self.mechanics.place_piece(field, piece, x[0])))
