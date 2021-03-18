from ai_players.base import AIPlayer

class RuleBasedAIPlayer(AIPlayer):

    def score_field(self, field):
        weights = [-1, -0.03, -0.01, 0]
        max_height = 0
        nr_holes = 0 # Counts tuckable holes as well
        unflatness = 0
        top_at_col = [max([y for y in range(self.mechanics.field_height) if field[x][y]],
                          default=0)
                      for x in range(self.mechanics.field_width)]
        unflatness = sum([(top_at_col[i+1]-top_at_col[i])**2
                          for i in range(self.mechanics.field_width-1)])
        hole_ready = 0
        for y in range(self.mechanics.field_height):
            if not field[self.mechanics.field_width-1][y]:
                hole_ready += (self.mechanics.field_height-y)**2
        for x in range(self.mechanics.field_width):
            for y in range(self.mechanics.field_height):
                if field[x][y]:
                    if y>=1 and not field[x][y-1]:
                        nr_holes += 1
                    max_height = max(max_height, y)
        rate = 0
        for weight, factor in zip(weights, [nr_holes, unflatness, max_height**2.5, hole_ready]):
            rate += weight * factor
        return rate
