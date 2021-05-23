

class Coin:
    def __init__(self, pos):
        self.pos = pos
        self.broken = False
        self.broken_freshness = 1.0
        self.decay_fn = lambda x: x/2
    
    def decay_one_step(self):
        self.broken_freshness = self.decay_fn(self.broken_freshness)