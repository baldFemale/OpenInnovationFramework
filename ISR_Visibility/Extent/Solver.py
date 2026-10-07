"""One solver type: binary local search under a fixed, bounded knowledge scope."""

import numpy as np
from Landscape import require_integer


class Solver:
    def __init__(self, N, landscape, knowledge_breadth, visibility_status=False, rng=None):
        if N != landscape.N:
            raise ValueError("N must match the landscape.")
        require_integer("knowledge_breadth", knowledge_breadth, 1, N)
        self.N = N
        self.landscape = landscape
        self.rng = rng if rng is not None else np.random.default_rng()
        self.knowledge_domain = tuple(sorted(
            self.rng.choice(N, knowledge_breadth, replace=False).tolist()))
        self.state = self.rng.choice(["0", "1"], N).tolist()
        self.perceived_fitness = self.evaluate_solution(self.state)
        self.fitness = landscape.query_fitness(self.state)
        self.visibility_status = bool(visibility_status)

    def evaluate_solution(self, state):
        return self.landscape.query_scoped_fitness(state, self.knowledge_domain)

    def consider_solution(self, state):
        """Adopt a complete position only on strict perceived improvement."""
        perception = self.evaluate_solution(state)
        if perception <= self.perceived_fitness:
            return False
        self.state = list(state)
        self.perceived_fitness = perception
        self.fitness = self.landscape.query_fitness(self.state)
        return True

    def search(self):
        candidate = self.state.copy()
        dimension = self.rng.choice(self.knowledge_domain)
        candidate[dimension] = "1" if candidate[dimension] == "0" else "0"
        return self.consider_solution(candidate)
