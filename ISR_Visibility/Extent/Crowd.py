"""Independent local search and optional within-crowd solution visibility."""

import numpy as np
from Solver import Solver
from Landscape import require_integer


class Crowd:
    def __init__(self, N, agent_num, knowledge_breadth, landscape,
                 rng=None, visibility_rng=None):
        require_integer("agent_num", agent_num, 1)
        self.N, self.agent_num = N, agent_num
        self.rng = rng if rng is not None else np.random.default_rng()
        self.visibility_rng = (visibility_rng if visibility_rng is not None
                               else np.random.default_rng())
        self.agents = [Solver(N, landscape, knowledge_breadth, rng=self.rng)
                       for _ in range(agent_num)]
        self.solution_pool = ()
        self.adopted_solution_fitness_history = []

    def search(self):
        for agent in self.agents:
            agent.search()

    def set_visibility_status(self, visibility_extent):
        """Assign round(extent * population) visible paths; retain until reassigned.

        A full permutation uses the same random draws at every extent. With a
        matched seed, smaller visible sets are subsets of larger visible sets.
        """
        if not np.isfinite(visibility_extent) or not 0 <= visibility_extent <= 1:
            raise ValueError("visibility_extent must be between 0 and 1.")
        visible_num = int(round(visibility_extent * self.agent_num))
        visible = set(self.visibility_rng.permutation(self.agent_num)[:visible_num])
        for index, agent in enumerate(self.agents):
            agent.visibility_status = index in visible

    def get_visible_pool(self):
        """Return immutable (sender index, complete state) snapshots in random order."""
        order = self.visibility_rng.permutation(self.agent_num)
        self.solution_pool = tuple(
            (int(index), tuple(self.agents[index].state))
            for index in order if self.agents[index].visibility_status)
        return self.solution_pool

    def learn_from_visible_pool(self):
        """Each receiver accepts the first improving external solution, if any.

        The fixed snapshot is never updated during this event. Pool order is
        shared across receivers, as in the existing self-selection rule.
        """
        adopted = []
        for receiver_id, agent in enumerate(self.agents):
            for sender_id, solution in self.solution_pool:
                if sender_id == receiver_id:
                    continue
                if agent.consider_solution(solution):
                    adopted.append(agent.fitness)
                    break
        self.adopted_solution_fitness_history.extend(adopted)
        return adopted

    def calculate_dispersion(self):
        """Average pairwise normalized Hamming distance over complete solutions."""
        if self.agent_num <= 1:
            return 0.0
        ones = np.sum([[bit == "1" for bit in agent.state] for agent in self.agents], axis=0)
        different_pairs = np.sum(ones * (self.agent_num - ones))
        pairs = self.agent_num * (self.agent_num - 1) / 2
        return float(different_pairs / (pairs * self.N))

    def unique_solution_count(self):
        return len({tuple(agent.state) for agent in self.agents})
