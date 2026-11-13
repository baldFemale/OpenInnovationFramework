# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd.py
# @Software  : PyCharm
# Observing PEP 8 coding style

from Solver import Solver
import numpy as np


class Crowd:
    def __init__(self, N: int, agent_num: int, knowledge_breadth: int,
                 landscape: object):
        self.N = N
        self.agent_num = agent_num
        self.agents = []

        for _ in range(agent_num):
            agent = Solver(N=N, landscape=landscape,
                           knowledge_breadth=knowledge_breadth)
            self.agents.append(agent)

        self.solution_pool = []
        self.adopted_solution_fitness_history = []

    def search(self):
        for agent in self.agents:
            agent.search()

    def set_visibility_status(self, visibility_extent: float):
        """
        Fix solver-level visibility status for the whole experiment.

        visibility_extent is interpreted as the proportion of solvers whose
        solutions are structurally visible. Once assigned, visibility_status
        does not change across visibility periods unless this method is called
        again.
        """
        if visibility_extent < 0 or visibility_extent > 1:
            raise ValueError("visibility_extent must be between 0 and 1.")

        visible_num = int(round(visibility_extent * self.agent_num))
        visible_indices = np.random.choice(range(self.agent_num),
                                           size=visible_num,
                                           replace=False).tolist()

        for index, agent in enumerate(self.agents):
            agent.visibility_status = index in visible_indices

    def get_visible_pool(self):
        """
        Construct the visible solution pool.

        Each structurally visible solver discloses its complete current
        solution. The order of visible solutions is randomized before
        receivers evaluate them.
        """
        self.solution_pool = []

        for agent in self.agents:
            if agent.visibility_status:
                solution = agent.state.copy()
                self.solution_pool.append(solution)

        np.random.shuffle(self.solution_pool)

    def learn_from_visible_pool(self):
        """
        Receiver solvers evaluate visible solutions using their own bounded
        knowledge and adopt the first solution that improves perceived fitness.
        """
        for agent in self.agents:
            for solution in self.solution_pool:
                if agent.search(solution=solution):
                    self.adopted_solution_fitness_history.append(agent.fitness)
                    break

    def calculate_pairwise_solution_distance(self):
        """Average pairwise normalized Hamming distance across complete solutions."""
        states = [agent.state for agent in self.agents]

        if len(states) <= 1:
            return 0

        distance_list = []

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                distance = (
                    sum(
                        1
                        for bit_i, bit_j in zip(states[i], states[j])
                        if bit_i != bit_j
                    )
                    / self.N
                )
                distance_list.append(distance)

        return np.mean(distance_list)

    def unique_solution_count(self):
        full_solution_set = set()

        for agent in self.agents:
            solution_str = "".join([str(bit) for bit in agent.state])
            full_solution_set.add(solution_str)

        return len(full_solution_set)
