# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd_imitation.py
# @Software  : PyCharm
# Observing PEP 8 coding style

from Solver import Solver
import numpy as np


class Crowd:
    def __init__(self, N: int, agent_num: int, knowledge_breadth: int,
                 landscape: object, generative_sample_size=0, generative_rng=None):
        self.N = N
        self.agent_num = agent_num
        self.agents = []

        for _ in range(agent_num):
            agent = Solver(N=N, landscape=landscape,
                           knowledge_breadth=knowledge_breadth)
            self.agents.append(agent)

        self.solution_pool = []
        self.adopted_solution_fitness_history = []

        # Uniform reservoir of adoption events; measurement is opt-in.
        if generative_sample_size < 0:
            raise ValueError("generative_sample_size must be nonnegative.")
        self.generative_sample_size = int(generative_sample_size)
        self.generative_samples = []
        self.generative_rng = (
            generative_rng if generative_rng is not None else np.random.default_rng()
        ) if self.generative_sample_size else None

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

    def learn_from_visible_pool(self, imitation_breadth=1.0):
        """
        Copy a proportion of the N dimensions from each evaluated visible solution.
        Receivers evaluate one recombinant candidate per visible solution and
        adopt the first candidate that improves perceived fitness.
        imitation_breadth=1.0 preserves the original whole-solution adoption.
        """
        if not 0.0 <= imitation_breadth <= 1.0:
            raise ValueError("imitation_breadth must be between 0 and 1.")
        copy_num = int(imitation_breadth * self.N)
        if copy_num == 0:
            return  # No decisions can change through visibility.

        for agent in self.agents:
            for solution in self.solution_pool:
                if copy_num == self.N:
                    candidate = solution  # Original behavior; no additional RNG draws.
                else:
                    candidate = agent.state.copy()
                    copied_dimensions = np.random.choice(self.N, size=copy_num, replace=False)
                    for dimension in copied_dimensions:
                        candidate[dimension] = solution[dimension]

                if agent.search(solution=candidate):
                    self.adopted_solution_fitness_history.append(agent.fitness)
                    if self.generative_sample_size:
                        # Sample events uniformly, irrespective of adoption time.
                        count = len(self.adopted_solution_fitness_history)
                        event = (agent, agent.state.copy())
                        if len(self.generative_samples) < self.generative_sample_size:
                            self.generative_samples.append(event)
                        else:
                            index = self.generative_rng.integers(count)
                            if index < self.generative_sample_size:
                                self.generative_samples[index] = event
                    break

    def calculate_generative_potential(self, horizon=30, replications=30):
        """Mean generative potential across sampled adoption events (NaN if none)."""
        if not self.generative_samples:
            return np.nan
        return np.mean([
            agent.estimate_generative_potential(
                starting_state=state, horizon=horizon,
                replications=replications, rng=self.generative_rng
            )
            for agent, state in self.generative_samples
        ])

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
