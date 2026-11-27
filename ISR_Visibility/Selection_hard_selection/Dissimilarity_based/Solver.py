# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Solver.py
# @Software  : PyCharm
# Observing PEP 8 coding style

import numpy as np
from Landscape import Landscape


class Solver:
    def __init__(self, N=None, landscape=None, knowledge_breadth=None,
                 visibility_status=False):
        """
        :param N: problem dimension
        :param landscape: assigned landscape
        :param knowledge_breadth: number of problem dimensions within the solver's knowledge scope
        :param visibility_status: whether this solver's solution is structurally visible
        """
        self.landscape = landscape
        self.N = N
        self.visibility_status = visibility_status

        self.knowledge_domain = np.random.choice(range(self.N), knowledge_breadth,
                                                 replace=False).tolist()

        self.state = np.random.choice(["0", "1"], self.N).tolist()

        self.cog_fitness = self.landscape.query_scoped_fitness(
            state=self.state,
            knowledge_domain=self.knowledge_domain
        )
        self.fitness = self.landscape.query_fitness(state=self.state)

        self.cog_fitness_across_time = [self.cog_fitness]
        self.fitness_across_time = [self.fitness]

    def search(self, solution=None) -> bool:
        """Try one local bit flip, or evaluate a supplied complete visible solution.

        Both moves use the receiver's knowledge and require strict perceived
        improvement. Histories retain one entry per ordinary search opportunity.
        """
        local_search = solution is None
        if local_search:
            next_state = self.state.copy()
            index = np.random.choice(self.knowledge_domain)
            next_state[index] = "1" if next_state[index] == "0" else "0"
        else:
            next_state = list(solution)

        next_cog_fitness = self.landscape.query_scoped_fitness(
            state=next_state,
            knowledge_domain=self.knowledge_domain
        )
        adopted = bool(next_cog_fitness > self.cog_fitness)
        if adopted:
            next_fitness = self.landscape.query_fitness(state=next_state)
            self.state = next_state
            self.cog_fitness = next_cog_fitness
            self.fitness = next_fitness

        if local_search:
            self.cog_fitness_across_time.append(self.cog_fitness)
            self.fitness_across_time.append(self.fitness)
        return adopted

    def estimate_generative_potential(self, starting_state, horizon=30,
                                      replications=30, rng=None) -> float:
        """Expected best objective-fitness gain from independent local search.

        Use this solver's fixed knowledge domain and the original landscape.
        All continuation states are local copies; the solver is unchanged.
        """
        if horizon < 0 or replications < 1:
            raise ValueError("horizon must be nonnegative and replications must be positive.")
        if horizon == 0:
            return 0.0
        if rng is None:
            rng = np.random.default_rng()  # Independent of the main search RNG.

        initial_state = list(starting_state)
        initial_fitness = self.landscape.query_fitness(state=initial_state)
        initial_cog_fitness = self.landscape.query_scoped_fitness(
            state=initial_state, knowledge_domain=self.knowledge_domain
        )

        total_gain = 0.0
        for _ in range(replications):
            state = initial_state.copy()
            cog_fitness = initial_cog_fitness
            best_fitness = initial_fitness

            for _ in range(horizon):
                # The original heuristic: flip one known bit and accept only
                # a strict improvement in perceived (scoped) fitness.
                next_state = state.copy()
                index = rng.choice(self.knowledge_domain)
                next_state[index] = "1" if next_state[index] == "0" else "0"
                next_cog_fitness = self.landscape.query_scoped_fitness(
                    state=next_state, knowledge_domain=self.knowledge_domain
                )
                if next_cog_fitness > cog_fitness:
                    state = next_state
                    cog_fitness = next_cog_fitness
                    best_fitness = max(
                        best_fitness, self.landscape.query_fitness(state=state)
                    )

            total_gain += best_fitness - initial_fitness

        return total_gain / replications

    def describe(self) -> None:
        print("Solver Knowledge Domain: ", self.knowledge_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive Fitness: ", self.cog_fitness)


if __name__ == '__main__':
    # Test Example
    import time

    t0 = time.time()

    np.random.seed(1000)
    search_iteration = 100

    N = 9
    K = 2
    knowledge_breadth = 3

    landscape = Landscape(N=N, K=K)
    solver = Solver(N=N, landscape=landscape,
                    knowledge_breadth=knowledge_breadth)

    for _ in range(search_iteration):
        solver.search()
        print(solver.state, solver.cog_fitness, solver.fitness)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
