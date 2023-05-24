# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape
import pickle


class Generalist:
    def __init__(self, N=None, landscape=None, cog_landscape=None, state_num=4, expertise_amount=None):
        self.landscape = landscape
        self.cog_landscape = cog_landscape
        self.N = N
        self.state_num = state_num
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: a list of string
        self.expertise_representation = ["A", "B"]
        self.expertise_domain = np.random.choice(range(self.N), expertise_amount // 2, replace=False).tolist()
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
        self.cog_partial_fitness = self.landscape.query_partial_fitness(cog_state=self.cog_state, generalist_domain=self.expertise_domain)
        self.fitness = self.landscape.query_fitness(state=self.state)
        # Mechanism: overlap with IM
        self.row_overlap = 0
        self.column_overlap = 0

    # def get_overlap_with_IM(self):
    #     influence_matrix = self.landscape.IM
    #     row_overlap, column_overlap = 0, 0
    #     for row in range(self.N):
    #         if row in self.expertise_domain:
    #             row_overlap += sum(influence_matrix[row])
    #     for column in range(self.N):
    #         if column in self.expertise_domain:
    #             column_overlap += sum(influence_matrix[:, column])
    #     self.column_overlap = column_overlap
    #     self.row_overlap = row_overlap

    # def align_default_state(self, state=None):
    #     for index in range(self.N):
    #         if index not in self.expertise_domain:
    #             self.state[index] = state[index]
    #     self.cog_state = self.state_2_cog_state(state=self.state)
    #     self.cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=self.cog_state, expertise_domain=self.expertise_domain)
    #     self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    # def learn_from_pool(self, pool=None):
    #     exposure_state = pool[np.random.choice(len(pool))]
    #     cog_exposure_state = self.state_2_cog_state(state=exposure_state)
    #     cog_fitness_of_exposure_state = self.landscape.query_cog_fitness_partial(cog_state=cog_exposure_state, expertise_domain=self.expertise_domain)
    #     if cog_fitness_of_exposure_state > self.cog_fitness:
    #         self.cog_state = cog_exposure_state
    #         self.cog_fitness = cog_fitness_of_exposure_state
    #         return True
    #     return False

    def search(self):
        next_state = self.state.copy()
        index = np.random.choice(self.N)
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.landscape.query_cog_fitness(cog_state=next_cog_state)
        next_cog_partial_fitness = self.landscape.query_partial_fitness(cog_state=next_cog_state, generalist_domain=self.expertise_domain)
        # if next_cog_fitness >= self.cog_fitness:
        if next_cog_partial_fitness >= self.cog_partial_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.cog_partial_fitness = next_cog_partial_fitness
            self.fitness = self.landscape.query_fitness(state=self.state)

    # def double_search(self, co_state=None, co_expertise_domain=None):
    #     # learning from coupled agent
    #     next_coarse_state = self.cog_state.copy()
    #     for index in range(self.N):
    #         if index in self.expertise_domain:
    #             if index in co_expertise_domain:
    #                 changed_cog_state = next_coarse_state.copy()
    #                 changed_cog_state[index] = co_state[index]
    #                 if self.landscape.query_cog_fitness_partial(cog_state=changed_cog_state, expertise_domain=self.expertise_domain) > self.cog_fitness:
    #                     next_coarse_state[index] = co_state[index]
    #             else:  # retain the private configuration
    #                 pass
    #         else:
    #             # for unknown domains, follow the co-state
    #             if index in co_expertise_domain:
    #                 next_coarse_state[index] = co_state[index]
    #             # for unknown domains to both agents, retain the private configuration
    #             else:
    #                 pass
    #     index = np.random.choice(self.expertise_domain)
    #     if next_coarse_state[index] == "A":
    #         next_coarse_state[index] = "B"
    #     else:
    #         next_coarse_state[index] = "A"
    #     next_coarse_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_coarse_state, expertise_domain=self.expertise_domain)
    #     if next_coarse_fitness > self.cog_fitness:
    #         self.cog_state = next_coarse_state
    #         self.cog_fitness = next_coarse_fitness
    #         self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    # def priority_search(self, co_state=None, co_expertise_domain=None):
    #     # learning from coupled agent
    #     next_coarse_state = self.cog_state.copy()
    #     for index in range(self.N):
    #         if index in co_expertise_domain:
    #             next_coarse_state[index] = co_state[index]
    #         else:
    #             pass
    #     index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
    #     # thus will not change the unknown domain
    #     space = ["0", "1", "2", "3"]
    #     space.remove(self.state[index])
    #     next_coarse_state[index] = np.random.choice(space)
    #     next_coarse_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_coarse_state,
    #                                                                 expertise_domain=self.expertise_domain)
    #     if next_coarse_fitness > self.cog_fitness:
    #         self.cog_state = next_coarse_state
    #         self.cog_fitness = next_coarse_fitness
    #         self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def state_2_cog_state(self, state=None):
        cog_state = self.state.copy()
        for index, bit_value in enumerate(state):
            if index in self.expertise_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            else:
                # pass  # remove the ambiguity in the unknown domain-> mindset or untunable domain
                cog_state[index] = "*"
        return cog_state

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if (index not in self.expertise_domain) and (bit_value == "*"):
                state[index] = str(random.choice(range(self.state_num)))
            else:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "B":
                    state[index] = random.choice(["2", "3"])
                else:
                    raise ValueError("Unsupported state element: ", bit_value)
        return state

    def describe(self):
        print("Generalist of Expertise Domain: ", self.expertise_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive State: {0}, Cognitive Fitness: {1}, Partial Fitness: {2}".format(self.cog_state, self.cog_fitness, self.cog_partial_fitness))


if __name__ == '__main__':
    # Test Example
    import time
    t0 = time.time()
    search_iteration = 1000
    N = 9
    K = 0
    state_num = 4
    expertise_amount = 18
    landscape = Landscape(N=N, K=K, state_num=state_num)
    generalist = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
    generalist.describe()
    performance_across_time = []
    cog_performance_across_time = []
    cog_partial_performance_across_time = []
    for _ in range(search_iteration):
        generalist.search()
        performance_across_time.append(generalist.fitness)
        cog_performance_across_time.append(generalist.cog_fitness)
        cog_partial_performance_across_time.append(generalist.cog_partial_fitness)
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(search_iteration)
    plt.plot(x, performance_across_time, "k-", label="Fitness")
    plt.plot(x, cog_performance_across_time, "k--", label="Cognitive Fitness")
    plt.plot(x, cog_partial_performance_across_time, "k:", label="Partial Fitness")
    plt.title('Performance at N={0}, K={1}, Kn={2}'.format(N, K, expertise_amount))
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig("G_performance.png", transparent=True, dpi=200)
    plt.show()
    plt.clf()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))