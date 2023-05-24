# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape
from CogLandscape import CogLandscape
import pickle


class Tshape:
    def __init__(self, N=None, landscape=None, state_num=4, generalist_expertise=None, specialist_expertise=None):
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        self.specialist_domain = np.random.choice(range(self.N), specialist_expertise // 4, replace=False).tolist()
        self.generalist_domain = np.random.choice([i for i in range(self.N) if i not in self.specialist_domain],
                                                  generalist_expertise // 2, replace=False).tolist()
        self.specialist_representation = ["0", "1", "2", "3"]
        self.generalist_representation = ["A", "B"]
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: a list of string
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
        self.cog_partial_fitness = self.landscape.query_partial_fitness(
            cog_state=self.cog_state, expertise_domain=self.generalist_domain + self.specialist_domain)
        self.fitness = self.landscape.query_fitness(state=self.state)
        if generalist_expertise // 2 + specialist_expertise // 4 > self.N:
            raise ValueError("Expertise Domain Exceed N")
        if (generalist_expertise % 2 != 0) or (specialist_expertise % 4 != 0):
            raise ValueError("Problematic Expertise Amount")

    def search(self):
        next_state = self.state.copy()
        index = np.random.choice(range(self.N))
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.landscape.query_cog_fitness(cog_state=next_cog_state)
        next_cog_partial_fitness = self.landscape.query_partial_fitness(
            cog_state=next_cog_state, expertise_domain=self.generalist_domain + self.specialist_domain)
        # print(next_cog_fitness, next_cog_partial_fitness)
        # if next_cog_fitness >= self.cog_fitness:
        if next_cog_partial_fitness >= self.cog_partial_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.cog_partial_fitness = next_cog_partial_fitness
            self.fitness = self.landscape.query_fitness(state=self.state)

    # def double_search(self, co_state=None, co_expertise_domain=None):
    #     # learning from coupled agent
    #     next_cog_state = self.cog_state.copy()
    #     for index in range(self.N):
    #         if index in self.expertise_domain:
    #             if index in co_expertise_domain:
    #                 changed_cog_state = next_cog_state.copy()
    #                 changed_cog_state[index] = co_state[index]
    #                 if self.landscape.query_cog_fitness_partial(cog_state=changed_cog_state, expertise_domain=self.expertise_domain) > self.cog_fitness:
    #                     next_cog_state[index] = co_state[index]
    #             else:  # retain the private configuration
    #                 pass
    #         else:
    #             # for unknown domains, follow the co-state
    #             if index in co_expertise_domain:
    #                 next_cog_state[index] = co_state[index]
    #             # for unknown domains to both agents, retain the private configuration
    #             else:
    #                 pass
    #     index = np.random.choice(self.expertise_domain)
    #     if index in self.generalist_domain:
    #         if next_cog_state[index] == "A":
    #             next_cog_state[index] = "B"
    #         else:
    #             next_cog_state[index] = "A"
    #     else:
    #         free_space = ["0", "1", "2", "3"]
    #         free_space.remove(self.cog_state[index])
    #         next_cog_state[index] = np.random.choice(free_space)
    #     next_cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_cog_state, expertise_domain=self.expertise_domain)
    #     if next_cog_fitness > self.cog_fitness:
    #         self.cog_state = next_cog_state
    #         self.cog_fitness = next_cog_fitness
    #         self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    # def priority_search(self, co_state=None, co_expertise_domain=None):
    #     # learning from coupled agent
    #     next_cog_state = self.cog_state.copy()
    #     for index in range(self.N):
    #         if index in co_expertise_domain:
    #             next_cog_state[index] = co_state[index]
    #         else:
    #             pass
    #     index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
    #     # thus will not change the unknown domain
    #     space = ["0", "1", "2", "3"]
    #     space.remove(self.state[index])
    #     next_cog_state[index] = np.random.choice(space)
    #     next_cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_cog_state,
    #                                                                 expertise_domain=self.expertise_domain)
    #     if next_cog_fitness > self.cog_fitness:
    #         self.cog_state = next_cog_state
    #         self.cog_fitness = next_cog_fitness
    #         self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def state_2_cog_state(self, state=None):
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if (index not in self.generalist_domain) and (index not in self.specialist_domain):
                cog_state[index] = "*"
            elif index in self.generalist_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            else:
                pass  # specialist_domain
        return cog_state

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if (index not in self.generalist_domain) and (index not in self.specialist_domain):
                state[index] = str(random.choice(range(self.state_num)))
            elif index in self.generalist_domain:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "B":
                    state[index] = random.choice(["2", "3"])
                else:
                    raise ValueError("Unsupported state element: ", bit_value)
            else:
                pass
        return state

    def describe(self):
        print("Tshape of Expertise Domain: ", self.generalist_domain, self.specialist_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive State: {0}, Cognitive Fitness: {1}, Partial Fitness: {2}".format(self.cog_state, self.cog_fitness, self.cog_partial_fitness))


if __name__ == '__main__':
    # Test Example
    import time
    t0 = time.time()
    np.random.seed(1000)
    search_iteration = 500
    N = 9
    K = 0
    state_num = 4
    generalist_expertise = 10
    specialist_expertise = 0
    landscape = Landscape(N=N, K=K, state_num=state_num)
    tshape = Tshape(N=N, landscape=landscape, state_num=state_num,
                    generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
    tshape.describe()
    performance_across_time = []
    cog_performance_across_time = []
    cog_partial_performance_across_time = []
    for _ in range(search_iteration):
        tshape.search()
        # tshape.describe()
        performance_across_time.append(tshape.fitness)
        cog_performance_across_time.append(tshape.cog_fitness)
        cog_partial_performance_across_time.append(tshape.cog_partial_fitness)
    # tshape.describe()
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(search_iteration)
    plt.plot(x, performance_across_time, "k-", label="Fitness")
    plt.plot(x, cog_performance_across_time, "k--", label="Cognitive Fitness")
    plt.plot(x, cog_partial_performance_across_time, "k:", label="Partial Fitness")
    plt.title('Performance at N={0}, K={1}, G={2}, S={3}'.format(N, K, generalist_expertise, specialist_expertise))
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig("T_performance.png", transparent=True, dpi=200)
    plt.show()
    plt.clf()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

