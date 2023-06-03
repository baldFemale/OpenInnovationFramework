# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape


class Agent:
    def __init__(self, N=None, landscape=None, state_num=4,
                 generalist_expertise=None, specialist_expertise=None, manner="Partial"):
        """
        :param N: problem dimension
        :param landscape: assigned landscape
        :param state_num: state number for each dimension
        :param generalist_expertise: the amount of G knowledge
        :param specialist_expertise: the amount of S knowledge
        """
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        if generalist_expertise and specialist_expertise:
            self.specialist_domain = np.random.choice(range(self.N), specialist_expertise // 4, replace=False).tolist()
            self.generalist_domain = np.random.choice([i for i in range(self.N) if i not in self.specialist_domain],
                                                  generalist_expertise // 2, replace=False).tolist()
        elif generalist_expertise:
            self.generalist_domain = np.random.choice(range(self.N),  generalist_expertise // 2, replace=False).tolist()
            self.specialist_domain = []
        elif specialist_expertise:
            self.generalist_domain = []
            self.specialist_domain = np.random.choice(range(self.N), specialist_expertise // 4, replace=False).tolist()
        self.specialist_representation = ["0", "1", "2", "3"]
        self.generalist_representation = ["A", "B"]
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: a list of string
        self.cog_state, self.cog_fitness, self.fitness = [], [], []
        self.manner = manner
        self.update_fitness()
        self.cog_cache = {}
        if specialist_expertise and generalist_expertise:
            if generalist_expertise // 2 + specialist_expertise // 4 > self.N:
                raise ValueError("Entire Expertise Exceed N")
        if generalist_expertise and (generalist_expertise % 2 != 0):
            raise ValueError("Problematic G Expertise")
        if specialist_expertise and (specialist_expertise % 4 != 0):
            raise ValueError("Problematic S Expertise")

    def update_fitness(self):
        self.cog_state = self.state_2_cog_state(state=self.state)
        if self.manner == "Full":
            self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state, state=self.state)
        else:
            self.cog_fitness = self.landscape.query_partial_fitness(
                cog_state=self.cog_state, state=self.state, expertise_domain=self.generalist_domain + self.specialist_domain)
        self.fitness = self.landscape.query_fitness(state=self.state)

    def search(self):
        next_state = self.state.copy()
        index = np.random.choice(self.generalist_domain + self.specialist_domain)
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        perception = "".join(next_cog_state)
        if self.manner == "Full":
            if perception not in self.cog_cache.keys():
                next_cog_fitness = self.landscape.query_cog_fitness(cog_state=next_cog_state, state=self.state)
                self.cog_cache[perception] = next_cog_fitness
            else:
                next_cog_fitness = self.cog_cache[perception]
        else:
            if perception not in self.cog_cache.keys():
                next_cog_fitness = self.landscape.query_partial_fitness(
                    cog_state=next_cog_state, state=self.state, expertise_domain=self.generalist_domain + self.specialist_domain)
                self.cog_cache[perception] = next_cog_fitness
            else:
                next_cog_fitness = self.cog_cache[perception]
        if next_cog_fitness > self.cog_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness = self.landscape.query_fitness(state=self.state)

    def state_2_cog_state(self, state=None):
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if index in self.generalist_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            elif index in self.specialist_domain:
                pass
            else:
                cog_state[index] = "*"
        return cog_state

    # def cog_state_2_state(self, cog_state=None):
    #     state = cog_state.copy()
    #     for index, bit_value in enumerate(cog_state):
    #         # if (index not in self.generalist_domain) and (index not in self.specialist_domain):
    #         #     state[index] = str(random.choice(range(self.state_num)))
    #         if index in self.generalist_domain:
    #             if bit_value == "A":
    #                 state[index] = random.choice(["0", "1"])
    #             elif bit_value == "B":
    #                 state[index] = random.choice(["2", "3"])
    #             else:
    #                 raise ValueError("Unsupported state element: ", bit_value)
    #         else:
    #             pass
    #     return state

    def describe(self):
        print("Agent of G/S Domain: ", self.generalist_domain, self.specialist_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive State: {0}, Cognitive Fitness: {1}".format(self.cog_state, self.cog_fitness))


if __name__ == '__main__':
    # Test Example
    import time
    t0 = time.time()
    np.random.seed(1000)
    search_iteration = 100
    N = 9
    K = 8
    state_num = 4
    generalist_expertise = 0
    specialist_expertise = 36
    landscape = Landscape(N=N, K=K, state_num=state_num)
    landscape.describe()
    agent = Agent(N=N, landscape=landscape, state_num=state_num,
                    generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
    # agent.describe()
    performance_across_time = []
    cog_performance_across_time = []
    for _ in range(search_iteration):
        agent.search()
        performance_across_time.append(agent.fitness)
        cog_performance_across_time.append(agent.cog_fitness)
    print("Search Result: ")
    print("".join(agent.state), "".join(agent.cog_state), agent.fitness)
    landscape.count_local_optima()
    focal_state = ["0", "0", "0", "0", "0", "0", "0", "0", "0"]
    print("Focal Neighbor")
    neighbor_list = landscape.get_neighbor_list(key="".join(focal_state))
    for neighbor in neighbor_list:
        fitness_ = landscape.query_fitness(state=neighbor)
        print(neighbor, fitness_)
    print("Local Optima")
    for key, value in landscape.local_optima.items():
        print(key, value)
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(search_iteration)
    plt.plot(x, performance_across_time, "k-", label="Fitness")
    plt.plot(x, cog_performance_across_time, "k--", label="Cognitive Fitness")
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

