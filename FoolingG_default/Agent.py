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
                 generalist_expertise=None, specialist_expertise=None):
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
        self.cog_state = []  # for G: shallow; for S: scoped -> built upon common sense
        self.cog_fitness, self.fitness = 0, 0
        self.cog_fitness_across_time, self.fitness_across_time = [], []
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
        if len(self.generalist_domain) != 0:  # iff G
            if len(self.generalist_domain) == self.N:  # iff full G
                self.cog_fitness = self.landscape.query_first_fitness(state=self.cog_state)
            else:
                self.cog_fitness = self.landscape.query_scoped_first_fitness(cog_state=self.cog_state, state=self.state)
        else:  # iff S
            if len(self.specialist_domain) == self.N:  # iff full S
                self.cog_fitness = self.landscape.query_second_fitness(state=self.state)
            else:
                self.cog_fitness = self.landscape.query_scoped_second_fitness(cog_state=self.cog_state, state=self.state)
        self.fitness = self.landscape.query_second_fitness(state=self.state)

    def search(self):
        next_state = self.state.copy()
        # index = np.random.choice(self.generalist_domain + self.specialist_domain)
        index = np.random.choice(range(self.N))  # if mindset changes; if environmental turbulence arise outside one's knowledge
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        if len(self.generalist_domain) != 0:  # iff G
            if len(self.generalist_domain) == self.N:  # iff full G
                next_cog_fitness = self.landscape.query_first_fitness(state=next_cog_state)
            else:
                next_cog_fitness = self.landscape.query_scoped_first_fitness(cog_state=next_cog_state, state=next_state)
        else:  # iff S
            if len(self.specialist_domain) == self.N:  # iff full S
                next_cog_fitness = self.landscape.query_second_fitness(state=next_state)
            else:
                next_cog_fitness = self.landscape.query_scoped_second_fitness(cog_state=next_cog_state, state=next_state)
        if next_cog_fitness >= self.cog_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.fitness_across_time.append(self.fitness)
        self.cog_fitness_across_time.append(self.cog_fitness)

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
    # np.random.seed(1000)
    search_iteration = 500
    N = 9
    K = 8
    state_num = 4
    generalist_expertise = 0
    specialist_expertise = 20
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.5)

    # landscape.describe()
    agent = Agent(N=N, landscape=landscape, state_num=state_num,
                    generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
    # agent.describe()
    for _ in range(search_iteration):
        agent.search()
    import matplotlib.pyplot as plt
    x = range(len(agent.fitness_across_time))
    plt.plot(x, agent.fitness_across_time, "k-", label="Fitness")
    plt.plot(x, agent.cog_fitness_across_time, "k--", label="Cognitive Fitness")
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

