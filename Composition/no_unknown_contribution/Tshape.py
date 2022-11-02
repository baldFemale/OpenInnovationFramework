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


class Tshape:
    def __init__(self, N=None, landscape=None, state_num=4, generalist_expertise=None, specialist_expertise=None):
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: string
        self.generalist_knowledge_representation = ["A", "B"]
        free_space = list(range(N))
        self.specialist_domain = np.random.choice(free_space, specialist_expertise // 4, replace=False).tolist()
        for domain in self.specialist_domain:
            free_space.remove(domain)
        self.generalist_domain = np.random.choice(free_space, generalist_expertise // 2, replace=False).tolist()
        self.expertise_domain = self.specialist_domain + self.generalist_domain
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
        self.fitness = None

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")
        if generalist_expertise % 2 != 0:
            raise ValueError("Generalist expertise amount needs to be a even number")
        if specialist_expertise % 4 != 0:
            raise ValueError("Specialist expertise amount needs to be a product of 4")
        if len(self.expertise_domain) > self.N:
            raise ValueError("The expertise domain should not be greater than N")

    def align_default_state(self, loop=None):
        with open("default_state_list", "rb") as infile:
            default_state_list = pickle.load(infile)
        default_state = default_state_list[loop]
        for index in range(self.N):
            if index not in self.expertise_domain:
                self.state[index] = default_state[index]
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)

    def learn_from_pool(self, pool=None):
        exposure_state = pool[np.random.choice(len(pool))]
        cog_exposure_state = self.state_2_cog_state(state=exposure_state)
        cog_fitness_of_exposure_state = self.landscape.query_cog_fitness(cog_state=cog_exposure_state)
        if cog_fitness_of_exposure_state > self.cog_fitness:
            self.cog_state = cog_exposure_state
            self.cog_fitness = cog_fitness_of_exposure_state
            return True
        return False

    def search(self):
        next_cog_state = self.cog_state.copy()
        index = np.random.choice(self.expertise_domain)
        if index in self.generalist_domain:
            if next_cog_state[index] == "A":
                next_cog_state[index] = "B"
            else:
                next_cog_state[index] = "A"
        else:
            free_space = ["0", "1", "2", "3"]
            free_space.remove(self.cog_state[index])
            next_cog_state[index] = np.random.choice(free_space)
        next_cog_fitness = self.landscape.query_cog_fitness_without_unknown(cog_state=next_cog_state, expertise_domain=self.expertise_domain)
        if next_cog_fitness > self.cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness

    def distant_jump(self):
        distant_state = np.random.choice(range(self.state_num), self.N).tolist()
        distant_state = [str(i) for i in distant_state]
        cog_distant_state = self.state_2_cog_state(state=distant_state)
        cog_fitness_of_distant_state = self.landscape.query_cog_fitness(cog_state=cog_distant_state)
        if cog_fitness_of_distant_state > self.cog_fitness:
            self.cog_state = cog_distant_state
            self.cog_fitness = cog_fitness_of_distant_state
            return True
        else:
            return False

    def state_2_cog_state(self, state=None):
        cog_state = self.state.copy()
        for index, bit_value in enumerate(state):
            if index not in self.expertise_domain:
                pass  # remove the ambiguity
                # cog_state[index] = "*"
            elif index in self.generalist_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            else:pass  # specialist_domain
        return cog_state

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if index not in self.expertise_domain:
                state[index] = str(random.choice(range(self.state_num)))
            elif index in self.generalist_domain:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "B":
                    state[index] = random.choice(["2", "3"])
                else:
                    raise ValueError("Unsupported state element: ", bit_value)
        return state

    def describe(self):
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state list: ", self.state)
        print("Current cognitive state list: ", self.cog_state)
        print("Current cognitive fitness: ", self.cog_fitness)
        print("Converged fitness: ", self.fitness)
        print("Expertise domain: ", self.expertise_domain)


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=10, state_num=4)
    landscape.type(IM_type="Traditional Directed", K=4, k=0)
    landscape.initialize()
    t_shape = Tshape(N=10, landscape=landscape, state_num=4, generalist_expertise=8, specialist_expertise=8)
    # jump_count = 0
    # for _ in range(1000):
    #     t_shape.search()
    #     if t_shape.distant_jump():
    #         jump_count += 1
    #     # print(generalist.cog_fitness)
    # print("jump_count: ", jump_count)
    # t_shape.state = t_shape.cog_state_2_state(cog_state=t_shape.cog_state)
    # t_shape.fitness = landscape.query_fitness(state=t_shape.state)
    # t_shape.describe()
    # print("END")

    # Test for the search rounds upper boundary
    cog_performance_across_time = []
    for _ in range(200):
        t_shape.search()
        t_shape.distant_jump()
        cog_performance_across_time.append(t_shape.cog_fitness)
    t_shape.state = t_shape.cog_state_2_state(cog_state=t_shape.cog_state)
    t_shape.fitness = landscape.query_fitness(state=t_shape.state)
    print(t_shape.fitness, t_shape.cog_fitness)
    import matplotlib.pyplot as plt
    x = range(200)
    plt.plot(x, cog_performance_across_time, "r-", label="G")
    # plt.title('Diversity Decrease')
    plt.xlabel('Time', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    plt.legend(frameon=False, ncol=3, fontsize=10)
    # plt.savefig("GST_performance_K.png", transparent=True, dpi=1200)
    plt.show()
