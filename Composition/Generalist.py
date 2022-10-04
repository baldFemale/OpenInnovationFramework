# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape


class Generalist:
    def __init__(self, N=None, landscape=None, state_num=4, expertise_amount=None):
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: string
        self.generalist_knowledge_representation = ["A", "B"]
        self.expertise_domain = np.random.choice(range(self.N), expertise_amount // 2, replace=False).tolist()
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
        self.fitness = None

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")
        if expertise_amount % 2 != 0:
            raise ValueError("Expertise amount needs to be a even number")
        if expertise_amount > self.N * 2:
            raise ValueError("Expertise amount should be less than {0}.".format(self.N * 2))

    def learn(self, pool=None):
        exposure_state = np.random.choice(pool)
        cog_exposure_state = self.state_2_cog_state(state=exposure_state)
        cog_fitness_of_exposure_state = self.landscape.query_cog_fitness(cog_state=cog_exposure_state)
        if cog_fitness_of_exposure_state > self.cog_fitness:
            self.cog_state = cog_exposure_state
            self.cog_fitness = cog_fitness_of_exposure_state

    def search(self):
        next_cog_state = self.cog_state.copy()
        index = np.random.choice(self.expertise_domain)
        if next_cog_state[index] == "A":
            next_cog_state[index] = "B"
        else:
            next_cog_state[index] = "A"
        next_cog_fitness = self.landscape.query_cog_fitness(cog_state=next_cog_state)
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
            if index in self.expertise_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            else:
                pass  # remove the ambiguity in the unknown domain-> mindset or untunable domain
        return cog_state

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if index not in self.expertise_domain:
                pass  # remove the ambiguity in the unknown domain
                # state[index] = str(random.choice(range(self.state_num)))
            else:
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
    landscape = Landscape(N=8, state_num=4)
    landscape.type(IM_type="Factor Directed", K=0, k=42)
    landscape.initialize()
    generalist = Generalist(N=8, landscape=landscape, state_num=4, expertise_amount=16)
    jump_count = 0
    for _ in range(1000):
        generalist.search()
        if generalist.distant_jump():
            jump_count += 1
        # print(generalist.cog_fitness)
    print("jump_count: ", jump_count)
    generalist.state = generalist.cog_state_2_state(cog_state=generalist.cog_state)
    generalist.fitness = landscape.query_fitness(state=generalist.state)
    generalist.describe()
    print("END")

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]


