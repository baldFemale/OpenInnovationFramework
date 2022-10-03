# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape


class Specialist:
    def __init__(self, N=None, landscape=None, state_num=4, expertise_amount=None):
        """
        For Specialist, there is no depth penalty or shallow understanding ambiguity
        """
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        self.expertise_domain = np.random.choice(range(self.N), expertise_amount // 4, replace=False).tolist()
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: string
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
        self.fitness = None

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")
        if expertise_amount % 2 != 0:
            raise ValueError("Expertise amount needs to be a even number")
        if expertise_amount > self.N * 4:
            raise ValueError("Expertise amount should be less than {0}.".format(self.N * 4))

    def learn(self, pool=None):
        exposure_state = np.random.choice(pool)
        exposure_cog_state = self.state_2_cog_state(state=exposure_state)
        exposure_cog_fitness = self.landscape.query_cog_fitness(cog_state=exposure_cog_state)
        if exposure_cog_fitness > self.cog_fitness:
            self.cog_state = exposure_cog_state
            self.cog_fitness = exposure_cog_fitness

    def search(self):
        next_cog_state = self.cog_state.copy()
        index = np.random.choice(self.expertise_domain)
        space = ["0", "1", "2", "3"]
        space.remove(self.state[index])
        next_cog_state[index] = np.random.choice(space)
        next_cog_fitness = self.landscape.query_cog_fitness(cog_state=next_cog_state)
        if next_cog_fitness > self.cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness

    def distant_jump(self):
        distant_state = np.random.choice(range(self.state_num), self.N).tolist()
        distant_state = [str(i) for i in distant_state]
        distant_cog_state = self.state_2_cog_state(state=distant_state)
        distant_cog_fitness = self.landscape.query_cog_fitness(cog_state=distant_cog_state)
        if distant_cog_fitness > self.cog_fitness:
            self.cog_state = distant_cog_state
            self.cog_fitness = distant_cog_fitness
            return True
        else:
            return False

    def state_2_cog_state(self, state=None):
        cog_state = state.copy()
        return ["*" if index not in self.expertise_domain else bit for index, bit in enumerate(cog_state)]

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        return [np.random.choice(["0", "1", "2", "3"]) if bit == "*" else bit for bit in state]

    def describe(self):
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state: ", self.state)
        print("Current cognitive state: ", self.cog_state)
        print("Converged fitness: ", self.fitness)
        print("Current cognitive fitness: ", self.cog_fitness)
        print("Expertise domain: ", self.expertise_domain)


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    landscape.type(IM_type="Traditional Directed", K=4, k=0)
    landscape.initialize()
    specialist = Specialist(N=8, landscape=landscape, state_num=4, expertise_amount=16)
    # state = ["0", "1", "2", "3", "0", "1", "2", "3"]
    # cog_state = specialist.state_2_cog_state(state=state)
    # specialist.describe()
    # print(cog_state)
    jump_count = 0
    for _ in range(1000):
        specialist.search()
        if specialist.distant_jump():
            jump_count += 1
        # print(generalist.cog_fitness)
    print("jump_count: ", jump_count)
    specialist.state = specialist.cog_state_2_state(cog_state=specialist.cog_state)
    specialist.fitness = landscape.query_fitness(state=specialist.state)
    specialist.describe()
    print("END")

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]


