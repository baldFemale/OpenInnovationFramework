# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import random
import numpy as np
from Landscape import Landscape


class Specialist:
    def __init__(self, N=None, landscape=None, coarse_landscape=None, state_num=4, expertise_amount=None):
        self.landscape = landscape
        self.coarse_landscape = coarse_landscape
        self.N = N
        self.state_num = state_num
        self.expertise_domain = np.random.choice(range(self.N), expertise_amount // 4, replace=False).tolist()
        self.expertise_representation = ["0", "1", "2", "3"]
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: string
        self.coarse_state = self.state_2_coarse_state(state=self.state)  # will be the same as state, thus search accurately
        self.coarse_fitness = 0
        self.ave_fitness, self.max_fitness, self.min_fitness = 0, 0, 0
        # Mechanism: overlap with IM
        self.row_overlap = 0
        self.column_overlap = 0

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")
        if expertise_amount % 2 != 0:
            raise ValueError("Expertise amount needs to be a even number")
        if expertise_amount > self.N * 4:
            raise ValueError("Expertise amount should be less than {0}.".format(self.N * 4))

    def update_cog_fitness(self):
        self.coarse_fitness = self.coarse_landscape.query_coarse_fitness(coarse_state=self.coarse_state)
        self.ave_fitness, self.max_fitness, self.min_fitness = self.landscape.query_potential_fitness(coarse_state=self.coarse_state)

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
    #     self.coarse_state = self.state_2_coarse_state(state=self.state)
    #     self.coarse_fitness = self.landscape.query_cog_fitness_partial(coarse_state=self.coarse_state, expertise_domain=self.expertise_domain)
    #     self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(coarse_state=self.coarse_state)

    # def learn_from_pool(self, pool=None):
    #     exposure_state = pool[np.random.choice(len(pool))]
    #     cog_exposure_state = self.state_2_coarse_state(state=exposure_state)
    #     cog_fitness_of_exposure_state = self.landscape.query_cog_fitness_partial(coarse_state=cog_exposure_state, expertise_domain=self.expertise_domain)
    #     if cog_fitness_of_exposure_state > self.coarse_fitness:
    #         self.coarse_state = cog_exposure_state
    #         self.coarse_fitness = cog_fitness_of_exposure_state
    #         self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(coarse_state=self.coarse_state)
    #         return True
    #     return False

    def search(self):
        next_coarse_state = self.coarse_state.copy()
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.coarse_state[index])
        next_coarse_state[index] = np.random.choice(space)
        next_coarse_fitness = self.coarse_landscape.query_coarse_fitness(coarse_state=next_coarse_state)
        if next_coarse_fitness > self.coarse_fitness:
            self.coarse_state = next_coarse_state
            self.coarse_fitness = next_coarse_fitness
            self.ave_fitness, self.max_fitness, self.min_fitness = self.landscape.query_potential_fitness(
                coarse_state=self.coarse_state)

    def coordinated_search(self, co_state=None, co_expertise_domain=None):
        # the focal agent's evaluation: whether to align with the teammate
        next_coarse_state = self.coarse_state.copy()
        for index in range(self.N):
            if index in co_expertise_domain:
                next_coarse_state[index] = co_state[index]
        next_coarse_fitness = self.landscape.query_cog_fitness_partial(coarse_state=next_coarse_state,
                                                    expertise_domain=self.expertise_domain)
        if next_coarse_fitness > self.coarse_fitness:
            self.coarse_state = next_coarse_state.copy()
            self.coarse_fitness = next_coarse_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(coarse_state=self.coarse_state)

        # Proposal from the focal agent
        next_coarse_state = self.coarse_state.copy()
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.coarse_state[index])
        next_coarse_state[index] = np.random.choice(space)
        next_coarse_fitness = self.landscape.query_cog_fitness_partial(coarse_state=next_coarse_state, expertise_domain=self.expertise_domain)
        if next_coarse_fitness > self.coarse_fitness:
            self.coarse_state = next_coarse_state
            self.coarse_fitness = next_coarse_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(coarse_state=self.coarse_state)

    def priority_search(self, co_state=None, co_expertise_domain=None):
        # learning from coupled agent
        next_coarse_state = self.coarse_state.copy()
        for index in range(self.N):
            if index in co_expertise_domain:
                next_coarse_state[index] = co_state[index]
            else:
                pass
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.coarse_state[index])
        next_coarse_state[index] = np.random.choice(space)
        next_coarse_fitness = self.landscape.query_cog_fitness_partial(coarse_state=next_coarse_state,
                                                                    expertise_domain=self.expertise_domain)
        if next_coarse_fitness > self.coarse_fitness:
            self.coarse_state = next_coarse_state
            self.coarse_fitness = next_coarse_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(coarse_state=self.coarse_state)

    def state_2_coarse_state(self, state=None):
        state = state.copy()
        return [bit if i in self.expertise_domain else "*" for i, bit in enumerate(state)]
        # return state

    def cog_state_2_state(self, coarse_state=None):
        state = coarse_state.copy()
        return [np.random.choice(["0", "1", "2", "3"]) if bit == "*" else bit for bit in state]
        # return coarse_state

    def describe(self):
        print("N: ", self.N)
        print("Expertise domain: ", self.expertise_domain)
        print("Current state: ", self.state)
        print("State number: ", self.state_num)
        print("Current cognitive state: ", self.coarse_state)
        print("Average real fitness: ", self.ave_fitness)
        print("Max real fitness: ", self.max_fitness)
        print("Min real fitness: ", self.min_fitness)



if __name__ == '__main__':
    # Test Example
    from CogLandscape import CogLandscape
    import time
    t0 = time.time()
    search_iteration = 500
    N = 9
    K = 0
    state_num = 4
    expertise_amount = 32
    landscape = Landscape(N=N, K=K, state_num=state_num)
    specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
    coarse_landscape = CogLandscape(landscape=landscape, expertise_domain=specialist.expertise_domain,
                                 expertise_representation=specialist.expertise_representation)
    specialist.coarse_landscape = coarse_landscape
    specialist.update_cog_fitness()
    specialist.describe()
    ave_performance_across_time = []
    max_performance_across_time = []
    min_performance_across_time = []
    cog_performance_across_time = []
    for _ in range(search_iteration):
        specialist.search()
        print(specialist.coarse_state, specialist.ave_fitness)
        ave_performance_across_time.append(specialist.ave_fitness)
        max_performance_across_time.append(specialist.max_fitness)
        min_performance_across_time.append(specialist.min_fitness)
        cog_performance_across_time.append(specialist.coarse_fitness)
    # specialist.describe()
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(search_iteration)
    plt.plot(x, ave_performance_across_time, "r-", label="Ave")
    plt.plot(x, max_performance_across_time, "b-", label="Max")
    plt.plot(x, min_performance_across_time, "g-", label="Min")
    plt.plot(x, cog_performance_across_time, "k-", label="Cog")
    plt.title('Performance at N={0}, K={1}, Kn={2}'.format(N, K, expertise_amount))
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig("S_performance.png", transparent=True, dpi=200)
    plt.show()
    plt.clf()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


