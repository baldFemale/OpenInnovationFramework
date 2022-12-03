# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
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
        self.cog_state = self.state_2_cog_state(state=self.state)  # will be the same as state, thus search accurately
        self.cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=self.cog_state, expertise_domain=self.expertise_domain)
        self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

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

    def get_overlap_with_IM(self):
        influence_matrix = self.landscape.IM
        row_overlap, column_overlap = 0, 0
        for row in range(self.N):
            if row in self.expertise_domain:
                row_overlap += sum(influence_matrix[row])
        for column in range(self.N):
            if column in self.expertise_domain:
                column_overlap += sum(influence_matrix[:, column])
        self.column_overlap = column_overlap
        self.row_overlap = row_overlap

    def align_default_state(self, state=None):
        for index in range(self.N):
            if index not in self.expertise_domain:
                self.state[index] = state[index]
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=self.cog_state, expertise_domain=self.expertise_domain)
        self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def learn_from_pool(self, pool=None):
        exposure_state = pool[np.random.choice(len(pool))]
        cog_exposure_state = self.state_2_cog_state(state=exposure_state)
        cog_fitness_of_exposure_state = self.landscape.query_cog_fitness_partial(cog_state=cog_exposure_state, expertise_domain=self.expertise_domain)
        if cog_fitness_of_exposure_state > self.cog_fitness:
            self.cog_state = cog_exposure_state
            self.cog_fitness = cog_fitness_of_exposure_state
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)
            return True
        return False

    def search(self):
        next_cog_state = self.cog_state.copy()
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.state[index])
        next_cog_state[index] = np.random.choice(space)
        next_cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_cog_state, expertise_domain=self.expertise_domain)
        if next_cog_fitness > self.cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def double_search(self, co_state=None, co_expertise_domain=None):
        # learning from coupled agent
        next_cog_state = self.cog_state.copy()
        for index in range(self.N):
            if index in self.expertise_domain:
                if index in co_expertise_domain:
                    changed_cog_state = next_cog_state.copy()
                    changed_cog_state[index] = co_state[index]
                    if self.landscape.query_cog_fitness_partial(cog_state=changed_cog_state, expertise_domain=self.expertise_domain) > self.cog_fitness:
                        next_cog_state[index] = co_state[index]
                else:  # retain the private configuration
                    pass
            else:
                # for unknown domains, follow the co-state
                if index in co_expertise_domain:
                    next_cog_state[index] = co_state[index]
                # for unknown domains to both agents, retain the private configuration
                else:
                    pass
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.state[index])
        next_cog_state[index] = np.random.choice(space)
        next_cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_cog_state, expertise_domain=self.expertise_domain)
        if next_cog_fitness > self.cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def priority_search(self, co_state=None, co_expertise_domain=None):
        # learning from coupled agent
        next_cog_state = self.cog_state.copy()
        for index in range(self.N):
            if index in co_expertise_domain:
                next_cog_state[index] = co_state[index]
            else:
                pass
        index = np.random.choice(self.expertise_domain)  # only select from the expertise domain,
        # thus will not change the unknown domain
        space = ["0", "1", "2", "3"]
        space.remove(self.state[index])
        next_cog_state[index] = np.random.choice(space)
        next_cog_fitness = self.landscape.query_cog_fitness_partial(cog_state=next_cog_state,
                                                                    expertise_domain=self.expertise_domain)
        if next_cog_fitness > self.cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness, self.potential_fitness = self.landscape.query_cog_fitness_full(cog_state=self.cog_state)

    def state_2_cog_state(self, state=None):
        return state
        # cog_state = state.copy()
        # return [bit for bit in cog_state]

    def cog_state_2_state(self, cog_state=None):
        # state = cog_state.copy()
        # return [np.random.choice(["0", "1", "2", "3"]) if bit == "*" else bit for bit in state]
        return cog_state

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
    search_iteration = 200
    landscape = Landscape(N=9, state_num=4)
    landscape.type(K=4)
    landscape.initialize()
    specialist = Specialist(N=9, landscape=landscape, state_num=4, expertise_amount=12)
    # state = ["0", "1", "2", "3", "0", "1", "2", "3"]
    # cog_state = specialist.state_2_cog_state(state=state)
    # specialist.describe()
    # print(cog_state)
    performance_across_time = []
    cog_performance_across_time = []
    for _ in range(search_iteration):
        specialist.search()
        performance_across_time.append(specialist.fitness)
        cog_performance_across_time.append(specialist.cog_fitness)
    specialist.describe()
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(search_iteration)
    plt.plot(x, performance_across_time, "k-", label="Absolute Fitness")
    plt.plot(x, cog_performance_across_time, "k--", label="Partial Fitness")
    # plt.title('Diversity Decrease')
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, ncol=3, fontsize=10)
    # plt.savefig("S_performance.png", transparent=True, dpi=200)
    plt.show()
    plt.clf()
    print("END")

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]


