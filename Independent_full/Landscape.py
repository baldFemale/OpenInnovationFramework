# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import product
import numpy as np


class Landscape:

    def __init__(self, N=None, K=None, state_num=4, norm="MaxMin"):
        self.N = N
        self.K = K
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.FC = None
        self.cache = {}  # state string to overall fitness: state_num ^ N: [1]
        self.max_normalizer = 1
        self.min_normalizer = 0
        self.norm = norm
        self.fitness_to_rank_dict = None  # using the rank information to measure the potential performance of GST
        self.state_to_rank_dict = {}
        self.initialize()  # Initialization and Normalization

    def create_IM(self):
        self.K = self.K
        if self.K == 0:
            self.IM = np.eye(self.N)
        elif self.K >= (self.N - 1):
            self.K = self.N - 1
            self.IM = np.ones((self.N, self.N))
        else:
            # each row has a fixed number of dependency (i.e., K)
            for i in range(self.N):
                probs = [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
                ids = np.random.choice(self.N, self.K, p=probs, replace=False)
                for index in ids:
                    self.IM[i][index] = 1
        for i in range(self.N):
            temp = []
            for j in range(self.N):
                if (i != j) & (self.IM[i][j] == 1):
                    temp.append(j)
            self.dependency_map[i] = temp

    def create_fitness_configuration(self,):
        FC = defaultdict(dict)
        for row in range(self.N):
            k = int(sum(self.IM[row]))  # typically k = K+1
            for column in range(pow(self.state_num, k)):
                FC[row][column] = np.random.uniform(0, 1)
        self.FC = FC

    def calculate_fitness(self, state):
        result = []
        for i in range(len(state)):
            dependency = self.dependency_map[i]
            bin_index = "".join([str(state[j]) for j in dependency])
            bin_index = str(state[i]) + bin_index
            index = int(bin_index, self.state_num)
            result.append(self.FC[i][index])
        return sum(result) / len(result)

    def store_cache(self):
        all_states = [state for state in product(range(self.state_num), repeat=self.N)]
        for state in all_states:
            bits = "".join([str(i) for i in state])
            self.cache[bits] = self.calculate_fitness(state)

    def creat_fitness_rank_dict(self):
        """
        Sort the cache fitness value and corresponding rank
        To get another performance indicator regarding the reaching rate of relatively high fitness (e.g., the top 10 %)
        """
        value_list = sorted(list(self.cache.values()), key=lambda x: -x)
        fitness_to_rank_dict = {}
        state_to_rank_dict = {}
        for index, value in enumerate(value_list):
            fitness_to_rank_dict[value] = index + 1
        for state, fitness in self.cache.items():
            state_to_rank_dict[state] = fitness_to_rank_dict[fitness]
        self.state_to_rank_dict = state_to_rank_dict
        self.fitness_to_rank_dict = fitness_to_rank_dict

    def initialize(self):
        self.create_IM()
        self.create_fitness_configuration()
        self.store_cache()
        self.max_normalizer = max(self.cache.values())
        self.min_normalizer = min(self.cache.values())
        # normalization
        if self.norm == "MaxMin":
            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - self.min_normalizer) / (self.max_normalizer - self.min_normalizer)
        elif self.norm == "Max":
            for k in self.cache.keys():
                self.cache[k] = self.cache[k] / self.max_normalizer

    def query_fitness(self, state):
        return self.cache["".join(state)]

    def query_cog_fitness(self, cog_state=None):
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        fitness_pool = [self.query_fitness(each) for each in alternatives]
        ave_fitness = sum(fitness_pool) / len(alternatives)
        return ave_fitness

    def query_partial_fitness(self, cog_state=None, expertise_domain=None):
        partial_fitness_list = []
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        for state in alternatives:
            partial_FC_across_bits = []
            for index in range(len(state)):
                if index not in expertise_domain:
                    continue
                dependency = self.dependency_map[index]
                bit_index = "".join([str(state[j]) for j in dependency])
                bit_index = str(state[index]) + bit_index
                FC_index = int(bit_index, self.state_num)
                partial_FC_across_bits.append(self.FC[index][FC_index])
            partial_fitness_state = sum(partial_FC_across_bits) / len(partial_FC_across_bits)
            partial_fitness_list.append(partial_fitness_state)
        return sum(partial_fitness_list) / len(partial_fitness_list)

    @staticmethod
    def cog_state_alternatives(cog_state=None):
        alternative_pool = []
        for bit in cog_state:
            if bit in ["0", "1", "2", "3"]:
                alternative_pool.append(bit)
            elif bit == "A":
                alternative_pool.append(["0", "1"])
            elif bit == "B":
                alternative_pool.append(["2", "3"])
            elif bit == "*":
                alternative_pool.append(["0", "1", "2", "3"])
            else:
                raise ValueError("Unsupported bit value: ", bit)
        return [i for i in product(*alternative_pool)]

    def describe(self):
        print("LandScape shape of N={0}, K={1}".format(self.N, self.K))
        print("Influential Matrix: \n", self.IM)
        print("Influential Dependency Map: ", self.dependency_map)
        print("Cache Samples:")
        for key, value in landscape.cache.items():
            print(key, value)
            break


if __name__ == '__main__':
    # Test Example
    N = 9
    K = 2
    state_num = 4
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num)
    # print(landscape.FC[0])
    # cog_state = ['A', 'A', 'A', 'A', 'A', 'A']
    # cog_state = ["0", "0", "0", "0", "0", "0"]
    # print(landscape.cog_state_alternatives(cog_state=cog_state))
    state_1_0 = ['2', '2', '2', '1', '1', '2', '1', '3', '0']
    state_1_1 = ['2', '2', '2', '1', '1', '2', '1', '3', '1']
    state_1_2 = ['2', '2', '2', '1', '1', '2', '1', '3', '2']
    state_1_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '3']
    state_2 = ['B', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'A']
    state_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '*']
    partial_fitness_1_0 = landscape.query_partial_fitness(cog_state=state_1_0, expertise_domain=range(0, 9))
    partial_fitness_1_1 = landscape.query_partial_fitness(cog_state=state_1_1, expertise_domain=range(0, 9))
    partial_fitness_1_2 = landscape.query_partial_fitness(cog_state=state_1_2, expertise_domain=range(0, 9))
    partial_fitness_1_3 = landscape.query_partial_fitness(cog_state=state_1_3, expertise_domain=range(0, 9))
    partial_fitness_2 = landscape.query_partial_fitness(cog_state=state_2, expertise_domain=range(0, 9))
    partial_fitness_3 = landscape.query_partial_fitness(cog_state=state_3, expertise_domain=range(0, 9))
    print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_0, partial_fitness_2))
    print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_1, partial_fitness_2))
    print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_2, partial_fitness_2))
    print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_3, partial_fitness_2))
    fine_state_fitness = [partial_fitness_1_0, partial_fitness_1_1, partial_fitness_1_2, partial_fitness_1_3]
    print("Fine One {0} should be equal to Average of Coarse Ones {1}".format(partial_fitness_3, sum(fine_state_fitness) / len(fine_state_fitness)))
    landscape.describe()

    import matplotlib.pyplot as plt
    data = landscape.cache.values()
    plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.title("Landscape Distribution")
    plt.xlabel("Range")
    plt.ylabel("Count")
    plt.show()

