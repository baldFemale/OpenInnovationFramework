# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from itertools import product
import numpy as np


class Landscape:

    def __init__(self, N, state_num=4):
        self.N = N
        self.K = None
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.FC = None
        self.cache = {}  # state string to overall fitness: state_num ^ N: [1]
        self.max_normalizer = 1
        self.min_normalizer = 0
        self.norm = True
        self.fitness_to_rank_dict = None  # using the rank information to measure the potential performance of GST
        self.state_to_rank_dict = {}

    def describe(self):
        print("*********LandScape information********* ")
        print("LandScape shape of (N={0}, K={1}, state number={2})".format(self.N, self.K, self.state_num))
        print("Influential matrix: \n", self.IM)
        print("Influential dependency map: ", self.dependency_map)
        print("********************************")

    def type(self, K=0):
        """
        Characterize the influential matrix
        :param IM_type: "random", "dependent",
        :param K: mutual excitation dependency (undirected links)
        :param k: single-way dependency (directed links); k=2K for mutual dependency
        :return: the influential matrix (self.IM); and the dependency rule (self.IM_dict)
        """
        self.K = K
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

    def create_fitness_config(self,):
        FC = defaultdict(dict)
        for row in range(len(self.IM)):
            k = int(sum(self.IM[row]))
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

    def store_cache(self,):
        all_states = [state for state in product(range(self.state_num), repeat=self.N)]
        for state in all_states:
            bits = ''.join([str(bit) for bit in state])
            self.cache[bits] = self.calculate_fitness(state)

    def creat_fitness_rank_dict(self,):
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

    def initialize(self, norm=True):
        """
        Cache the fitness value
        :param norm: normalization
        :return: fitness cache
        """
        self.create_fitness_config()
        self.store_cache()
        self.norm = norm
        self.max_normalizer = max(self.cache.values())
        self.min_normalizer = min(self.cache.values())
        # normalization
        if self.norm:
            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - self.min_normalizer) / (self.max_normalizer - self.min_normalizer)
                # self.cache[k] = self.cache[k] / self.max_normalizer
        self.creat_fitness_rank_dict()

    def query_fitness(self, state):
        """
        Query the accurate fitness from the landscape cache for *intact* decision string
        *intact* means there only [0,1,2,3] without any ["A", "B", "*"] masking.
        """
        bits = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bits]

    def query_cog_fitness_full(self, cog_state=None):
        """
        Query the cognitive (average) fitness given a cognitive state
                For S domain, there is only one alternative, so it follows the default search
                For G domain, there is an alternative pool, so it takes the average of fitness across alternative states
        :param cog_state: the cognitive state
        :return: average, maximum, minimum of all alternative refined solutions
        """
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        fitness_pool = [self.query_fitness(each) for each in alternatives]
        cog_fitness = sum(fitness_pool) / len(alternatives)
        return cog_fitness, max(fitness_pool), min(fitness_pool)

    def query_cog_fitness_partial(self, cog_state=None, expertise_domain=None):
        """
        Query the cognitive (average) fitness given a cognitive state
                For S domain, there is only one alternative, so it follows the default search
                For G domain, there is an alternative pool, so it takes the average of fitness across alternative states.
        :param cog_state: the cognitive state
        :return: the average across the alternative pool.
        """
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        partial_fitness_alternatives = []
        for state in alternatives:
            partial_FC_across_bits = []  # only the expertise domains have fitness contribution
            for index in range(self.N):
                if index not in expertise_domain:
                    continue
                else:
                    # the unknown domain will still affect the condition
                    dependency = self.dependency_map[index]
                    bin_index = "".join([str(state[d]) for d in dependency])
                    bin_index = str(state[index]) + bin_index
                    FC_index = int(bin_index, self.state_num)
                    partial_FC_across_bits.append(self.FC[index][FC_index])
            # print("partial_FC_across_bits: ", partial_FC_across_bits)
            # Normalize it using the maximum
            partial_fitness_state = sum(partial_FC_across_bits) / len(expertise_domain) / max(partial_FC_across_bits)
            partial_fitness_alternatives.append(partial_fitness_state)
        return sum(partial_fitness_alternatives) / len(partial_fitness_alternatives)

    def cog_state_alternatives(self, cog_state=None):
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
        if self.state_num != 4:
            raise ValueError("Only support state_num = 4")
        return [i for i in product(*alternative_pool)]

    def generate_divergence_pool(self, divergence=None):
        """
        Randomly select one seed state, and form the pool around a given divergence being away from the seed
        For example
        1) random seed: 1 1 1 1 1 1 (N=6)
        2) 1 bit divergence: C_6^1 * 3 = 18 alternatives
        3) 2 bits divergence: C_6^2 * 3^2 = 15 * 9 = 135 alternatives
        4) In order to make the pool length the same across divergence, limit it into 18
        :param divergence: change how many bits to shape the pool
        :return:a list of pool
        """
        state_pool = []
        seed_state = np.random.choice(range(self.state_num), self.N).tolist()
        seed_state = [str(i) for i in seed_state]  # state format: string
        # print("seed_state: ", seed_state)
        if divergence == 1:
            for index in range(self.N):
                alternative_state = seed_state.copy()
                freedom_space = ["0", "1", "2", "3"]
                freedom_space.remove(seed_state[index])
                for bit in freedom_space:
                    alternative_state[index] = bit
                    state_pool.append(alternative_state.copy())
            return state_pool
        while True:
            index_for_change = np.random.choice(range(self.N), divergence, replace=False)
            alternative_state = seed_state.copy()
            for index in index_for_change:
                freedom_space = ["0", "1", "2", "3"]
                freedom_space.remove(seed_state[index])
                alternative_state[index] = freedom_space[np.random.choice(range(3))]
            if alternative_state not in state_pool:
                state_pool.append(alternative_state.copy())
            if len(state_pool) >= 18:
                break
        return state_pool

    def generate_quality_pool(self, quality_percentage=None):
        """
        Form the pool around a given quality percentage (e.g., 50% - 60%)
        :param quality:
        :return:
        """
        alternative_state_pool = []
        reference = quality_percentage * (self.N ** self.state_num)
        for state, rank in self.state_to_rank_dict.items():
            if abs(rank - reference) / (self.N ** self.state_num) <= 0.05:
                alternative_state_pool.append(state)
        result = np.random.choice(alternative_state_pool, 18, replace=False)  #this is a string state: "33300201", instead of list
        result = [list(each) for each in result]
        return result


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    landscape.type(K=7)
    landscape.initialize(norm=True)
    # landscape.describe()
    # list_cache = list(landscape.cache.values())
    # print("sd:", np.std(list_cache))

    cog_state = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    partial_pool, partial_alternative = landscape.query_cog_fitness_partial(cog_state=cog_state, expertise_domain=range(len(cog_state)))
    full_pool, full_alternative = landscape.query_cog_fitness_full(cog_state=cog_state)
    # for a, b in zip(partial_pool, full_pool):
    #     if a != b:
    #         print(a, b)
    for a, b in zip(partial_alternative, full_alternative):
        if a != b:
            print(a, b)

    # cog_fitness_pertial = landscape.query_cog_fitness_partial(cog_state=cog_state, expertise_domain=range(len(cog_state)))
    # # cog_fitness = landscape.query_cog_fitness_partial(cog_state=cog_state, expertise_domain=[1, 2, 3])
    # print("partial_cog_fitness: ", cog_fitness_pertial)
    # cog_fitness_full = landscape.query_cog_fitness_full(cog_state=cog_state)
    # print("full_cog_fitness: {0}; potential_fitness: {1}".format(cog_fitness_full[0],  cog_fitness_full[1]))
    # print("max_cache: ", max(landscape.cache.values()))

    # import matplotlib.pyplot as plt
    # data = landscape.cache.values()
    # plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("Range")
    # plt.ylabel("Count")
    # plt.show()