# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from itertools import product
import numpy as np


class Landscape:

    def __init__(self, N=None, K=None, state_num=4, norm=True):
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
        # Initialization
        self.initialize()

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

    def create_fitness_config(self,):
        FC = defaultdict(dict)
        for row in range(self.N):
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
        """
        Cache the fitness value
        :return: fitness cache
        """
        self.create_IM()
        self.create_fitness_config()
        self.store_cache()
        self.max_normalizer = max(self.cache.values())
        self.min_normalizer = min(self.cache.values())
        # normalization
        if self.norm:
            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - self.min_normalizer) / (self.max_normalizer - self.min_normalizer)
                # self.cache[k] = self.cache[k] / self.max_normalizer

    def query_fitness(self, state):
        return self.cache["".join(state)]

    def query_potential_fitness(self, cog_state=None):
        """
        Return the performance trajectory at the finest level;
        Show comparison for the trajectory at the coarse level
        :param cog_state:
        :return: potential performance of a given cog_state
        """
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        fitness_pool = [self.query_fitness(each) for each in alternatives]
        ave_fitness = sum(fitness_pool) / len(alternatives)
        return ave_fitness, max(fitness_pool), min(fitness_pool)

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

    def describe(self):
        print("*********LandScape information********* ")
        print("LandScape shape of (N={0}, K={1}, state number={2})".format(self.N, self.K, self.state_num))
        print("Influential matrix: \n", self.IM)
        print("Influential dependency map: ", self.dependency_map)
        print("Samples: \n")
        for key, value in landscape.cache.items():
            print(key, value)
            break
        print("********************************")


if __name__ == '__main__':
    # Test Example
    N = 9
    K = 8
    state_num = 4
    landscape = Landscape(N=N, K=K, state_num=state_num)
    landscape.describe()

    cog_state = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    ave_, max_, min_ = landscape.query_potential_fitness(cog_state=cog_state)

    import matplotlib.pyplot as plt
    data = landscape.cache.values()
    plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Range")
    plt.ylabel("Count")
    plt.show()