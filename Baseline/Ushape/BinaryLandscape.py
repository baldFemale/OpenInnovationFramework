# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from collections import defaultdict
from itertools import product
import numpy as np


class BinaryLandscape:
    def __init__(self, N: int, K: int, norm="MaxMin"):
        """
        :param N:
        :param K:
        :param state_num:
        :param norm: normalization methods
        :param alpha: the interval of refined space with respect to the binary shallow space
        """
        self.N = N
        self.K = K
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.FC = None
        self.cache = {}
        self.max_normalizer, self.min_normalizer = 1, 0
        self.norm = norm
        self.local_optima = {}
        self.fitness_to_rank_dict = None  # using the rank information to measure the potential performance of GST
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

    def create_fitness_configuration(self):
        FC = defaultdict(dict)
        for row in range(self.N):
            k = int(sum(self.IM[row]))  # typically k = K+1; for A, B combinations
            for column in range(pow(2, k)):
                FC[row][column] = np.random.uniform(0, 1)
        self.FC = FC

    def calculate_fitness(self, state: list) -> float:
        result = []
        state = "".join(state)
        for i in range(self.N):
            dependency = self.dependency_map[i]
            bin_index = "".join([state[j] for j in dependency])
            bin_index = state[i] + bin_index
            index = int(bin_index, 2)
            result.append(self.FC[i][index])
        return sum(result) / self.N

    def store_cache(self):
        all_states = [state for state in product(["0", "1"], repeat=self.N)]
        for state in all_states:
            bits = "".join([i for i in state])
            self.cache[bits] = self.calculate_fitness(state)

    def initialize(self):
        self.create_IM()
        self.create_fitness_configuration()
        # self.create_skewed_fitness_configuration()
        self.store_cache()
        self.max_normalizer = max(self.cache.values())
        self.min_normalizer = min(self.cache.values())
        # normalization
        if self.norm == "MaxMin":
            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - self.min_normalizer) / (self.max_normalizer - self.min_normalizer)

    def query_fitness(self, state: list) -> float:
        return self.cache["".join(state)]

    def query_scoped_fitness(self, cog_state: list, state: list) -> float:
        """
        Remove the fitness contribution of the unknown domain;
        But the unknown domains indirectly contribute to other elements' contributions via interdependency
        :param cog_state: state of "AB" with unknown shelter
        :param state: original "0123"
        :return: partial fitness on the shallow landscape
        """
        scoped_fitness = []
        for row, bit in enumerate(cog_state):
            if bit == "*":
                continue
            dependency = self.dependency_map[row]
            bin_index = "".join([state[j] for j in dependency])  # the unknown domain will shape the contingency condition
            bin_index = state[row] + bin_index
            index = int(bin_index, 2)
            scoped_fitness.append(self.FC[row][index])  # not need to normalize; does not affect the local search
        return sum(scoped_fitness) / len(scoped_fitness)

    @staticmethod
    def cog_state_alternatives(cog_state: list) -> list:
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

    def count_local_optima(self):
        counter = 0
        for key, value in self.cache.items():
            neighbor_list = self.get_neighbor_list(key=key)
            is_local_optima = True
            for neighbor in neighbor_list:
                if self.query_fitness(state=list(neighbor)) > value:
                    is_local_optima = False
                    break
            if is_local_optima:
                counter += 1
                self.local_optima[key] = value
        return counter

    # def calculate_avg_fitness_distance(self):
    #     total_distance = 0
    #     total_neighbors = 0
    #     for key in self.first_cache.keys():
    #         neighbors = self.get_neighbor_list(key)
    #         total_distance += sum(abs(self.first_cache[key] - self.first_cache[neighbor]) for neighbor in neighbors)
    #         total_neighbors += len(neighbors)
    #     avg_fitness_distance = total_distance / total_neighbors
    #     return avg_fitness_distance

    def get_neighbor_list(self, key: str) -> list:
        """
        This is also for the Coarse Landscape
        :param key: string from the coarse landscape cache dict, e.g., "0011"
        :return:list of the neighbor state, e.g., [["0", "0", "1", "2"], ["0", "0", "1", "3"]]
        """
        neighbor_states = []
        for i, char in enumerate(key):
            neighbors = []
            for neighbor in [0, 1]:
                if neighbor != int(char):
                    new_state = key[:i] + str(neighbor) + key[i + 1:]
                    neighbors.append(new_state)
            neighbor_states.extend(neighbors)
        return neighbor_states

    @staticmethod
    def get_hamming_distance(state_1: list, state_2: list) -> int:
        distance = 0
        for a, b in zip(state_1, state_2):
            if a != b:
                distance += 1
        return distance

    def describe(self):
        print("LandScape shape of N={0}, K={1}".format(self.N, self.K))
        print("Influential Matrix: \n", self.IM)
        print("Influential Dependency Map: ", self.dependency_map)
        print("Cache Samples:")
        for key, value in self.cache.items():
            print(key, value)
            break
        # for seed_state in self.seed_state_list:
        #     print(seed_state, self.query_fitness(state=seed_state))
        #     for i in range(self.N):
        #         dependency = self.dependency_map[i]
        #         bin_index = "".join([str(seed_state[j]) for j in dependency])
        #         bin_index = str(seed_state[i]) + bin_index
        #         index = int(bin_index, self.state_num)
        #         print("Component: ", self.FC[i][index])
        #     break
        # neighbors = [['0', '3', '3', '3', '3', '3', '3', '3', '3'], ['1', '3', '3', '3', '3', '3', '3', '3', '3'],
        #              ['2', '3', '3', '3', '3', '3', '3', '3', '3'], ['3', '0', '0', '3', '3', '3', '3', '3', '3']]
        # for neighbor_state in neighbors:
        #     print(neighbor_state, self.query_fitness(state=neighbor_state))


if __name__ == '__main__':
    # Test Example
    import time
    t0 = time.time()
    N = 10
    K = 3
    state_num = 4
    np.random.seed(1000)
    landscape = BinaryLandscape(N=N, K=K, norm="MaxMin")
    landscape.describe()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
    # landscape.create_fitness_rank()
    landscape.count_local_optima()
    print("local peak: ", landscape.local_optima)
    # ave_distance = landscape.calculate_avg_fitness_distance()
    # ave_distance = round(ave_distance, 4)
    # print(landscape.local_optima)
    # print("Number of Local Optima: ", len(landscape.local_optima.keys()))
    # print("Average Distance: ", ave_distance)

    import matplotlib.pyplot as plt
    # plt.hist(landscape.local_optima.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("Range")
    # plt.ylabel("Count")
    # plt.title("Local Optima N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.savefig("Local Optima N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.show()
    # plt.clf()

    plt.hist(landscape.cache.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.title("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.title("First Cache")
    plt.xlabel("Range")
    plt.ylabel("Count")
    # plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.show()

