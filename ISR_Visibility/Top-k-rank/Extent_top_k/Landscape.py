# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Landscape.py
# @Software  : PyCharm
# Observing PEP 8 coding style

from collections import defaultdict
from itertools import product
import numpy as np


class Landscape:
    def __init__(self, N: int, K: int, norm="MaxMin"):
        """
        :param N:
        :param K:
        :param norm: normalization methods
        """
        self.N = N
        self.K = K
        self.IM, self.dependency_map = np.eye(self.N), [[]] * self.N
        self.FC = None

        self.local_optima = {}
        self.fitness_cache = {}
        self.fitness_rank_cache = {}

        self.max_normalizer, self.min_normalizer = 1, 0
        self.norm = norm
        self.initialize()  # Initialization and Normalization

    def create_IM(self):
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
            k = int(sum(self.IM[row]))  # typically k = K + 1

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

        return sum(result) / len(result)

    def store_cache(self):
        for state in product(["0", "1"], repeat=self.N):
            bits = "".join(state)
            self.fitness_cache[bits] = self.calculate_fitness(state)

    def store_rank_cache(self):
        """
        Store the rank of each solution on the binary NK landscape.

        Rank 1 corresponds to the highest-fitness solution.
        Ties receive the same competition rank, e.g., 1, 2, 2, 4.
        """
        self.fitness_rank_cache = {}

        sorted_items = sorted(self.fitness_cache.items(),
                              key=lambda item: item[1],
                              reverse=True)

        previous_fitness = None
        current_rank = 0

        for index, (state, fitness) in enumerate(sorted_items, start=1):
            if fitness != previous_fitness:
                current_rank = index
                previous_fitness = fitness

            self.fitness_rank_cache[state] = current_rank

    def initialize(self):
        self.create_IM()
        self.create_fitness_configuration()
        self.store_cache()

        self.max_normalizer = max(self.fitness_cache.values())
        self.min_normalizer = min(self.fitness_cache.values())

        # normalization
        if self.norm == "MaxMin":
            for key in self.fitness_cache.keys():
                self.fitness_cache[key] = (
                    (self.fitness_cache[key] - self.min_normalizer)
                    / (self.max_normalizer - self.min_normalizer)
                )

        self.store_rank_cache()

    def query_fitness(self, state: list) -> float:
        return self.fitness_cache["".join(state)]

    def query_fitness_rank(self, state: list) -> int:
        return self.fitness_rank_cache["".join(state)]

    def query_scoped_fitness(self, state: list, knowledge_domain: list) -> float:
        """
        Remove the fitness contribution of the unknown domains.

        Unknown domains still indirectly shape the known domains' fitness
        contributions through the NK interdependency structure.
        """
        scoped_fitness = []
        state = "".join(state)

        for row in knowledge_domain:
            dependency = self.dependency_map[row]
            bin_index = "".join([state[j] for j in dependency])
            bin_index = state[row] + bin_index
            index = int(bin_index, 2)
            scoped_fitness.append(self.FC[row][index])

        return sum(scoped_fitness) / len(scoped_fitness)

    def count_local_optima(self):
        counter = 0

        for key, value in self.fitness_cache.items():
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

    def get_neighbor_list(self, key: str) -> list:
        """
        :param key: string from the binary landscape cache, e.g., "0011"
        :return: list of neighboring states with one bit flipped
        """
        neighbor_states = []

        for i, char in enumerate(key):
            if char == "0":
                new_state = key[:i] + "1" + key[i + 1:]
            else:
                new_state = key[:i] + "0" + key[i + 1:]

            neighbor_states.append(new_state)

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

        for key, value in self.fitness_cache.items():
            print(key, value, "Rank:", self.fitness_rank_cache[key])
            break


if __name__ == '__main__':
    # Test Example
    import time

    t0 = time.time()

    N = 6
    K = 2

    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, norm="MaxMin")
    landscape.describe()

    max_state = max(landscape.fitness_cache, key=landscape.fitness_cache.get)
    min_state = min(landscape.fitness_cache, key=landscape.fitness_cache.get)

    print("Highest-fitness state:", max_state,
          "Fitness:", landscape.query_fitness(list(max_state)),
          "Rank:", landscape.query_fitness_rank(list(max_state)))

    print("Lowest-fitness state:", min_state,
          "Fitness:", landscape.query_fitness(list(min_state)),
          "Rank:", landscape.query_fitness_rank(list(min_state)))

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
