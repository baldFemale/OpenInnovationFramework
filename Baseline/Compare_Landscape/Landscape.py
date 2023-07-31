# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from collections import defaultdict
from scipy.stats import rankdata
from itertools import product
import numpy as np


class Landscape:
    def __init__(self, N: int, K: int, state_num=4, norm="MaxMin", alpha=0.5):
        """
        :param N:
        :param K:
        :param state_num:
        :param norm: normalization methods
        :param alpha: the interval of refined space with respect to the binary shallow space
        """
        self.N = N
        self.K = K
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.alpha = alpha
        self.FC_1 = None
        self.FC_2 = None
        self.seed = None
        self.first_local_optima = {}
        self.second_local_optima = {}
        self.first_cache = {}  # state string to overall fitness: state_num ^ N: [1]
        self.second_cache = {}
        self.max_normalizer_1, self.min_normalizer_1 = 1, 0
        self.max_normalizer_2, self.min_normalizer_2 = 1, 0
        self.norm = norm
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

    def create_first_fitness_configuration(self):
        FC_1 = defaultdict(dict)
        for row in range(self.N):
            k = int(sum(self.IM[row]))  # typically k = K+1; for A, B combinations
            for column in range(pow(self.state_num // 2, k)):
                FC_1[row][column] = np.random.uniform(0, 1)
        self.FC_1 = FC_1

    def create_second_fitness_configuration(self):
        FC_2 = defaultdict(dict)
        translation_table = str.maketrans('01', 'AB')
        for row in range(self.N):
            k = int(sum(self.IM[row]))  # typically k = K+1; for 0123 combinations
            for key, value in self.FC_1[row].items():
                binary_value = bin(key)[2:].zfill(k)  # 0 -> "AAA"; 1-> "AAB"
                binary_value = binary_value.translate(translation_table)
                quaternary_value_list = self.cog_state_alternatives(cog_state=list(binary_value))
                for quaternary_value in quaternary_value_list:
                    quaternary_index = int("".join(quaternary_value), 4)
                    FC_2[row][quaternary_index] = np.random.uniform(value - self.alpha, value + self.alpha)
        self.FC_2 = FC_2

    def calculate_first_fitness(self, state: list) -> float:
        result = []
        state = "".join(state)
        translation_table = str.maketrans('AB', '01')  # "A" -> "0"; "B" -> "1"
        converted_string = state.translate(translation_table)  # decoded into "01"
        for i in range(self.N):
            dependency = self.dependency_map[i]
            bin_index = "".join([converted_string[j] for j in dependency])
            bin_index = converted_string[i] + bin_index
            index = int(bin_index, 2)
            result.append(self.FC_1[i][index])
        return sum(result) / len(result)

    def calculate_second_fitness(self, state: list) -> float:
        result = []
        state = "".join(state)
        for i in range(self.N):
            dependency = self.dependency_map[i]
            bin_index = "".join([state[j] for j in dependency])
            bin_index = state[i] + bin_index
            index = int(bin_index, self.state_num)
            result.append(self.FC_2[i][index])
        return sum(result) / len(result)

    def store_first_cache(self):
        all_states = [state for state in product(["A", "B"], repeat=self.N)]
        for state in all_states:
            bits = "".join([str(i) for i in state])
            self.first_cache[bits] = self.calculate_first_fitness(state)

    def store_second_cache(self):
        all_states = [state for state in product(["0", "1", "2", "3"], repeat=self.N)]
        for state in all_states:
            bits = "".join(state)
            self.second_cache[bits] = self.calculate_second_fitness(state)

    def initialize(self):
        self.create_IM()
        self.create_first_fitness_configuration()
        self.create_second_fitness_configuration()
        # self.create_skewed_fitness_configuration()
        self.store_first_cache()
        self.store_second_cache()
        self.max_normalizer_1 = max(self.first_cache.values())
        self.min_normalizer_1 = min(self.first_cache.values())
        self.max_normalizer_2 = max(self.second_cache.values())
        self.min_normalizer_2 = min(self.second_cache.values())
        # normalization
        if self.norm == "MaxMin":
            for k in self.first_cache.keys():
                self.first_cache[k] = (self.first_cache[k] - self.min_normalizer_1) / (self.max_normalizer_1 - self.min_normalizer_1)
            for k in self.second_cache.keys():
                self.second_cache[k] = (self.second_cache[k] - self.min_normalizer_2) / (self.max_normalizer_2 - self.min_normalizer_2)
        # elif self.norm == "Max":
        #     for k in self.first_cache.keys():
        #         self.first_cache[k] = self.first_cache[k] / self.max_normalizer
        # elif self.norm == "RangeScaling":
        #     # Single center, inspired by March's model
        #     seed = np.random.choice(range(self.state_num), self.N).tolist()
        #     seed = [str(i) for i in seed]
        #     self.seed = seed
        #     high_area, low_area = {}, {}
        #     for key in self.cache.keys():
        #         if key > "200000000":
        #             high_area[key] = 1
        #         else:
        #             low_area[key] = 1
        #     high_area_fitness_list = [self.cache[key] for key in high_area.keys()]
        #     min_fitness_high = min(high_area_fitness_list)
        #     max_fitness_high = max(high_area_fitness_list)
        #     high_top, high_bottom = 1.0, 0.5
        #     low_area_fitness_list = [self.cache[key] for key in low_area.keys()]
        #     min_fitness_low = min(low_area_fitness_list)
        #     max_fitness_low = max(low_area_fitness_list)
        #     low_top, low_bottom = 0.5, 0
        #     for key in self.cache.keys():
        #         if key in high_area.keys():
        #             self.cache[key] = (self.cache[key] - min_fitness_high) * (high_top - high_bottom) / \
        #                               (max_fitness_high - min_fitness_high) + high_bottom
        #         else:
        #             self.cache[key] = (self.cache[key] - min_fitness_low) * (low_top - low_bottom) / \
        #                               (max_fitness_low - min_fitness_low) + low_bottom
        #
        # elif self.norm == "ClusterRangeScaling":
        #     cluster_0, cluster_1, cluster_2, cluster_3 = {}, {}, {}, {}
        #     for key in self.cache.keys():
        #         number_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
        #         for char in key:
        #             if char in number_counts:
        #                 number_counts[char] += 1
        #         max_count = max(number_counts.values())
        #         most_frequent_numbers = [number for number, count in number_counts.items() if count == max_count]
        #         most_frequent_numbers = most_frequent_numbers[0]
        #         if most_frequent_numbers == "0":
        #             cluster_0[key] = True
        #         elif most_frequent_numbers == "1":
        #             cluster_1[key] = True
        #         elif most_frequent_numbers == "2":
        #             cluster_2[key] = True
        #         else:
        #             cluster_3[key] = True
        #     fitness_list_0 = [self.cache[key] for key in cluster_0.keys()]
        #     fitness_list_1 = [self.cache[key] for key in cluster_1.keys()]
        #     fitness_list_2 = [self.cache[key] for key in cluster_2.keys()]
        #     fitness_list_3 = [self.cache[key] for key in cluster_3.keys()]
        #     # max_0, min_0 = max(fitness_list_0), min(fitness_list_0)
        #     # max_1, min_1 = max(fitness_list_1), min(fitness_list_1)
        #     # max_2, min_2 = max(fitness_list_2), min(fitness_list_2)
        #     # max_3, min_3 = max(fitness_list_3), min(fitness_list_3)
        #
        #     # Only two clusters: low and high type
        #     max_low, min_low = max(fitness_list_0 + fitness_list_1), min(fitness_list_0 + fitness_list_1)
        #     max_high, min_high = max(fitness_list_2 + fitness_list_3), min(fitness_list_2 + fitness_list_3)
        #     for key in self.cache.keys():
        #         if (key in cluster_0) or (key in cluster_1):
        #             self.cache[key] = (self.cache[key] - min_low) * 0.5 / \
        #                               (max_low - min_low)
        #         else:
        #             self.cache[key] = (self.cache[key] - min_high) * 0.5 / \
        #                               (max_high - min_high) + 0.50

    def query_first_fitness(self, state: list) -> float:
        return self.first_cache["".join(state)]

    def query_scoped_first_fitness(self, cog_state: list, state: list) -> float:
        """
        Remove the fitness contribution of the unknown domain;
        But the unknown domains indirectly contribute to other elements' contributions via interdependency
        :param cog_state: state of "AB" with unknown shelter
        :param state: original "AB"
        :return: partial fitness on the shallow landscape
        """
        translation_table = str.maketrans('AB', '01')
        scoped_fitness = []
        for row, bit in enumerate(cog_state):
            if bit == "*":
                continue
            dependency = self.dependency_map[row]
            bin_index = "".join([state[j] for j in dependency])  # the unknown domain will shape the contingency condition
            bin_index = state[row] + bin_index
            bin_index = bin_index.translate(translation_table)  # "AB" to "01"
            index = int(bin_index, 2)
            scoped_fitness.append(self.FC_1[row][index])  # not need to normalize; does not affect the local search
        return sum(scoped_fitness) / len(scoped_fitness)

    def query_second_fitness(self, state: list) -> float:
        return self.second_cache["".join(state)]

    def query_scoped_second_fitness(self, cog_state: list, state: list) -> float:
        scoped_fitness = []
        for row, bit in enumerate(cog_state):
            if bit == "*":
                continue
            dependency = self.dependency_map[row]
            qua_index = "".join([state[j] for j in dependency])  # the unknown domain will shape the contingency condition
            qua_index = state[row] + qua_index
            index = int(qua_index, 4)
            scoped_fitness.append(self.FC_2[row][index])  # not need to normalize; does not affect the local search
        return sum(scoped_fitness) / len(scoped_fitness)

    # def query_cog_fitness(self, cog_state: list, state: list) -> float:
    #     cog_fitness = 0
    #     for row, cog_bit in enumerate(cog_state):
    #         dependency = self.dependency_map[row]
    #         bin_index = "".join([str(state[j]) for j in dependency])
    #         if cog_bit == "A":
    #             bin_index_1 = "0" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "1" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
    #         elif cog_bit == "B":
    #             bin_index_1 = "2" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "3" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
    #         elif cog_bit == "*":
    #             bin_index_0 = "0" + bin_index
    #             index_0 = int(bin_index_0, self.state_num)
    #             bin_index_1 = "1" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "2" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             bin_index_3 = "3" + bin_index
    #             index_3 = int(bin_index_3, self.state_num)
    #             c_i = (self.FC[row][index_0] + self.FC[row][index_1] +
    #                    self.FC[row][index_2] + self.FC[row][index_3]) / 4
    #         else:
    #             bin_index = cog_bit + bin_index
    #             index = int(bin_index, self.state_num)
    #             c_i = self.FC[row][index]
    #         cog_fitness += c_i
    #     return cog_fitness / self.N

    # def query_partial_fitness(self, cog_state: list, state: list, expertise_domain: list) -> float:
    #     cog_fitness = 0
    #     for row, cog_bit in enumerate(cog_state):
    #         if row not in expertise_domain:
    #             continue
    #         dependency = self.dependency_map[row]
    #         bin_index = "".join([str(state[j]) for j in dependency])
    #         if cog_bit == "A":
    #             bin_index_1 = "0" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "1" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
    #         elif cog_bit == "B":
    #             bin_index_1 = "2" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "3" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
    #         elif cog_bit == "*":
    #             bin_index_0 = "0" + bin_index
    #             index_0 = int(bin_index_0, self.state_num)
    #             bin_index_1 = "1" + bin_index
    #             index_1 = int(bin_index_1, self.state_num)
    #             bin_index_2 = "2" + bin_index
    #             index_2 = int(bin_index_2, self.state_num)
    #             bin_index_3 = "3" + bin_index
    #             index_3 = int(bin_index_3, self.state_num)
    #             c_i = (self.FC[row][index_0] + self.FC[row][index_1] +
    #                    self.FC[row][index_2] + self.FC[row][index_3]) / 4
    #         else:
    #             bin_index = cog_bit + bin_index
    #             index = int(bin_index, self.state_num)
    #             c_i = self.FC[row][index]
    #         cog_fitness += c_i
    #     return cog_fitness / len(expertise_domain)

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

    def create_first_fitness_rank(self):
        fitness_cache = list(self.first_cache.values())
        ranks = rankdata(fitness_cache)
        ranks = [int(each) for each in ranks]
        rank_dict = {key: rank for key, rank in zip(self.first_cache.keys(), ranks)}
        return rank_dict

    def create_second_fitness_rank(self):
        fitness_cache = list(self.second_cache.values())
        ranks = rankdata(fitness_cache)
        ranks = [int(each) for each in ranks]
        rank_dict = {key: rank for key, rank in zip(self.second_cache.keys(), ranks)}
        return rank_dict

    def count_first_local_optima(self):
        counter = 0
        for key, value in self.first_cache.items():
            neighbor_list = self.get_neighbor_list(key=key)
            is_local_optima = True
            for neighbor in neighbor_list:
                if self.query_first_fitness(state=list(neighbor)) > value:
                    is_local_optima = False
                    break
            if is_local_optima:
                counter += 1
                self.first_local_optima[key] = value
        return counter

    def count_second_local_optima(self):
        counter = 0
        for key, value in self.second_cache.items():
            neighbor_list = self.get_neighbor_list(key=key)
            is_local_optima = True
            for neighbor in neighbor_list:
                if self.query_second_fitness(state=list(neighbor)) > value:
                    is_local_optima = False
                    break
            if is_local_optima:
                counter += 1
                self.second_local_optima[key] = value
        return counter

    def calculate_first_avg_fitness_distance(self):
        total_distance = 0
        total_neighbors = 0
        for key in self.first_cache.keys():
            neighbors = self.get_first_neighbor_list(key)
            total_distance += sum(abs(self.first_cache[key] - self.first_cache[neighbor]) for neighbor in neighbors)
            total_neighbors += len(neighbors)
        avg_fitness_distance = total_distance / total_neighbors
        return avg_fitness_distance

    def calculate_second_avg_fitness_distance(self):
        total_distance = 0
        total_neighbors = 0
        for key in self.second_cache.keys():
            neighbors = self.get_second_neighbor_list(key)
            total_distance += sum(abs(self.second_cache[key] - self.second_cache[neighbor]) for neighbor in neighbors)
            total_neighbors += len(neighbors)
        avg_fitness_distance = total_distance / total_neighbors
        return avg_fitness_distance

    def get_first_neighbor_list(self, key: str) -> list:
        neighbor_states = []
        for i, char in enumerate(key):  # "AB" string
            neighbors = []
            for neighbor in ["A", "B"]:
                if neighbor != char:
                    new_state = key[:i] + str(neighbor) + key[i + 1:]
                    neighbors.append(new_state)
            neighbor_states.extend(neighbors)
        return neighbor_states

    def get_second_neighbor_list(self, key: str) -> list:
        neighbor_states = []
        for i, char in enumerate(key):  # "0123" string
            neighbors = []
            for neighbor in ["0", "1", "2", "3"]:
                if neighbor != char:
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
        for key, value in self.first_cache.items():
            print(key, value)
            break
        for key, value in self.second_cache.items():
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
    K = 9
    state_num = 4
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin")
    landscape.describe()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
    # landscape.create_fitness_rank()
    # landscape.count_local_optima()
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

    plt.hist(landscape.first_cache.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.title("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.title("First Cache")
    plt.xlabel("Range")
    plt.ylabel("Count")
    # plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.show()

    plt.hist(landscape.second_cache.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.title("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.title("Second Cache")
    plt.xlabel("Range")
    plt.ylabel("Count")
    # plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.show()

