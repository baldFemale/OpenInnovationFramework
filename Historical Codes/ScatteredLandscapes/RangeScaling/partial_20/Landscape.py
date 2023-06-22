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
    def __init__(self, N: int, K: int, state_num=4, norm="MaxMin"):
        """
        :param N: problem dimension
        :param K: complexity degree
        :param state_num: state number for each dimension
        :param norm: normalization manner
        """
        self.N = N
        self.K = K
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.FC = None
        self.seed = None
        self.local_optima = {}
        self.cache = {}  # state string to overall fitness: state_num ^ N: [1]
        self.max_normalizer = 1
        self.min_normalizer = 0
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

    def create_fitness_configuration(self):
        FC = defaultdict(dict)
        for row in range(self.N):
            k = int(sum(self.IM[row]))  # typically k = K+1
            for column in range(pow(self.state_num, k)):
                FC[row][column] = np.random.uniform(0, 1)
        self.FC = FC

    # def create_skewed_fitness_configuration(self):
        # skewed_seed_num = 1
        # seed_list = []
        # for _ in range(skewed_seed_num):
        #     seed_state = np.random.choice(range(self.state_num), self.N).tolist()
        #     seed_state = [str(i) for i in seed_state]
        #     seed_list.append(seed_state)
        # seed_list = [['3', '3', '3', '3', '3', '3', '3', '3', '3']]  # Pre-define a global peak, similar to March's (1991) model
        # according to the distance between seed the any focal position, rescale the fitness value to shape gradient
        # self.seed_state_list = seed_list
        # FC = defaultdict(dict)
        # for row in range(self.N):
        #     k = int(sum(self.IM[row]))  # typically k = K+1
        #     for column in range(pow(self.state_num, k)):
        #         FC[row][column] = np.random.uniform(-1, 0)
        # for seed_state in seed_list:
        #     for row in range(self.N):
        #         dependency = self.dependency_map[row]
        #         bin_index = "".join([seed_state[j] for j in dependency])
        #         bin_index = seed_state[row] + bin_index
        #         index = int(bin_index, self.state_num)
        #         value = np.random.uniform(0, 1)
        #         FC[row][index] = value
        # self.FC = FC

    # def create_skewed_fitness_configuration(self):
    #     FC = defaultdict(dict)
    #     for row in range(self.N):
    #         k = int(sum(self.IM[row]))  # typically k = K+1
    #         for column in range(pow(self.state_num, k)):
    #             FC[row][column] = np.random.uniform(0, 1)
            # for column in range(pow(self.state_num, k)):
            #     if column < 4 ** self.K:  # for state 0 & 1
            #         FC[row][column] = np.random.uniform(-1, 0)
            #     elif column < 2 * 4 ** self.K:
            #         FC[row][column] = np.random.uniform(0, 1)
            #     elif column < 3 * 4 ** self.K:
            #         FC[row][column] = np.random.uniform(1, 2)
            #     else:
            #         FC[row][column] = np.random.uniform(2, 3)
        # self.FC = FC

    def calculate_fitness(self, state: list) -> float:
        result = []
        for i in range(self.N):
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
        elif self.norm == "Max":
            for k in self.cache.keys():
                self.cache[k] = self.cache[k] / self.max_normalizer
        elif self.norm == "RangeScaling":
            # Single center, inspired by March's model
            seed = np.random.choice(range(self.state_num), self.N).tolist()
            seed = [str(i) for i in seed]
            self.seed = seed
            high_area, low_area = {}, {}
            for key in self.cache.keys():
                if key > "200000000":
                    high_area[key] = 1
                else:
                    low_area[key] = 1
            high_area_fitness_list = [self.cache[key] for key in high_area.keys()]
            min_fitness_high = min(high_area_fitness_list)
            max_fitness_high = max(high_area_fitness_list)
            high_top, high_bottom = 1.0, 0.5
            low_area_fitness_list = [self.cache[key] for key in low_area.keys()]
            min_fitness_low = min(low_area_fitness_list)
            max_fitness_low = max(low_area_fitness_list)
            low_top, low_bottom = 0.5, 0
            for key in self.cache.keys():
                if key in high_area.keys():
                    self.cache[key] = (self.cache[key] - min_fitness_high) * (high_top - high_bottom) / \
                                      (max_fitness_high - min_fitness_high) + high_bottom
                else:
                    self.cache[key] = (self.cache[key] - min_fitness_low) * (low_top - low_bottom) / \
                                      (max_fitness_low - min_fitness_low) + low_bottom

        elif self.norm == "ClusterRangeScaling":
            cluster_0, cluster_1, cluster_2, cluster_3 = {}, {}, {}, {}
            for key in self.cache.keys():
                number_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
                for char in key:
                    if char in number_counts:
                        number_counts[char] += 1
                max_count = max(number_counts.values())
                most_frequent_numbers = [number for number, count in number_counts.items() if count == max_count]
                most_frequent_numbers = most_frequent_numbers[0]
                if most_frequent_numbers == "0":
                    cluster_0[key] = True
                elif most_frequent_numbers == "1":
                    cluster_1[key] = True
                elif most_frequent_numbers == "2":
                    cluster_2[key] = True
                else:
                    cluster_3[key] = True
            fitness_list_0 = [self.cache[key] for key in cluster_0.keys()]
            fitness_list_1 = [self.cache[key] for key in cluster_1.keys()]
            fitness_list_2 = [self.cache[key] for key in cluster_2.keys()]
            fitness_list_3 = [self.cache[key] for key in cluster_3.keys()]
            max_0, min_0 = max(fitness_list_0), min(fitness_list_0)
            max_1, min_1 = max(fitness_list_1), min(fitness_list_1)
            max_2, min_2 = max(fitness_list_2), min(fitness_list_2)
            max_3, min_3 = max(fitness_list_3), min(fitness_list_3)
            for key in self.cache.keys():
                if key in cluster_0:
                    self.cache[key] = (self.cache[key] - min_0) * 0.25 / \
                                      (max_0 - min_0)
                elif key in cluster_1:
                    self.cache[key] = (self.cache[key] - min_1) * 0.25 / \
                                      (max_1 - min_1) + 0.25
                elif key in cluster_2:
                    self.cache[key] = (self.cache[key] - min_2) * 0.25 / \
                                      (max_2 - min_2) + 0.50
                else:
                    self.cache[key] = (self.cache[key] - min_3) * 0.25 / \
                                      (max_3 - min_3) + 0.75

    def query_fitness(self, state: list) -> float:
        return self.cache["".join(state)]

    def query_cog_fitness(self, cog_state: list, state: list) -> float:
        cog_fitness = 0
        for row, cog_bit in enumerate(cog_state):
            dependency = self.dependency_map[row]
            bin_index = "".join([str(state[j]) for j in dependency])
            if cog_bit == "A":
                bin_index_1 = "0" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "1" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
            elif cog_bit == "B":
                bin_index_1 = "2" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "3" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
            elif cog_bit == "*":
                bin_index_0 = "0" + bin_index
                index_0 = int(bin_index_0, self.state_num)
                bin_index_1 = "1" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "2" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                bin_index_3 = "3" + bin_index
                index_3 = int(bin_index_3, self.state_num)
                c_i = (self.FC[row][index_0] + self.FC[row][index_1] +
                       self.FC[row][index_2] + self.FC[row][index_3]) / 4
            else:
                bin_index = cog_bit + bin_index
                index = int(bin_index, self.state_num)
                c_i = self.FC[row][index]
            cog_fitness += c_i
        return cog_fitness / self.N

    def query_partial_fitness(self, cog_state: list, state: list, expertise_domain: list) -> float:
        cog_fitness = 0
        for row, cog_bit in enumerate(cog_state):
            if row not in expertise_domain:
                continue
            dependency = self.dependency_map[row]
            bin_index = "".join([str(state[j]) for j in dependency])
            if cog_bit == "A":
                bin_index_1 = "0" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "1" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
            elif cog_bit == "B":
                bin_index_1 = "2" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "3" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                c_i = (self.FC[row][index_1] + self.FC[row][index_2]) / 2
            elif cog_bit == "*":
                bin_index_0 = "0" + bin_index
                index_0 = int(bin_index_0, self.state_num)
                bin_index_1 = "1" + bin_index
                index_1 = int(bin_index_1, self.state_num)
                bin_index_2 = "2" + bin_index
                index_2 = int(bin_index_2, self.state_num)
                bin_index_3 = "3" + bin_index
                index_3 = int(bin_index_3, self.state_num)
                c_i = (self.FC[row][index_0] + self.FC[row][index_1] +
                       self.FC[row][index_2] + self.FC[row][index_3]) / 4
            else:
                bin_index = cog_bit + bin_index
                index = int(bin_index, self.state_num)
                c_i = self.FC[row][index]
                print(bin_index, index, c_i)
            cog_fitness += c_i
        return cog_fitness / len(expertise_domain)

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

    def create_fitness_rank(self):
        fitness_cache = list(self.cache.values())
        ranks = rankdata(fitness_cache)
        ranks = [int(each) for each in ranks]
        rank_dict = {key: rank for key, rank in zip(self.cache.keys(), ranks)}
        return rank_dict

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

    def calculate_avg_fitness_distance(self):
        total_distance = 0
        total_neighbors = 0
        for key in self.cache.keys():
            neighbors = self.get_neighbor_list(key)
            total_distance += sum(abs(self.cache[key] - self.cache[neighbor]) for neighbor in neighbors)
            total_neighbors += len(neighbors)
        avg_fitness_distance = total_distance / total_neighbors
        return avg_fitness_distance

    def get_neighbor_list(self, key: str) -> list:
        """
        This is also for the Coarse Landscape
        :param key: string from the coarse landscape cache dict, e.g., "0011"
        :return:list of the neighbor state, e.g., [["0", "0", "1", "2"], ["0", "0", "1", "3"]]
        """
        neighbor_states = []
        for i, char in enumerate(key):
            neighbors = []
            for neighbor in range(4):
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
        print("Skewed Seed: ", self.seed)
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
    N = 9
    K = 8
    state_num = 4
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin")
    # print(landscape.FC[0])
    cog_state = ['B', 'B', 'B', 'A', 'A', 'B', "A", "B", "A"]
    # cog_state = ["0", "0", "0", "0", "0", "0"]
    # print(landscape.cog_state_alternatives(cog_state=cog_state))
    state_1_0 = ['2', '2', '2', '1', '1', '2', '1', '3', '0']
    state_1_1 = ['2', '2', '2', '1', '1', '2', '1', '3', '1']
    state_1_2 = ['2', '2', '2', '1', '1', '2', '1', '3', '2']
    state_1_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '3']
    state_2 = ['B', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'A']
    state_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '*']
    partial_fitness_1_0 = landscape.query_partial_fitness(cog_state=state_1_0, state=state_1_0, expertise_domain=list(range(N)))
    calculated_fitness = landscape.calculate_fitness(state=state_1_0)
    real_fitness = landscape.query_fitness(state=state_1_0)
    print(state_1_0, partial_fitness_1_0, real_fitness, calculated_fitness)
    # landscape.describe()

    # landscape.create_fitness_rank()
    # landscape.count_local_optima()
    # ave_distance = landscape.calculate_avg_fitness_distance()
    # ave_distance = round(ave_distance, 4)
    # print(landscape.local_optima)
    # print("Number of Local Optima: ", len(landscape.local_optima.keys()))
    # print("Average Distance: ", ave_distance)

    # import matplotlib.pyplot as plt
    # plt.hist(landscape.local_optima.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("Range")
    # plt.ylabel("Count")
    # plt.title("Local Optima N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.savefig("Local Optima N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.show()
    # plt.clf()

    # plt.hist(landscape.cache.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.title("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.xlabel("Range")
    # plt.ylabel("Count")
    # plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
    #     N, K, len(landscape.local_optima.keys()), ave_distance))
    # plt.show()

