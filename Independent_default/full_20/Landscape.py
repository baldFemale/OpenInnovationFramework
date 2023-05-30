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
    def __init__(self, N=None, K=None, state_num=4, norm="MaxMin"):
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

    def calculate_fitness(self, state):
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
            high_area, low_area = [], []
            for key in self.cache.keys():
                if key > "200000000":
                    high_area.append(key)
                else:
                    low_area.append(key)
            # scaled_fitness_low = (fitness - min_fitness_low) * (desired_upper_bound - desired_lower_bound) / (
            #             max_fitness_low - min_fitness_low) + desired_lower_bound
            high_area_fitness_list = [self.cache[key] for key in high_area]
            min_fitness_high = min(high_area_fitness_list)
            max_fitness_high = max(high_area_fitness_list)
            high_top, high_bottom = 1.0, 0.5
            low_area_fitness_list = [self.cache[key] for key in low_area]
            min_fitness_low = min(low_area_fitness_list)
            max_fitness_low = max(low_area_fitness_list)
            low_top, low_bottom = 0.5, 0
            for key in self.cache.keys():
                if key in high_area:
                    self.cache[key] = (self.cache[key] - min_fitness_high) * (high_top - high_bottom) / \
                                      (max_fitness_high - min_fitness_high) + high_bottom
                else:
                    self.cache[key] = (self.cache[key] - min_fitness_low) * (low_top - low_bottom) / \
                                      (max_fitness_low - min_fitness_low) + low_bottom
            # print("High Area: ", len(high_area), "Low Area: ", len(low_area))

            # Multiple center or multiple attractive centers
            # seed_num = 50
            # good_seed_list, bad_seed_list = [], []
            # for _ in range(seed_num):
            #     good_seed = np.random.choice(range(self.state_num), self.N).tolist()
            #     good_seed = [str(i) for i in good_seed]
            #     good_seed_list.append(good_seed)
            # for _ in range(seed_num):
            #     bad_seed = np.random.choice(range(self.state_num), self.N).tolist()
            #     bad_seed = [str(i) for i in bad_seed]
            #     bad_seed_list.append(bad_seed)
            # area_good, area_bad, area_middle = [], [], []
            # for key in self.cache.keys():
            #     distance_good = [self.get_hamming_distance(state_1=seed, state_2=list(key)) for seed in good_seed_list]
            #     distance_bad = [self.get_hamming_distance(state_1=seed, state_2=list(key)) for seed in bad_seed_list]
            #     if sum(distance_good) > sum(distance_bad):
            #         area_bad.append(key)
            #     elif sum(distance_bad) > sum(distance_good):
            #         area_good.append(key)
            #     else:
            #         area_middle.append(key)
            # print("Bad: ", len(area_bad), "Good: ", len(area_good), "Middle: ", len(area_middle))

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
        # print(rank_dict)
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
    K = 1
    state_num = 4
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="RangeScaling")
    # print(landscape.FC[0])
    # cog_state = ['A', 'A', 'A', 'A', 'A', 'A']
    # cog_state = ["0", "0", "0", "0", "0", "0"]
    # print(landscape.cog_state_alternatives(cog_state=cog_state))
    # state_1_0 = ['2', '2', '2', '1', '1', '2', '1', '3', '0']
    # state_1_1 = ['2', '2', '2', '1', '1', '2', '1', '3', '1']
    # state_1_2 = ['2', '2', '2', '1', '1', '2', '1', '3', '2']
    # state_1_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '3']
    # state_2 = ['B', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'A']
    # state_3 = ['2', '2', '2', '1', '1', '2', '1', '3', '*']
    # partial_fitness_1_0 = landscape.query_partial_fitness(cog_state=state_1_0, expertise_domain=range(0, 9))
    # partial_fitness_1_1 = landscape.query_partial_fitness(cog_state=state_1_1, expertise_domain=range(0, 9))
    # partial_fitness_1_2 = landscape.query_partial_fitness(cog_state=state_1_2, expertise_domain=range(0, 9))
    # partial_fitness_1_3 = landscape.query_partial_fitness(cog_state=state_1_3, expertise_domain=range(0, 9))
    # partial_fitness_2 = landscape.query_partial_fitness(cog_state=state_2, expertise_domain=range(0, 9))
    # partial_fitness_3 = landscape.query_partial_fitness(cog_state=state_3, expertise_domain=range(0, 9))
    # print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_0, partial_fitness_2))
    # print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_1, partial_fitness_2))
    # print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_2, partial_fitness_2))
    # print("Fine One {0} should NOT be equal to Coarse One {1}".format(partial_fitness_1_3, partial_fitness_2))
    # fine_state_fitness = [partial_fitness_1_0, partial_fitness_1_1, partial_fitness_1_2, partial_fitness_1_3]
    # print("Fine One {0} should be equal to Average of Coarse Ones {1}".format(partial_fitness_3, sum(fine_state_fitness) / len(fine_state_fitness)))
    landscape.describe()
    landscape.create_fitness_rank()
    landscape.count_local_optima()
    ave_distance = landscape.calculate_avg_fitness_distance()
    ave_distance = round(ave_distance, 4)
    # print(landscape.local_optima)
    print("Number of Local Optima: ", len(landscape.local_optima.keys()))
    print("Average Distance: ", ave_distance)

    import matplotlib.pyplot as plt
    # plt.plot(range(len(landscape.local_optima.values())), landscape.local_optima.values())
    # plt.xlabel("Local Optima")
    # plt.ylabel("Value")
    # plt.show()

    plt.hist(landscape.local_optima.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Range")
    plt.ylabel("Count")
    plt.title("Local Optima N={0}, K={1}, local optima={2}, ave_distance={3}".format(
        N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.savefig(" N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
        N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.show()
    plt.clf()
    plt.hist(landscape.cache.values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.title("N={0}, K={1}, local optima={2}, ave_distance={3}".format(
        N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.xlabel("Range")
    plt.ylabel("Count")
    plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
        N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.savefig("Landscape N={0}, K={1}, local optima={2}, ave_distance={3}.png".format(
        N, K, len(landscape.local_optima.keys()), ave_distance))
    plt.show()

