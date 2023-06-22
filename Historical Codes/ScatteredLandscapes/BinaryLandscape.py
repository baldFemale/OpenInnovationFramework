# -*- coding: utf-8 -*-
# @Time     : 5/16/2023 16:31
# @Author   : Junyi
# @FileName: BinaryLandscape.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from collections import defaultdict
import time


class BinaryLandscape():
    def __init__(self, N=None, K=None, K_within=None, K_between=None, norm="MaxMin"):
        self.N = N
        self.K = K
        self.K_within = K_within
        self.K_between = K_between
        self.IM, self.IM_dic = None, None
        self.FC = None
        self.cache = {}  # the hashed dict has a higher indexing speed, which helps improve the running speed
        self.cog_cache = {}
        self.norm = norm
        self.max_normalizer, self.min_normalizer = 0, 0
        self.initialize()

    def create_influence_matrix(self):
        IM = np.eye(self.N)
        if self.K_within is None:
            for i in range(self.N):
                probs = [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
                ids = np.random.choice(self.N, self.K, p=probs, replace=False)
                for index in ids:
                    IM[i][index] = 1
        else:
            for i in range(self.N):
                if i // (self.N // 2) < 1:
                    within = [j for j in range(self.N // 2)]
                    between = [j for j in range(self.N // 2, self.N)]
                    probs = [1 / (self.N // 2 - 1)] * i + [0] + [1 / (self.N // 2 - 1)] * (self.N // 2 - 1 - i)
                    ids_within = np.random.choice(within, self.K_within, p=probs, replace=False)
                    ids_between = np.random.choice(between, self.K_between, replace=False)
                    for index in ids_within:
                        IM[i][index] = 1
                    for index in ids_between:
                        IM[i][index] = 1

                else:
                    within = [j for j in range(self.N // 2, self.N)]
                    between = [j for j in range(self.N // 2)]
                    probs = [1 / (self.N // 2 - 1)] * (i - self.N // 2) + [0] + [1 / (self.N // 2 - 1)] * (
                            self.N - 1 - i)
                    ids_between = np.random.choice(between, self.K_between, replace=False)
                    ids_within = np.random.choice(within, self.K_within, p=probs, replace=False)
                    for index in ids_within:
                        IM[i][index] = 1
                    for index in ids_between:
                        IM[i][index] = 1

        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if i == j or IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)
        self.IM, self.IM_dic = IM, IM_dic

    def create_fitness_config(self, ):
        FC = defaultdict(dict)
        for row in range(len(self.IM)):

            k = int(sum(self.IM[row]))
            for i in range(pow(2, k)):
                FC[row][i] = np.random.uniform(0, 1)
        self.FC = FC

    def calculate_fitness(self, state):
        res = 0.0
        for i in range(len(state)):
            dependency = self.IM_dic[i]
            bin_index = "".join([str(state[j]) for j in dependency])
            if state[i] == 0:
                bin_index = "0" + bin_index
            else:
                bin_index = "1" + bin_index
            index = int(bin_index, 2)
            res += self.FC[i][index]
        return res / len(state)

    def store_cache(self, ):
        for i in range(pow(2, self.N)):
            bit = bin(i)[2:]
            if len(bit) < self.N:
                bit = "0" * (self.N - len(bit)) + bit
            state = [int(cur) for cur in bit]
            self.cache[bit] = self.calculate_fitness(state)

    def initialize(self):
        self.create_influence_matrix()
        self.create_fitness_config()
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
        self.cog_cache = {}

    def query_fitness(self, state):
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bit]

    def query_cog_fitness(self, state, knowledge_sapce):
        remainder = [cur for cur in range(self.N) if cur not in knowledge_sapce]
        regular_expression = "".join(str(state[i]) if i in knowledge_sapce else "*" for i in range(len(state)))
        if regular_expression in self.cog_cache:
            return self.cog_cache[regular_expression]

        remain_length = len(remainder)
        res = 0
        for i in range(pow(2, remain_length)):
            bit = bin(i)[2:]
            if len(bit) < remain_length:
                bit = "0" * (remain_length - len(bit)) + bit
            temp_state = list(state)

            for j in range(remain_length):
                temp_state[remainder[j]] = int(bit[j])
            res += self.query_fitness(temp_state)
        res = 1.0 * res / pow(2, remain_length)
        self.cog_cache[regular_expression] = res

        return res


class Agent:

    def __init__(self, N, landscape):
        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.landscape = landscape
        self.fitness = self.landscape.query_fitness(self.state)

    def search(self, ):
        next_state = list(self.state)
        next_index = np.random.choice(self.N)
        next_state[next_index] ^= 1

        if self.landscape.query_fitness(self.state) < self.landscape.query_fitness(next_state):
            self.state = next_state
            self.fitness = self.landscape.query_fitness(next_state)

if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(1000)
    gap_list = []
    ress = []

    N = 10
    agent_num = 100
    search_iteration = 100
    landscape_repeat = 5
    K = 0
    landscape = BinaryLandscape(N, K, None, None)
    bin_data = list(landscape.cache.values())
    plt.hist(bin_data, bins=40, alpha=0.5, label='Binary Landscape', color="blue", density=True)
    plt.show()
    # K_list = [0, 2, 4, 6]
    # performance_across_para = []
    # for K in K_list:  # key parameter for complexity
    #     performance_one_para = []
    #     agents_performance = []
    #     for i in range(landscape_repeat):  # landscape repetitions
    #         landscape = BinaryLandscape(N, K, None, None)
    #         landscape.initialize(norm=True)
    #         crowd = []
    #         for _ in range(agent_num):
    #             agent = Agent(N, landscape)
    #             crowd.append(agent)
    #         for agent in crowd:  # agent repetitions
    #             agent_performance = []
    #             for _ in range(search_iteration):
    #                 agent.search()
    #                 agent_performance.append(agent.fitness)
    #             agents_performance.append(agent_performance)
    #
    #     for period in range(search_iteration):
    #         temp = [agent_performance[period] for agent_performance in agents_performance]
    #         performance_one_para.append(sum(temp) / len(temp))
    #     performance_across_para.append(performance_one_para)
    #
    # x = range(search_iteration)
    # for index, K in enumerate(K_list):
    #     plt.plot(x, performance_across_para[index], label="K={0}".format(K))
    # plt.xlabel('Time', fontweight='bold', fontsize=10)
    # plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # # plt.xticks(x)
    # plt.legend()
    # plt.show()
    # plt.savefig("\Performance_across_K.png", transparent=False, dpi=200)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

