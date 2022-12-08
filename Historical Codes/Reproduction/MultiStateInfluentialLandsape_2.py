# -*- coding: utf-8 -*-
# @Time     : 12/13/2021 15:44
# @Author   : Junyi
# @FileName: MultiStateInfluentialLandsape.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from collections import defaultdict
from itertools import product
import time
import matplotlib.pyplot as plt


class LandScape:
    def __init__(self, N, K, IM_type, IM_random_ratio=None, state_num=4):
        """
        :param N:
        :param K:
        :param IM_type: three types {random, influential, dependent}
        :param IM_random_ratio:
        :param state_num:
        """
        if state_num >= N:
            raise ValueError("The depth of state cannot be greater than its width")

        self.N = N
        self.K = K
        self.IM_type = IM_type
        self.IM_random_ratio = IM_random_ratio

        self.IM_np = None  # finally will be np.array
        self.state_num = state_num
        self.FC_np = None  # finally will be np.array
        self.cache_np = []  # (1* state_num^N), using int(str) as the index, rather than hash the bit string in dict {}.
        self.contribution_cache_np = [] # (1*state_num^N), using int(str) as the index, rather than hash the bit string in dict {}.

        self.cog_cache_np = {}  # this cognitive cache will not be as large as 2^N, and there are some masked element.
        self.cog_contribution_cache_np = {}
        self.fitness_to_rank_dict = None
        self.rank_to_fitness_dict = None

    def create_influence_matrix(self):
        """
        Generate different IM structure
        How many structures in the literature?
        """
        IM = np.eye(self.N)
        choices = None
        if self.IM_type == "random":
            cells = [i*self.N+j for i in range(self.N) for j in range(self.N) if i != j]
            choices = np.random.choice(cells, self.K, replace=False).tolist()
        elif self.IM_type == "influential":
            cells = [i*self.N+j for j in range(self.N) for i in range(self.N) if i != j]
            choices = cells[:self.K]

            if self.IM_random_ratio is not None:
                remain_choice = np.random.choice(
                    choices, int(len(choices) * self.IM_random_ratio), replace=False
                ).tolist()
                random_choice = np.random.choice(
                    [i * self.N + j for j in range(self.N) for i in range(self.N) if i != j and i * self.N + j not in remain_choice],
                    self.K - len(remain_choice),
                    replace=False
                ).tolist()

                choices = remain_choice + random_choice

        elif self.IM_type == "dependent":
            cells = [i*self.N+j for i in range(self.N) for j in range(self.N) if i != j]
            choices = cells[:self.K]

            if self.IM_random_ratio is not None:
                remain_choice = np.random.choice(
                    choices, int(len(choices) * self.IM_random_ratio), replace=False
                ).tolist()
                random_choice = np.random.choice(
                    [i * self.N + j for j in range(self.N) for i in range(self.N) if i != j and i * self.N + j not in remain_choice],
                    self.K - len(remain_choice),
                    replace=False
                ).tolist()

                choices = remain_choice + random_choice

        for c in choices:
            IM[c // self.N][c % self.N] = 1

        self.IM_np = np.array(IM, dtype=int)
        # print("IM:\n",self.K,self.IM)


    def create_fitness_config_np(self, ):
        # dynamic IM type such that each row will have different k, rather than consistent K
        FC = defaultdict(dict)
        for row in range(len(self.IM_np)):
            k = np.sum(self.IM_np[row])
            for i in range(pow(self.state_num, k)):
                FC[row][i] = np.random.uniform(0, 1)
        self.FC_np = np.array(FC, dtype=object)

    def calculate_fitness_np(self, state):
        """
        Given a intact decision string (without '*' ), calculate the according fitness average across each element
        :param state: intact decision string
        :return: overall fitness for the introduction of each element;
                    detailed fitness list for the introduction of each element;
        """
        element_to_fitness = []
        for row in range(len(state)):
            bin_index = "".join([str(state[j]) for j in self.IM_np[row]])
            column = int(bin_index, self.state_num)
            element_to_fitness.append(self.FC_np[row][column])
        return sum(element_to_fitness)/len(element_to_fitness), element_to_fitness

    def store_cache_np(self, ):
        self.cache_np = [self.calculate_fitness_np(each_state)[0] for each_state in product(range(self.state_num), repeat=self.N)]
        self.contribution_cache_np = [self.calculate_fitness_np(each_state)[1] for each_state in product(range(self.state_num), repeat=self.N)]

    def rank_dict_np(self,cache):
        """
        Sort the cache fitness value and corresponding rank
        To get another performance indicator regarding the reaching rate of relatively high fitness (e.g., the top 10 %)
        :param cache: the fitness cache given a landscape
        :return:
        """
        value_list = sorted(list(cache.values()), key=lambda x:-x)
        fitness_to_rank_dict = {}
        rank_to_fitness_dict = {}
        for index, value in enumerate(value_list):
            fitness_to_rank_dict[value] = index+1
            rank_to_fitness_dict[index+1] = value
        return fitness_to_rank_dict, rank_to_fitness_dict



    def initialize_np(self, first_time=True, norm=True):
        if first_time:
            self.create_influence_matrix()
        self.create_fitness_config_np()
        self.store_cache_np()
        if norm:
            max_ = max(self.cache_np)
            min_ = min(self.cache_np)
            self.cache_np = [(i - min_) / (max_ - min_) for i in self.cache_np]

    def query_fitness_np(self, state):
        """
        Query the fitness value
        :param state: the decision string
        :return: the accurate fitness value in landscape
        """
        state = np.array(state)
        state_index = state.dot(self.state_num ** np.arange(state.size)[::-1])
        print(state_index)
        print(len(self.cache_np))
        return self.cache_np[state_index]


    def query_cog_fitness_np(self, state, knowledge_space=[3,4]):
        """
        Cognitive search where agent only have limited knowledge; Query the cognitive fitness
        :param state: full decision string
        :param knowledge_space: the agent manageable element for all full decision string
        :return: the perceived fitness value in cognitive landscape
        """
        unknown_domains = [cur for cur in range(self.N) if cur not in knowledge_space]
        # cognitive search has some masked/unknown/unmanageable elements in the state string
        regular_expression = "".join(str(state[i]) if i in knowledge_space else "*" for i in range(len(state)))
        if regular_expression in self.cog_cache_np:
            return self.cog_cache_np[regular_expression]

        unknown_domains_length = len(unknown_domains)
        res = 0
        # take the average across all alternative unknown combinations as the cognitive fitness.
        for i in range(pow(self.state_num, unknown_domains_length)):
            bit = bin(i)[2:]
            if len(bit) < unknown_domains_length:
                bit = "0" * (unknown_domains_length-len(bit)) + bit  # one possible combination
            temp_state = list(state)  # the original state string
            for j in range(unknown_domains_length):
                temp_state[unknown_domains[j]] = int(bit[j])  # iteratively padding with all the alternative combinations
            res += self.query_fitness_np(temp_state)
        res = 1.0 * res / pow(self.state_num, unknown_domains_length)
        self.cog_cache_np[regular_expression] = res
        return res


class Agent:

    def __init__(self, N, landscape):
        self.N = N
        self.state = np.random.choice([0, 1], self.N)
        self.landscape = landscape
        self.state_num = landscape.state_num
        self.fitness_np = self.landscape.query_fitness_np(self.state)
        self.temp_state = None

    def adaptation_np(self, ):
        """
        Simple search where agents known all the knowledge domains.
        :return: None
        """
        self.temp_state = self.state.copy()  # 显式复制
        choice = np.random.choice(self.N)
        self.temp_state[choice] ^= 1
        if self.landscape.query_fitness_np(self.state) < self.landscape.query_fitness_np(self.temp_state):
            self.state = self.temp_state.copy()
            self.fitness_np = self.landscape.query_fitness_np(self.temp_state)

    def cog_adaptation_np(self, ):
        """
        Cognitive search where agents known limited knowledge domains.
        :return: None
        """
        self.temp_state = self.state.copy()  # 显式复制
        choice = np.random.choice(self.N)
        self.temp_state[choice] ^= 1
        if self.landscape.query_cog_fitness_np(self.state) < self.landscape.query_cog_fitness_np(self.temp_state):
            self.state = self.temp_state.copy()
            self.fitness_np = self.landscape.query_cog_fitness_np(self.temp_state)


if __name__ == '__main__':
    np.random.seed(100)
    gap_list = []
    for i in range(5):
        start_time = time.time()
        N = 8
        ress = []
        for k in [2,3,4]:
            res = []
            landscape = LandScape(N, k, IM_type='dependent', IM_random_ratio=None)
            landscape.initialize_np()
            for repeat in range(200):
                fitness = []
                agent = Agent(N, landscape)
                for step in range(100):
                    agent.adaptation_np()
                    fitness.append(agent.fitness_np)
                res.append(fitness)
            ress.append(res)
        end_time = time.time()
        gap = end_time - start_time
        gap_list.append(gap)

    print("Average Running Time: ", sum(gap_list) / len(gap_list))

    for k in range(4):
        plt.plot(np.mean(np.array(ress[k]), axis=0), label="k=%d" % (k * 2))
    plt.legend()
    plt.show()

