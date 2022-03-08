import numpy as np
from collections import defaultdict
from tools import *
from itertools import product
import time
import bisect


class LandScape:

    def __init__(self, N, K, state_num=2):
        """
        :param N:
        :param K: range from 0 - N^2-N
        :param type: three types {random, influential, dependent}
        :param state_num:
        """
        self.N = N
        self.K = K
        self.state_num = state_num

        self.full_IM, self.full_IM_dict = self.create_influence_matrix(N, N*(N-1))
        self.full_FC = self.create_fitness_config(self.full_IM)
        self.full_mapping = np.random.choice(state_num, self.N).tolist()

        self.IM, self.IM_dic = self.create_influence_matrix(self.N, self.K)
        self.cache = {}
        self.contribution_cache = {}

    def create_influence_matrix(self, N, K):
        """
        Generate different IM structure
        How many structures in the literature?
        """
        IM = np.eye(N)

        cells = [i*N+j for i in range(N) for j in range(N) if i != j]
        choices = np.random.choice(cells, K, replace=False).tolist()

        for c in choices:
            IM[c // N][c % N] = 1

        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if i == j or IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)

        return IM, IM_dic

    def create_fitness_config(self, IM):
        FC = defaultdict(dict)
        for row in range(len(IM)):

            k = int(sum(IM[row]))
            for i in range(pow(self.state_num, k)):
                FC[row][i] = np.random.uniform(0, 1)
        return FC

    def update_fitness_cogfig(self, error_weight):

        for row in range(len(self.full_IM)):
            k = int(sum(self.full_IM[row]))
            for i in range(pow(self.state_num, k)):
                # print(self.FC[row][i], error_weight)
                self.full_FC[row][i] = self.full_FC[row][i] * (1-error_weight) + error_weight * np.random.uniform(0, 1)

    def update_IM_structure(self, extent):

        cells = [i * self.N + j for i in range(self.N) for j in range(self.N) if i != j]
        choices = np.random.choice(cells, extent, replace=False).tolist()

        for c in choices:
            row = c//self.N
            col = c % self.N

            if self.IM[row][col] == 1:
                self.IM[row][col] = 0
                index = self.IM_dic[row].index(col)
                self.IM_dic[row].pop(index)
            else:
                self.IM[row][col] = 1
                bisect.insort(self.IM_dic[row], col)

    def store_cache(self,):

        self.min_normalizor = float("inf")
        self.max_normalizor = -float("inf")

        for i in range(pow(self.state_num, self.N)):
            bit = numberToBase(i, self.state_num)
            if len(bit)<self.N:
                bit = "0"*(self.N-len(bit))+bit

            temp_state = [int(x) for x in bit]
            fitness, contribution = self.calculate_fitness(temp_state)
            self.cache[bit] = fitness
            self.contribution_cache[bit] = contribution

            if self.cache[bit] > self.max_normalizor:
                self.max_normalizor = float(self.cache[bit])
            if self.cache[bit] < self.min_normalizor:
                self.min_normalizor = float(self.cache[bit])

    def calculate_fitness(self, state):
        """
        Param state: the decision string
        Return: 1. the average fitness across state bits: 1 by 1
                    2. the original 1D fitness list: 1 by len(state)
        """
        res = []
        for i in range(len(state)):
            dependency = self.IM_dic[i]

            bin_index = str(state[i])
            for cur in range(self.N):
                if i == cur:
                    continue
                else:
                    if cur in dependency:
                        bin_index += str(state[cur])
                    else:
                        bin_index += str(self.full_mapping[cur])
            index = int(bin_index, self.state_num)
            res.append(self.full_FC[i][index])
        return np.mean(res), res

    def initialize(self):
        self.store_cache()

    def query_fitness(self, state):
        """
        Query the average fitness from the landscape cache for each decision string
        """
        bit = "".join([str(state[i]) for i in range(len(state))])
        return (self.cache[bit]-self.min_normalizor)/(self.max_normalizor-self.min_normalizor)

    def query_fitness_contribution(self, state):
        """
        Query the fitness list from the detailed contribution cache for each decision string
        """
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.contribution_cache[bit]

    def query_partial_fitness_tree(self, state, decision, knowledge_tree_list, tree_depth,):

        overall_fitness = []

        for cur in range(self.N):
            if cur not in decision:
                pass
            else:

                index = decision.index(cur)
                v = state[cur]
                node = knowledge_tree_list[index].leaf_map_node_list[v + (pow(2, tree_depth - 1) - 1)]
                node_alternative = knowledge_tree_list[index].node_map_leaves_list[node]
                node_alternative = [x - (pow(2, tree_depth - 1) - 1) for x in node_alternative]

                temp_contribution = []

                for alternative in node_alternative:
                    temp_state = list(state)
                    temp_state[cur] = alternative
                    temp_contribution.append(self.query_fitness_contribution(temp_state)[cur])
                overall_fitness.append(np.mean(temp_contribution))
        return np.mean(overall_fitness)

    def update_contribution_weight(self, change_bool, magnitude, change_weight=True):
        if not change_bool:
            return
        else:
            if change_weight:
                self.update_fitness_cogfig(magnitude)
                self.store_cache()
            else:
                self.update_IM_structure(magnitude)
                self.store_cache()
                # print(self.IM, self.IM_dic)

