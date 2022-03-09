# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from itertools import product
from itertools import combinations
import numpy as np


class Landscape:
    """
    Dynamic Landscape Instance
    """

    def __init__(self, N, state_num=4, dynamic_flag=0):
        self.N = N
        self.K = None
        self.k = None
        self.IM_type = None
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N), [[]]*self.N  # [[]] & {int:[]}
        self.FC = None
        self.cache = {}  # state string to overall fitness: state_num ^ N: [1]
        # self.contribution_cache = {}  # the original 1D fitness list before averaging: state_num ^ N: [N]
        self.cog_cache = {}  # for coordination where agents have some unknown element that might be changed by teammates
        self.fitness_to_rank_dict = None  # using the rank information to measure the potential performance of GST
        self.potential_cache = {}  # cache the potential of the position
        self.dynamic_flag = dynamic_flag  # by default, starting from the first element in the alternative combinations pool


    def describe(self):
        print("*********LandScape information********* ")
        print("LandScape shape of (N={0}, K={1}, k={2}, state number={3})".format(self.N, self.K, self.k, self.state_num))
        print("Influential matrix type: ", self.IM_type)
        print("Influential matrix: \n", self.IM)
        print("Influential dependency map: ", self.dependency_map)
        print("********************************")

    def help(self):
        valid_type = ["None", "Traditional Directed", "Diagonal Mutual", "Random Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
        print("Supported IM_type: ", valid_type)

    def type(self, IM_type="None", K=0, k=0, factor_num=0, influential_num=0):
        """
        Characterize the influential matrix
        :param IM_type: "random", "dependent",
        :param K: mutual excitation dependency (undirected links)
        :param k: single-way dependency (directed links); k=2K for mutual dependency
        :return: the influential matrix (self.IM); and the dependency rule (self.IM_dict)
        """
        if K * k != 0:
            raise ValueError("K is for mutual/undirected excitation dependency (i.e., Traditional Mutual & Diagonal Mutual), "
                             "while k is for a new design regarding the total number of links, "
                             "a directed dependency (i.e., Influential Directed & Random Directed)."
                             "These two parameter cannot co-exist")
        if (K > self.N) or (k > self.N*self.N):
            raise ValueError("K or k is too large than the size of N")
        valid_IM_type = ["None", "Traditional Directed", "Diagonal Mutual", "Random Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
        if IM_type not in valid_IM_type:
            raise ValueError("Only support {0}".format(valid_IM_type))
        if (IM_type in ["Traditional Directed", "Diagonal Mutual", "Random Mutual"]):
            if k != 0:
                raise ValueError("k ({0}) is for directed or single-way dependency, rather than {1}".format(k, IM_type))
        if (IM_type in ["Influential Directed", "Random Directed", "Factor Directed"]):
            if k == 0:
                raise ValueError("Mismatch between k={0} and IM_type={1}".format(k, IM_type))
            if K != 0:
                raise ValueError("K ({0}) is for undirected or double-way dependency, "
                                 "or fixed number for each row/column,"
                                 " rather than {1}".format(K, IM_type))
        if (IM_type == 'Factor Directed') & (influential_num != 0):
            raise ValueError("Factor Directed: influential_num != 0")
        if (IM_type == 'Influential Directed') & (factor_num != 0):
            raise ValueError("Influential Directed cannot have factor_num != 0")
        if (IM_type == "None") & (K+k != 0):
            raise ValueError("Need a IM type. Current (default) IM type ({0}) mismatch with K ({1}) = 0 and k ({2}) = 0".format(IM_type,K,k))

        self.IM_type = IM_type
        self.K = K
        self.k = k
        if K == 0:
            self.IM = np.eye(self.N)
        else:
            if self.IM_type == "Traditional Directed":
                # each row has a fixed number of dependency (i.e., K)
                # dynamic pattern: use the combinations to run through all the possibilities
                # rather than the traditional random selection design
                ids = self.get_dependency_combinations(dynamic_flag=self.dynamic_flag)
                for i in range(self.N):
                    for index in ids:
                        self.IM[i][index] = 1

            elif self.IM_type == "Diagonal Mutual":
                pass
            elif self.IM_type == "Random Mutual":
                # select some dependencies, and such dependencies will be mutual.
                pass

        if k != 0:
            if self.IM_type == "Random Directed":
                print("Inapplicable type for dynamic dependency where dependency should be arranged")
                # cells = [i * self.N + j for i in range(self.N) for j in range(self.N) if i != j]
                # choices = np.random.choice(cells, self.k, replace=False).tolist()  # change this into combination, instead of random selection
                # for each in choices:
                #     self.IM[each // self.N][each % self.N] = 1
            elif self.IM_type == "Factor Directed":  # columns as factor-> some key columns are more dependent to others
                if factor_num == 0:
                    factor_num = self.k // self.N
                # factor_columns = np.random.choice(self.N, factor_num, replace=False).tolist()
                # using a arranged dependency combination
                factor_columns = self.get_dependency_combinations(dynamic_flag=self.dynamic_flag)
                for cur_i in range(self.N):
                    for cur_j in range(self.N):
                        if (cur_j in factor_columns) & (k > 0):
                            self.IM[cur_i][cur_j] = 1
                            k -= 1
                if k > 0:
                    zero_positions = np.argwhere(self.IM == 0)
                    fill_with_one_positions = np.random.choice(len(zero_positions), k, replace=False)
                    fill_with_one_positions = [zero_positions[i] for i in fill_with_one_positions]
                    for indexs in fill_with_one_positions:
                        self.IM[indexs[0]][indexs[1]] = 1

            elif self.IM_type == "Influential Directed":  # rows as influential -> some key rows depend more on others
                if influential_num == 0:
                    influential_num = self.k // self.N
                # influential_rows = np.random.choice(self.N, influential_num, replace=False).tolist()
                influential_rows = self.get_dependency_combinations(dynamic_flag=self.dynamic_flag)
                for cur_i in range(self.N):
                    for cur_j in range(self.N):
                        if (cur_i in influential_rows) & (k > 0):
                            self.IM[cur_i][cur_j] = 1
                            k -= 1
                if k > 0:
                    zero_positions = np.argwhere(self.IM == 0)
                    fill_with_one_positions = np.random.choice(len(zero_positions), k, replace=False)
                    fill_with_one_positions = [zero_positions[i] for i in fill_with_one_positions]
                    for indexs in fill_with_one_positions:
                        self.IM[indexs[0]][indexs[1]] = 1

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
        res = []
        for i in range(len(state)):
            dependency = self.dependency_map[i]
            binary_index = "".join([str(state[j]) for j in dependency])
            binary_index = str(state[i]) + binary_index
            index = int(binary_index, self.state_num)
            res.append(self.FC[i][index])
        return np.mean(res)

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
        for index, value in enumerate(value_list):
            fitness_to_rank_dict[value] = index+1
        self.fitness_to_rank_dict = fitness_to_rank_dict

    def initialize(self, norm=True):
        """
        Cache the fitness value
        :param norm: normalization
        :return: fitness cache
        """
        self.create_fitness_config()
        self.store_cache()
        # normalization
        if norm:
            normalizor = max(self.cache.values())
            min_normalizor = min(self.cache.values())
            for k in self.cache.keys():
                self.cache[k] = (self.cache[k]-min_normalizor)/(normalizor-min_normalizor)
        self.creat_fitness_rank_dict()

    def query_fitness(self, state):
        """
        Query the average fitness from the landscape cache for *intact* decision string
        """
        bits = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bits]

    def query_cog_fitness(self, cog_state=None):
        cog_state_string = ''.join([str(i) for i in cog_state])
        if cog_state_string in self.cog_cache.keys():
            return self.cog_cache[cog_state_string]
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        fitness_pool = [self.query_fitness(each) for each in alternatives]
        cog_fitness = sum(fitness_pool)/len(alternatives)
        self.cog_cache[cog_state_string] = cog_fitness
        return cog_fitness

    def query_potential_performance(self, cog_state=None, top=1):
        cog_state_string = ''.join([str(i) for i in cog_state])
        if cog_state_string in self.potential_cache.keys():
            return self.potential_cache[cog_state_string]
        alternatives = self.cog_state_alternatives(cog_state=cog_state)
        fitness_pool = [self.query_fitness(each) for each in alternatives]
        position_potential = sorted(fitness_pool)[-top]
        position_potential_rank = self.fitness_to_rank_dict[position_potential]
        self.potential_cache[cog_state_string] = position_potential_rank
        return position_potential_rank

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

    def get_dependency_combinations(self, dynamic_flag):
        """
        only for the dynamic dependency design
        So we try to simulate each dependency with same iteration rounds
        Given the same k/K, the performance across different dependency would be taken as resiliance of performance
        :param dynamic_flag: 0 by default, otherwise change it over iterations
        :return: the selected dependency combination given N, K/k
        """
        absolute_k = self.K if self.K else self.k // 10  # a flag for to dependency combination
        return list(combinations(range(self.N), absolute_k))[dynamic_flag]



if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    # landscape.help() # just record some key hints
    # landscape.type(IM_type="Influential Directed", k=20, influential_num=2)
    # landscape.type(IM_type="Factor Directed", k=20, factor_num=2)
    landscape.type(IM_type="Factor Directed", k=44)
    landscape.initialize()
    landscape.describe()
    cog_state = ['*', 'B', '1', '1', 'A', '3', 'A', '2']
    a = landscape.query_cog_fitness(cog_state)
    print(a)

