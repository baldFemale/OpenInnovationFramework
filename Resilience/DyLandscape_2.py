# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import product
import numpy as np
from ParentLandscape import ParentLandscape


class DyLandscape:
    """
    Dynamic Landscape Instance
    """
    def __init__(self, N, state_num=4, parent=None):
        self.N = N
        self.K = None
        self.k = None
        self.IM_type = None
        self.state_num = state_num
        self.IM, self.dependency_map = np.eye(self.N, dtype=int), [[]]*self.N  # [[]] & {int:[]}
        self.parent_FC = parent.FC  # keep slightly dependent due to the same parent
        self.FC = None
        self.cache = {}  # state string to overall fitness: state_num ^ N: [1]
        # self.contribution_cache = {}  # the original 1D fitness list before averaging: state_num ^ N: [N]
        self.cog_cache = {}  # for coordination where agents have some unknown element that might be changed by teammates
        self.fitness_to_rank_dict = None  # using the rank information to measure the potential performance of GST
        self.potential_cache = {}  # cache the potential of the position

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

    def type(self, IM_type="None", K=0, k=0, factor_num=0, influential_num=0, previous_IM=None, IM_change_bit=1):
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
        # if give the prevous IM, bypass the traditional generation process
        if previous_IM is not None:
            previous_IM = np.array(previous_IM)
            while True:
                # must use copy, otherwise the previous IM will also change and get into endless loop
                self.IM = previous_IM.copy()
                zero_positions = np.argwhere(self.IM == 0)
                one_positions = np.argwhere(self.IM == 1)
                shift_to_one_positions = np.random.choice(len(zero_positions), IM_change_bit, replace=False)
                shift_to_zero_positions = np.random.choice(len(one_positions), IM_change_bit, replace=False)
                shift_to_one_positions = [zero_positions[i] for i in shift_to_one_positions]  # 1 -> 0
                shift_to_zero_positions = [one_positions[i] for i in shift_to_zero_positions]  # 0 -> 1
                # print("shift_to_one_positions: ", shift_to_one_positions)
                # print("shift_to_zero_positions", shift_to_zero_positions)
                for indexs in shift_to_one_positions:
                    self.IM[indexs[0]][indexs[1]] = 1
                for indexs in shift_to_zero_positions:
                    # we cannot change the diagonal element into zero
                    if indexs[0] == indexs[1]:
                        print("re-assignment 1")
                        continue
                    self.IM[indexs[0]][indexs[1]] = 0
                # print("IM: \n", self.IM)
                # print("Previous: \n", previous_IM)
                # print("Y/N: ", (self.IM == previous_IM).all())
                if (self.IM == previous_IM).all():
                    print("re-assignment 2")
                    continue
                else:
                    break
        else:
            if K == 0:
                self.IM = np.eye(self.N)
            else:
                if self.IM_type == "Traditional Directed":
                    # each row has a fixed number of dependency (i.e., K)
                    for i in range(self.N):
                        probs = [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
                        if self.K < self.N:
                            ids = np.random.choice(self.N, self.K, p=probs, replace=False)
                            for index in ids:
                                self.IM[i][index] = int(1)
                        else:  # full dependency
                            for index in range(self.N):
                                self.IM[i][index] = int(1)
                elif self.IM_type == "Diagonal Mutual":
                    pass
                elif self.IM_type == "Random Mutual":
                    # select some dependencies, and such dependencies will be mutual.
                    pass

            if k != 0:
                if self.IM_type == "Random Directed":
                    cells = [i * self.N + j for i in range(self.N) for j in range(self.N) if i != j]
                    choices = np.random.choice(cells, self.k, replace=False).tolist()
                    for each in choices:
                        self.IM[each // self.N][each % self.N] = 1
                elif self.IM_type == "Factor Directed":  # columns as factor-> some key columns are more dependent to others
                    if factor_num == 0:
                        factor_num = self.k // self.N
                    factor_columns = np.random.choice(self.N, factor_num, replace=False).tolist()
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
                    influential_rows = np.random.choice(self.N, influential_num, replace=False).tolist()
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
            # e.g., dependency[0] = [1, 2, 3]
            # the fitness contribution of location 0 depends on the status of location 1,2,3

    def create_fitness_config(self,):
        """
        Create the child FC from the parent FC. This is a parent-child design pattern.
        We first set up the IM, and then according to the IM, we set up the FC.
        The IM evolute one bit each time, that is, one position in IM changes from 0 to 1. Accordingly, another position changes from 1 to 0.
        The child's FC is based on parent's full FC.
        Eventually, the FC is based on the same bigger parent FC.
        In other words, the child landscape is inherited from the parent landscape.
        This makes the child simular to each other, but slightly different, which is called dynamic landscape.
        This part is different from the original Landscape instance
        :return:
        """
        FC = defaultdict(dict)
        # the original FC generation process
        # for row in range(len(self.IM)):
        #     k = int(sum(self.IM[row]))
        #     for column in range(pow(self.state_num, k)):
        #         FC[row][column] = np.random.uniform(0, 1)
        # self.FC = FC

        # The dynamic/child FC generation process
        alternative_dependency = self.get_dependency_alternatives()
        # print("alternative_dependency: ", len(alternative_dependency[0]))
        for row in range(len(self.IM)):
            alternative_binary_index_scope = alternative_dependency[row]
            # change the possible state string into int index
            alternative_FC_index = [int("".join(each_str), self.state_num) for each_str in alternative_binary_index_scope]
            k = int(sum(self.IM[row]))
            # print(row, k, alternative_FC_index)
            # print("alternative_binary_index_scope", alternative_binary_index_scope)
            if len(alternative_FC_index) != pow(self.state_num, k):
                print("The calculated length of alternative pool: ", len(alternative_FC_index),)
                print("The expected length of alternative pool: ", pow(self.state_num, k))
                raise ValueError("the FC is not calculated correctly for row: ", row, self.IM[row], self.IM)
            for column_child, column_parent in zip(range(pow(self.state_num, k)), alternative_FC_index):
                FC[row][column_child] = self.parent_FC[row][column_parent]
                # the child lanscape's IM is cutted from the parent landscape's IM
                # according to the child's IM distribution, we calculate all the alternatives
                # those alternative dependency string will be transferred into separate index
        self.FC = FC

    def calculate_fitness(self, state):
        res = []
        for i in range(len(state)):
            dependency = self.dependency_map[i]
            binary_index = "".join([str(state[j]) for j in dependency])
            binary_index = str(state[i]) + binary_index
            index = int(binary_index, self.state_num)
            # print("binary_index: ", binary_index, index)
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

    def get_dependency_alternatives(self):
        """
        Create the alternatives given a dependency (e.g., [1,0,0,1])
        :param dependency:
        :return:(e.g., [1,0,0,1], [1,0,0,0], [1,0,0,2], [1,0,0,3], [2,0,0,1], ..., etc.)
                    the shape of resulting alternatives is N * (state_num ^ k).
                     the column number depends on the dependency sum (i.e., k) for each row, so it is dynamic.
        """
        alternative_pool_row = []
        for row in range(self.N):
            alternative_pool_column = []
            dependency_location = self.dependency_map[row]
            for column in range(self.N):
                if (column not in dependency_location) and (column != row):
                    alternative_pool_column.append(["0"])
                else:
                    alternative_pool_column.append(["0", "1", "2", "3"])
            alternative_pool_row.append([i for i in product(*alternative_pool_column)])
        # the shape of the alternative dependency pool
        # In other words, the clipped FC pieces from the parent landscape
        # for index, each_row in enumerate(alternative_pool_row):
        #     print("length of row %s: %s" % (index, len(each_row)))
        return alternative_pool_row

    def is_different_from(self, landscape=None):
        """
        How to calculate the simularity between landscapes
        In current algorithm, each point will have different value, as two elements will have different contribution.
        :param landscape: another child landscape
        :return: simularity indicator
        """
        count = 0
        total = self.state_num ** self.N
        for a, b in zip(self.cache.values(), landscape.cache.values()):
            if a == b:
                count += 1
        ratio = count / total
        print("Simularity: ", ratio, count)
        return ratio


if __name__ == '__main__':
    # Test Example
    parent = ParentLandscape(N=8, state_num=4)
    child_landscape_1 = DyLandscape(N=8, state_num=4, parent=parent)
    # # landscape.help() # just record some key hints
    # # landscape.type(IM_type="Influential Directed", k=20, influential_num=2)
    # landscape.type(IM_type="Factor Directed", k=20, factor_num=2)
    child_landscape_1.type(IM_type="Factor Directed", k=20)
    child_landscape_1.initialize()
    child_landscape_1.describe()
    child_landscape_2 = DyLandscape(N=8, state_num=4, parent=parent)
    child_landscape_2.type(IM_type=child_landscape_1.IM_type, previous_IM=child_landscape_1.IM, k=child_landscape_1.k)
    child_landscape_2.initialize()
    child_landscape_2.describe()
    cog_state = ['*', 'B', '1', '1', 'A', '3', 'A', '2']
    a = child_landscape_2.query_cog_fitness(cog_state)
    print(a)


