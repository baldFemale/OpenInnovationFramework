import numpy as np
from collections import defaultdict
from tools import *
from itertools import product


class LandScape:

    def __init__(self, N, K, IM_type, state_num=4):
        """
        :param N:
        :param K: range from 0 - N^2-N
        :param type: three types {random, influential, dependent}
        :param state_num:
        """
        self.N = N
        self.K = K
        self.IM_type = IM_type
        self.state_num = state_num
        self.IM, self.IM_dic = None, None
        self.FC = None
        self.cache = {}
        self.contribution_cache = {}
        self.cog_cache = {}

        self.fitness_to_rank_dict = None
        self.rank_to_fitness_dict = None

    def create_influence_matrix(self):
        IM = np.eye(self.N)

        if self.IM_type == "random":
            cells = [i*self.N+j for i in range(self.N) for j in range(self.N) if i != j]
            choices = np.random.choice(cells, self.K, replace=False).tolist()
        elif self.IM_type == "influential":
            cells = [i*self.N+j for j in range(self.N) for i in range(self.N) if i != j]
            choices = cells[:self.K]
        elif self.IM_type == "dependent":
            cells = [i*self.N+j for i in range(self.N) for j in range(self.N) if i != j]
            choices = cells[:self.K]

        for c in choices:
            IM[c // self.N][c % self.N] = 1

        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if i == j or IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)
        self.IM, self.IM_dic = IM, IM_dic

    def create_fitness_config(self,):
        FC = defaultdict(dict)
        for row in range(len(self.IM)):

            k = int(sum(self.IM[row]))
            for i in range(pow(self.state_num, k)):
                FC[row][i] = np.random.uniform(0, 1)
        self.FC = FC

    def calculate_fitness(self, state):
        res = []
        for i in range(len(state)):
            dependency = self.IM_dic[i]
            bin_index = "".join([str(state[j]) for j in dependency])

            bin_index = str(state[i]) + bin_index
            index = int(bin_index, self.state_num)
            res.append(self.FC[i][index])
        return np.mean(res), res

    def store_cache(self,):
        for i in range(pow(self.state_num,self.N)):
            bit = numberToBase(i, self.state_num)
            if len(bit)<self.N:
                bit = "0"*(self.N-len(bit))+bit
            state = [int(cur) for cur in bit]
            fitness, fitness_contribution = self.calculate_fitness(state)
            self.cache[bit] = fitness
            self.contribution_cache[bit] = fitness_contribution

    def rank_dict(self, cache):
        value_list = sorted(list(cache.values()), key=lambda x:-x)
        fitness_to_rank_dict = {}
        rank_to_fitness_dict = {}
        for index, value in enumerate(value_list):
            fitness_to_rank_dict[value] = index+1
            rank_to_fitness_dict[index+1] = value
        return fitness_to_rank_dict, rank_to_fitness_dict

    def initialize(self, first_time=True, norm=True):
        if first_time:
            self.create_influence_matrix()
        self.create_fitness_config()
        self.store_cache()

        # normalization
        if norm:
            normalizor = max(self.cache.values())
            min_normalizor = min(self.cache.values())

            for k in self.cache.keys():
                self.cache[k] = (self.cache[k]-min_normalizor)/(normalizor-min_normalizor)

        self.fitness_to_rank_dict, self.rank_to_fitness_dict = self.rank_dict(self.cache)
        self.cog_cache = {}

    def query_fitness(self, state):
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bit]

    def query_fitness_contribution(self, state):
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.contribution_cache[bit]

    def query_cog_fitness(self, state, knowledge_sapce, ):
        remainder = [cur for cur in range(self.N) if cur not in knowledge_sapce]
        regular_expression = "".join(str(state[i]) if i in knowledge_sapce else "*" for i in range(len(state)))
        if regular_expression in self.cog_cache:
            return self.cog_cache[regular_expression]

        remain_length = len(remainder)
        res = 0
        for i in range(pow(self.state_num, remain_length)):
            bit = numberToBase(i, self.state_num)
            if len(bit)<remain_length:
                bit = "0"*(remain_length-len(bit))+bit
            temp_state = list(state)

            for j in range(remain_length):
                temp_state[remainder[j]] = int(bit[j])
            res += self.query_fitness(temp_state)
        res = 1.0*res/pow(self.state_num, remain_length)
        self.cog_cache[regular_expression] = res

        return res

    def query_cog_fitness_gst(self, state, general_space, special_space, bit_difference=1):

        # print(state)

        alternative = []

        for cur in range(self.N):
            if cur in special_space:
                continue
            elif cur in general_space:
                temp = []
                for i in range(pow(2, bit_difference)):
                    bit_string = bin(i)[2:]
                    bit_string = "0"*(bit_difference-len(bit_string)) + bit_string
                    bit_string = str(state[cur]) + bit_string
                    temp.append(int(bit_string, 2))
                alternative.append(list(temp))
            else:
                temp = []
                for i in range(self.state_num):
                    temp.append(i)
                alternative.append(list(temp))

        res = 0
        alternative = list(product(*alternative))

        for alter in alternative:
            index = 0
            temp_state = list(state)
            for cur in range(self.N):
                if cur in special_space:
                    continue
                else:
                    temp_state[cur] = alter[index]
                    index += 1
            res += self.query_fitness(temp_state)
        return res/len(alternative)

    def query_cog_fitness_contribution_gst(self, state, general_space, special_space, bit_difference=1):

        alternative = []

        for cur in range(self.N):
            if cur in special_space:
                continue
            elif cur in general_space:
                temp = []
                for i in range(pow(2, bit_difference)):
                    bit_string = bin(i)[2:]
                    bit_string = "0" * (bit_difference - len(bit_string)) + bit_string
                    bit_string = str(state[cur]) + bit_string
                    temp.append(int(bit_string, 2))
                alternative.append(list(temp))
            else:
                temp = []
                for i in range(self.state_num):
                    temp.append(i)
                alternative.append(list(temp))

        res = [[] for cur in range(len(state))]
        alternative = list(product(*alternative))

        for alter in alternative:
            index = 0
            temp_state = list(state)
            for cur in range(self.N):
                if cur in special_space:
                    continue
                else:
                    temp_state[cur] = alter[index]
                    index += 1
            contribution = self.query_fitness_contribution(state)

            for cur in range(len(state)):
                res[cur].append(contribution[cur])

        return [np.mean(x) for x in res]

    def query_cog_fitness_tree(
            self, state, decision, knowledge_tree_list, tree_depth, learned_decision=None,
            teammate_decision=None, teammate_knowledge_tree_list=None,
    ):

        if teammate_decision is None:

            alternatives = []

            for cur in range(self.N):
                if cur not in decision:
                    alternatives.append([cur for cur in range(self.state_num)])
                else:
                    v = state[cur]
                    index = decision.index(cur)
                    node = knowledge_tree_list[index].leaf_map_node_list[v+(pow(2, tree_depth-1)-1)]
                    node_alternative = knowledge_tree_list[index].node_map_leaves_list[node]

                    node_alternative = [x-(pow(2, tree_depth-1)-1) for x in node_alternative]
                    alternatives.append(node_alternative)

            res = 0
            alternatives = list(product(*alternatives))
            for alter in alternatives:
                alter = list(alter)
                res += self.query_fitness(alter)

            return res / len(alternatives)

        else:
            alternatives = []

            for cur in range(self.N):
                if cur not in decision and cur not in learned_decision:
                    alternatives.append([cur for cur in range(self.state_num)])
                elif cur in decision and cur not in learned_decision:
                    v = state[cur]
                    focal_index = decision.index(cur)
                    node = knowledge_tree_list[focal_index].leaf_map_node_list[v+(pow(2, tree_depth-1)-1)]
                    node_alternative = knowledge_tree_list[focal_index].node_map_leaves_list[node]

                    node_alternative = [x - (pow(2, tree_depth - 1) - 1) for x in node_alternative]
                    alternatives.append(node_alternative)
                elif cur not in decision and cur in learned_decision:
                    v = state[cur]
                    team_index = teammate_decision.index(cur)
                    node = teammate_knowledge_tree_list[team_index].leaf_map_node_list[v+(pow(2, tree_depth-1)-1)]
                    node_alternative = teammate_knowledge_tree_list[team_index].node_map_leaves_list[node]

                    node_alternative = [x - (pow(2, tree_depth - 1) - 1) for x in node_alternative]
                    alternatives.append(node_alternative)
                else:
                    v = state[cur]

                    focal_index = decision.index(cur)
                    node = knowledge_tree_list[focal_index].leaf_map_node_list[v+(pow(2, tree_depth-1)-1)]
                    node_alternative = knowledge_tree_list[focal_index].node_map_leaves_list[node]
                    focal_node_alternative = [x - (pow(2, tree_depth - 1) - 1) for x in node_alternative]

                    team_index = teammate_decision.index(cur)
                    node = teammate_knowledge_tree_list[team_index].leaf_map_node_list[v+(pow(2, tree_depth-1)-1)]
                    team_node_alternative = teammate_knowledge_tree_list[team_index].node_map_leaves_list[node]

                    alternatives.append(
                        focal_node_alternative if len(
                            focal_node_alternative
                        ) <= len(team_node_alternative) else team_node_alternative
                    )

            res = 0
            alternatives = list(product(*alternatives))
            for alter in alternatives:
                alter = list(alter)
                res += self.query_fitness(alter)

            return res / len(alternatives)
