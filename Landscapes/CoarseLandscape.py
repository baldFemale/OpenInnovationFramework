# -*- coding: utf-8 -*-
# @Time     : 5/11/2023 15:26
# @Author   : Junyi
# @FileName: Cog_landscape.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from collections import defaultdict
from itertools import product
import numpy as np


class CoarseLandscape:
    def __init__(self, landscape=None, expertise_domain=None, expertise_representation=None,
                 norm="None", collaborator="None"):
        self.landscape = landscape
        self.expertise_domain = expertise_domain
        self.expertise_representation = expertise_representation
        self.N = self.landscape.N
        self.K = self.landscape.K
        self.state_num = self.landscape.state_num
        self.IM, self.dependency_map = self.landscape.IM, self.landscape.dependency_map
        self.FC = self.landscape.FC
        self.cache = {}  # state string to partial fitness: state_num ^ N: [1]
        self.max_normalizer = 1
        self.min_normalizer = 0
        self.norm = norm
        self.collaborator = collaborator
        self.initialize()

    def describe(self):
        print("*********Coarse LandScape Information********* ")
        print("LandScape shape of (N={0}, K={1}, state number={2})".format(self.N, self.K, self.state_num))
        print("Influential matrix: \n", self.IM)
        print("Influential dependency map: ", self.dependency_map)
        print("Max Fitness: ", self.max_normalizer)
        print("Min Fitness: ", self.min_normalizer)
        print("Ave Fitness: ", sum(self.cache.values()) / len(self.cache.values()))
        print("********************************")

    # def calculate_cog_fitness(self, coarse_state=None):
    #     partial_fitness_alternatives = []
    #     alternatives = self.cog_state_alternatives(coarse_state=coarse_state)  # get back to the finest level
    #     # so that we can calculate it using FC, which is all at the finest level (i.e., 0, 1, 2, 3)
    #     for state in alternatives:
    #         partial_FC_across_bits = []
    #         for index in range(len(state)):
    #             if index not in self.expertise_domain:
    #                 continue
    #             dependency = self.dependency_map[index]
    #             bit_index = "".join([str(state[j]) for j in dependency])
    #             bit_index = str(state[index]) + bit_index
    #             FC_index = int(bit_index, self.state_num)
    #             partial_FC_across_bits.append(self.FC[index][FC_index])
    #         partial_fitness_state = sum(partial_FC_across_bits) / len(self.expertise_domain)
    #         partial_fitness_alternatives.append(partial_fitness_state)
    #     return sum(partial_fitness_alternatives) / len(partial_fitness_alternatives)

    def calculate_coarse_fitness(self, coarse_state=None):
        alternative_states = self.coarse_state_alternatives(coarse_state=coarse_state)  # get back to the finest level
        alternative_fitness = [self.landscape.query_fitness(state=state) for state in alternative_states]
        return sum(alternative_fitness) / len(alternative_fitness)

    def store_cache(self):
        # Cartesian products within knowledge scope (not combinations or permutations)
        known_products = list(product(self.expertise_representation, repeat=len(self.expertise_domain)))
        if self.collaborator == "Generalist":
            unknown_representation = ["A", "B", "*"]
        elif self.collaborator == "Specialist":
            unknown_representation = ["0", "1", "2", "3", "*"]
        else:
            unknown_representation = ["*"]
        if len(self.expertise_domain) < self.N:
            # Cartesian products outside knowledge scope (not combinations or permutations)
            unknown_products = list(product(unknown_representation, repeat=self.N - len(self.expertise_domain)))
            all_states = []
            unknown_domain = [i for i in range(self.N) if i not in self.expertise_domain]
            for each_known in known_products:
                for each_unknown in unknown_products:
                    combined_list = np.zeros(self.N, dtype=object)
                    combined_list[self.expertise_domain] = each_known
                    combined_list[unknown_domain] = each_unknown
                    all_states.append(combined_list)
        else:
            all_states = known_products
        if len(all_states) != len(self.expertise_representation) ** len(self.expertise_domain) * \
                len(unknown_representation) ** (self.N - len(self.expertise_domain)):
            raise ValueError("All State is Problematic")
        for coarse_state in all_states:
            bits = ''.join(coarse_state)
            self.cache[bits] = self.calculate_coarse_fitness(coarse_state=coarse_state)

    def initialize(self):
        """
        Cache the fitness value
        :param norm: normalization
        :return: fitness cache
        """
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
        else:
            pass  # no normalization

    def query_coarse_fitness(self, coarse_state=None):
        return self.cache["".join(coarse_state)]

    @staticmethod
    def coarse_state_alternatives(coarse_state=None):
        alternative_pool = []
        for bit in coarse_state:
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

    def count_local_optima(self):
        counter = 0
        for key, value in self.cache.items():
            neighbor_list = self.get_neighbor_list(key=key)
            is_local_optima = True
            for neighbor in neighbor_list:
                if self.query_coarse_fitness(coarse_state=neighbor) > value:
                    is_local_optima = False
                    break
            if is_local_optima:
                counter += 1
        return counter

    def get_neighbor_list(self, key=None):
        """
        This is also for the Coarse Landscape
        :param key: string from the coarse landscape cache dict, e.g., "AABB"
        :return:list of the neighbor state, e.g., [["A", "A", "B", "A"], ["A", "A", "A", "B"]]
        """
        neighbor_list = []
        for index in range(self.N):
            neighbor = list(key)
            if neighbor[index] == "A":
                neighbor[index] = "B"
            elif neighbor[index] == "B":
                neighbor[index] = "A"
            neighbor_list.append(neighbor)
        return neighbor_list


if __name__ == '__main__':
    # Test Example
    import time
    from Landscape import Landscape
    from Generalist import Generalist
    from Specialist import Specialist
    t0 = time.time()
    N = 9
    K = 7
    state_num = 4
    expertise_amount = 16
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin")
    specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
    specialist.describe()
    coarse_landscape = CoarseLandscape(landscape=landscape, expertise_domain=specialist.expertise_domain,
                                       expertise_representation=specialist.expertise_representation)
    specialist.coarse_landscape = coarse_landscape
    specialist.update_cog_fitness()
    specialist.describe()
    coarse_landscape.describe()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
