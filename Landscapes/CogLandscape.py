# -*- coding: utf-8 -*-
# @Time     : 5/11/2023 15:26
# @Author   : Junyi
# @FileName: Cog_landscape.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from collections import defaultdict
from itertools import product
import numpy as np


class CogLandscape:
    def __init__(self, landscape=None, expertise_domain=None, expertise_representation=None, norm=True):
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
        self.initialize()

    def describe(self):
        print("*********CogLandScape information********* ")
        print("LandScape shape of (N={0}, K={1}, state number={2})".format(self.N, self.K, self.state_num))
        print("Influential matrix: \n", self.IM)
        print("Influential dependency map: ", self.dependency_map)
        print("********************************")

    def calculate_cog_fitness(self, cog_state=None):
        partial_fitness_alternatives = []
        alternatives = self.cog_state_alternatives(cog_state=cog_state)  # get back to the finest level
        # so that we can calculate it using FC, which is all at the finest level (i.e., 0, 1, 2, 3)
        for state in alternatives:
            partial_FC_across_bits = []
            for index in range(len(state)):
                if index not in self.expertise_domain:
                    continue
                dependency = self.dependency_map[index]
                bit_index = "".join([str(state[j]) for j in dependency])
                bit_index = str(state[index]) + bit_index
                FC_index = int(bit_index, self.state_num)
                partial_FC_across_bits.append(self.FC[index][FC_index])
            partial_fitness_state = sum(partial_FC_across_bits) / len(self.expertise_domain)
            partial_fitness_alternatives.append(partial_fitness_state)
        return sum(partial_fitness_alternatives) / len(partial_fitness_alternatives)

    def store_cache(self):
        # Cartesian products within knowledge scope (not combinations or permutations)
        known_products = list(product(self.expertise_representation, repeat=len(self.expertise_domain)))
        all_representation = ["A", "B", "0", "1", "2", "3", "*"]
        # Cartesian products outside knowledge scope (not combinations or permutations)
        unknown_products = list(product(all_representation, repeat=self.N - len(self.expertise_domain)))
        all_cog_states = []
        unknown_domain = [i for i in range(self.N) if i not in self.expertise_domain]
        for each_known in known_products:
            for each_unknown in unknown_products:
                combined_list = np.zeros(self.N, dtype=object)
                combined_list[self.expertise_domain] = each_known
                combined_list[unknown_domain] = each_unknown
                all_cog_states.append(combined_list)
        if len(all_cog_states) != len(self.expertise_representation) ** len(self.expertise_domain) * \
                len(all_representation) ** (self.N - len(self.expertise_domain)):
            raise ValueError("All State is Problematic")
        for cog_state in all_cog_states:
            bits = ''.join(cog_state)
            self.cache[bits] = self.calculate_cog_fitness(cog_state)

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
        if self.norm:
            for k in self.cache.keys():
                # self.cache[k] = (self.cache[k] - self.min_normalizer) / (self.max_normalizer - self.min_normalizer)
                self.cache[k] = self.cache[k] / self.max_normalizer

    def query_cog_fitness(self, cog_state=None):
        return self.cache["".join(cog_state)]

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
        return [i for i in product(*alternative_pool)]


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
    landscape = Landscape(N=N, K=K, state_num=state_num)
    specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
    specialist.describe()
    cog_landscape = CogLandscape(landscape=landscape, expertise_domain=specialist.expertise_domain, expertise_representation=specialist.expertise_representation)
    specialist.cog_landscape = cog_landscape
    specialist.update_cog_fitness()
    specialist.describe()
    cog_landscape.describe()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))




