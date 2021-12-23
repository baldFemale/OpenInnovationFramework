# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import random
import time
from MultiStateInfluentialLandscape import LandScape
from Agent import Agent
import numpy as np


class Simulator:
    def __init__(self, N=0, state_num=4, landscape_iteration=5, agent_iteration=200):
        self.N = N
        self.state_num = state_num
        self.landscape = None  # update this value after landscape setting
        self.agent = None  # update this value after agent setting
        self.fitness = []
        self.team_level = False
        self.landscape_iteration = landscape_iteration
        self.agent_iteration = agent_iteration
        self.search_iteration = search_iteration

    def set_landscape(self, K=0, k=0, IM_type=None,factor_num=0, influential_num=0):
        self.landscape = LandScape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=IM_type, K=K, k=k, factor_num=factor_num, influential_num=influential_num)
        self.landscape.initialize()
        self.landscape.describe()

    def set_agent(self, name="None", lr=0, generalist_num=0, specialist_num=0):
        self.agent = Agent(N=self.N, lr=0, landscape=self.landscape, state_num=self.state_num)
        self.agent.type(name=name, generalist_num=generalist_num, specialist_num=specialist_num)
        self.agent.describe()

    def individual_run(self):
        """
        Given the iteration parameters, conduct the individual search
        :return: the fitness list [L1[A1, A2, ... AN], L2, ..., LN]
        """
        if self.team_level:
            raise ValueError("This is only for individual level search")
        fitness_landscape = []
        for landscape_loop in range(self.landscape_iteration):

            landscape = LandScape(N=self.N, state_num=self.state_num)
            landscape.type(IM_type="Random Directed", k=66)
            landscape.initialize()

            fitness_agent = []
            for agent_loop in range(self.agent_iteration):
                print("Current landscape iteration: {0}; Agent iteration: {1}".format(self.landscape_iteration, self.agent_iteration))
                for search_loop in range(self.search_iteration):
                    print("Search Loop: ", search_loop)
                    temp_fitness = self.agent.independent_search()
                    fitness_agent.append(temp_fitness)
            fitness_landscape.append(fitness_agent)

        file_name = self.agent.name + '_N' + str(self.agent.N) + '_K' + str(self.landscape.K) + \
                    '_k' + str(self.landscape.k) + '_E' + str(self.agent.element_num)
        with open(file_name, 'wb') as out_file:
            pickle.dump(fitness_landscape, out_file)
        return fitness_landscape


if __name__ == '__main__':
    # Test Example (Waiting for reshaping into class above)
    # The test code below works.
    start_time = time.time()
    random.seed(1024)
    N = 10
    state_num = 4
    landscape_iteration = 5
    agent_iteration = 200
    search_iteration = 100
    k_list = [23, 33, 43]
    K_list = [2, 4, 6, 8]
    agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
    IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
    generalist_list = [6, 0, 4, 2]
    specialist_list = [0, 3, 1, 2]

    # state = 10, E = 12 (=2*6; 4*3; 2*4+ 4*1; 2*2+4*2)
    for K in K_list:
        for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list):
            simulator = Simulator(N=N, state_num=state_num)
            fitness_landscape = []
            for landscape_loop in range(landscape_iteration):
                simulator.set_landscape(K=K, IM_type="Traditional Directed",factor_num=0, influential_num=0)
                fitness_agent = []
                for agent_loop in range(agent_iteration):
                    simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num, specialist_num=specialist_num)
                    fitness_search = []
                    for search_loop in range(search_iteration):
                        temp_fitness = simulator.agent.independent_search()
                        fitness_search.append(temp_fitness)
                    fitness_agent.append(fitness_search)
                    print("Current landscape iteration: {0}; Agent iteration: {1}".format(landscape_loop, agent_loop))
                fitness_landscape.append(fitness_agent)

            file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
                        '_K' + str(simulator.landscape.K) + '_k' + str(simulator.landscape.k) + '_E' + str(simulator.agent.element_num) + \
                        '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num)
            with open(file_name, 'wb') as out_file:
                pickle.dump(fitness_landscape, out_file)
    end_time = time.time()
    print("Time used: ", end_time-start_time)
        # plt.plot(np.mean(np.mean(np.array(fitness_landscape), axis=0), axis=0))
        # plt.legend()
        # plt.show()


