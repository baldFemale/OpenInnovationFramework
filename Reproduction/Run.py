# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
from Simulator import Simulator
from MultiStateInfluentialLandscape import LandScape
from Agent import Agent
import time
import pickle

random.seed(1010)
start_time = time.time()
N = 10
state_num = 4
landscape_iteration = 5
agent_iteration = 200
search_iteration = 100
k_list = [23, 33, 43]
K_list = [2, 4, 6, 8, 10]
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
            simulator.set_landscape(K=K, IM_type="Traditional Directed", factor_num=0, influential_num=0)
            fitness_agent = []
            for agent_loop in range(agent_iteration):
                simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num,
                                    specialist_num=specialist_num)
                fitness_search = []
                for search_loop in range(search_iteration):
                    temp_fitness = simulator.agent.independent_search()
                    fitness_search.append(temp_fitness)
                fitness_agent.append(fitness_search)
                print("Current landscape iteration: {0}; Agent iteration: {1}".format(landscape_loop, agent_loop))
            fitness_landscape.append(fitness_agent)

        file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
                    '_K' + str(simulator.landscape.K) + '_k' + str(simulator.landscape.k) + '_E' + str(
            simulator.agent.element_num) + \
                    '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num)
        with open(file_name, 'wb') as out_file:
            pickle.dump(fitness_landscape, out_file)
end_time = time.time()
print("Time used: ", end_time - start_time)