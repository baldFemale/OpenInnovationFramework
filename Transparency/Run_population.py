# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run_population.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Simulator_transparency import Simulator
import pickle
from itertools import product
import multiprocessing as mp

N = 8
state_num = 4
landscape_iteration = 100
agent_num = 400
search_iteration = 100
# k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
# IM_type = ["Traditional Directed", "Factor Directed", "Influential Directed", "Random Directed"]
IM_type = "Traditional Directed"
knowledge_num = 16
# valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
exposure_type = "Self-interested"
# exposure_type = "Overall-ranking"
# exposure_type = "Random"
# Frequency & Openness
frequency_list = [1, 5, 10, 20, 50]
openness_list = [0.2, 0.4, 0.8, 1.0]
# Quality
quality_list = [0.1, 0.2, 0.4, 0.8, 1.0]
# Directions
G_exposed_to_G_list = [0, 0.4, 0.8, 1.0]
S_exposed_to_S_list = [0, 0.4, 0.8, 1.0]
gs_proportion_list = [0.1, 0.2, 0.4, 0.8, 1.0]  # the proportion of G
gs_proportion = 0.5


def loop(k=0, K=0, socialization_freq=None, quality=None, openness=None, S_exposed_to_S=None,
         G_exposed_to_G=None, parallel_flag=None,):
    # After simulators
    A_converged_potential_simulators = []
    B_converged_fitness_simulators = []
    C_unique_fitness_simulators = []
    for landscape_loop in range(landscape_iteration):
        simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration,
                              IM_type=IM_type,
                              K=K, k=k, gs_proportion=0.5, knowledge_num=knowledge_num,
                              exposure_type=exposure_type, openness=openness, quality=quality,
                              S_exposed_to_S=S_exposed_to_S, G_exposed_to_G=G_exposed_to_G)
        simulator.process(socialization_freq=socialization_freq)
        A_converged_potential_simulators.append(simulator.potential_after_convergence_landscape)
        B_converged_fitness_simulators.append(simulator.converged_fitness_landscape)
        C_unique_fitness_simulators.append(simulator.unique_fitness_landscape)

    if parallel_flag:
        basic_file_name = 'N' + str(N) + '_K' + str(K) + '_E' + str(knowledge_num) + '_' + \
                          exposure_type + '_SS' + S_exposed_to_S + '_GG' + G_exposed_to_G + '_F' +\
                          str(socialization_freq) + '_Prop' + str(gs_proportion) + 'flag_' + str(parallel_flag)
    else:
        basic_file_name = 'N' + str(N) + '_K' + str(K) + '_k0' + str(k) + '_E' + str(knowledge_num) + '_' + \
                          exposure_type + '_SS' + S_exposed_to_S + '_GG' + G_exposed_to_G + '_F' +\
                          str(socialization_freq) + '_Prop' + str(gs_proportion)
    A_file_name_potential = "1Potential_" + basic_file_name
    B_file_name_convergence = "2Convergence_" + basic_file_name
    C_file_unique_fitness = "3Unique_" + basic_file_name

    with open(A_file_name_potential, 'wb') as out_file:
        pickle.dump(A_converged_potential_simulators, out_file)
    with open(B_file_name_convergence, 'wb') as out_file:
        pickle.dump(B_converged_fitness_simulators, out_file)
    with open(C_file_unique_fitness, 'wb') as out_file:
        pickle.dump(C_unique_fitness_simulators, out_file)


if __name__ == '__main__':
    k = 0
    quality = None
    openness = 1
    alternative_pool = [G_exposed_to_G_list, S_exposed_to_S_list]
    for K in K_list:
        for G_exposed_to_G,  S_exposed_to_S in [i for i in product(*alternative_pool)]:
            for socialization_freq in frequency_list:
                # for openness in openness_list:
                    # for quality in quality_list:
                p = mp.Process(target=loop, args=(k, K, socialization_freq, quality, openness, S_exposed_to_S,G_exposed_to_G))
                p.start()
        # loop(k=0, K=0, transparency_direction=None, socialization_freq=None)


    # k = 0
    # flag = 0
    # for K in K_list:
    #     flag += 1
    #     p = mp.Process(target=loop, args=(k, K, flag))
    #     p.start()

    # k = 0
    # for K in K_list:
    #     for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list):
    #         loop(k, K, each_agent_type, generalist_num, specialist_num)
    #         break
    #     break

    # data = []
    # with open(r'1Potential_Generalist_Traditional Directed_N10_K0_k0_E12_G6_S0', 'rb') as infile:
    #     # print(each_file)
    #     temp = pickle.load(infile)
    #     data.append(temp)
    # print(data)
