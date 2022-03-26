# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run_population.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Simulator_transparency import Simulator
import pickle
import multiprocessing as mp

N = 8
state_num = 4
landscape_iteration = 100
agent_num = 500
search_iteration = 100
k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
# IM_type = ["Traditional Directed", "Factor Directed", "Influential Directed", "Random Directed"]
IM_type = "Traditional Directed"
knowledge_num = 16
transparency_direction_list = ["A", "G", "S", "SG", "GS"]
exposure_type = "Self-interested"
socialization_freq_list = [1, 5, 10, 20, 50]
gs_proportion = 0.5


def loop(k=0, K=0, transparency_direction=None, socialization_freq=None, parallel_flag=None):
    # After simulators
    A_converged_potential_simulators = []
    B_converged_fitness_simulators = []
    C_unique_fitness_simulators = []
    for landscape_loop in range(landscape_iteration):
        simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration, IM_type=IM_type,
                     K=K, k=k, gs_proportion=gs_proportion, knowledge_num=knowledge_num,
                     exposure_type=exposure_type, transparency_direction=transparency_direction)
        simulator.process(socialization_freq=socialization_freq)
        A_converged_potential_simulators.append(simulator.potential_after_convergence_landscape)
        B_converged_fitness_simulators.append(simulator.converged_fitness_landscape)
        C_unique_fitness_simulators.append(simulator.unique_fitness_landscape)

    if (k < 10) & (K == 0):
        if parallel_flag:
            basic_file_name = 'N' + str(N) + '_K' + str(K) + '_k0' + str(k) + '_E' + str(knowledge_num) + '_' + \
                              exposure_type + '_' + transparency_direction + '_F' +\
                              str(socialization_freq) + '_' + str(gs_proportion) + 'flag_' + str(parallel_flag)
        else:
            basic_file_name = 'N' + str(N) + '_K' + str(K) + '_k0' + str(k) + '_E' + str(knowledge_num) + '_' + \
                              exposure_type + '_' + transparency_direction + '_F' + \
                              str(socialization_freq) + '_' + str(gs_proportion)
    else:
        if parallel_flag:
            # extend the repeated K_list to make it parallel and utilize the CPU pool
            basic_file_name = "N" + str(N) + '_K' + str(K) + '_k' + str(k) + '_E' + str(knowledge_num) + '_' + \
                              exposure_type + '_' + transparency_direction + '_F' + \
                              str(socialization_freq) + '_' + str(gs_proportion) + 'flag_' + str(parallel_flag)
        else:
            basic_file_name = 'N' + str(N) + '_K' + str(K) + '_k' + str(k) + '_E' + str(knowledge_num) + '_' + \
                              exposure_type + '_' + transparency_direction + '_F' + \
                              str(socialization_freq) + '_' + str(gs_proportion)

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
    for K in K_list:
        for transparency_direction in transparency_direction_list:
            for socialization_freq in socialization_freq_list:
                p = mp.Process(target=loop, args=(k, K, transparency_direction, socialization_freq))
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
