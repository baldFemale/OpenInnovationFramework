# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run_population.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Simulator_transparency import Simulator
import pickle
import multiprocessing as mp


# Simulation Configuration
landscape_iteration = 100
agent_num = 400
search_iteration = 50
# Parameter
N = 6
state_num = 4
knowledge_num = 8
K_list = [1, 3, 5]
frequency_list = [1]
openness_list = [1.0]
quality_list = [1.0]
G_exposed_to_G_list = [0, 0.25, 0.5, 0.75, 1.0]
S_exposed_to_S_list = [0, 0.25, 0.5, 0.75, 1.0]
gs_proportion_list = [0.5]
exposure_type_list = ["Self-interested"]


def loop(k=0, K=0, exposure_type=None, socialization_freq=None, quality=None, openness=None, S_exposed_to_S=None,
         G_exposed_to_G=None, gs_proportion=None):
    # After simulators
    A_converged_potential_simulators = []
    B_converged_fitness_simulators = []
    C_unique_fitness_simulators = []
    for landscape_loop in range(landscape_iteration):
        simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration,
                              IM_type="Traditional Directed",
                              K=K, k=k, gs_proportion=gs_proportion, knowledge_num=knowledge_num,
                              exposure_type=exposure_type, openness=openness, quality=quality,
                              S_exposed_to_S=S_exposed_to_S, G_exposed_to_G=G_exposed_to_G)
        simulator.process(socialization_freq=socialization_freq)
        A_converged_potential_simulators.append(simulator.potential_after_convergence_landscape)
        B_converged_fitness_simulators.append(simulator.converged_fitness_landscape)
        C_unique_fitness_simulators.append(simulator.unique_fitness_landscape)

    basic_file_name = 'N' + str(N) + '_K' + str(K) + '_E' + str(knowledge_num) + '_' + \
                      exposure_type + '_SS' + str(S_exposed_to_S) + '_GG' + str(G_exposed_to_G) + '_F' +\
                      str(socialization_freq) + '_Prop' + str(gs_proportion) + "_Q" + str(quality) + "_O" + str(openness)
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
        for socialization_freq in frequency_list:
            for openness in openness_list:
                for quality in quality_list:
                    for G_exposed_to_G in G_exposed_to_G_list:
                        for S_exposed_to_S in S_exposed_to_S_list:
                            for gs_proportion in gs_proportion_list:
                                for exposure_type in exposure_type_list:
                                    p = mp.Process(target=loop,
                                                   args=(k, K, exposure_type, socialization_freq, quality, openness,
                                                         S_exposed_to_S, G_exposed_to_G, gs_proportion))
                                    p.start()