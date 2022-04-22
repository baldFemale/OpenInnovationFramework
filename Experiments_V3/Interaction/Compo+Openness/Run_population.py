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
landscape_iteration = 1000
agent_num = 200
search_iteration = 100
# Parameter
N = 9
state_num = 4
knowledge_num = 16
K_list = [2, 4, 6, 8]
frequency_list = [1]
openness_list = [0, 0.25, 0.5, 0.75, 1.0]
quality_list = [1.0]
G_exposed_to_G_list = [0.5]
S_exposed_to_S_list = [0.5]
gs_proportion_list = [0, 0.25, 0.5, 0.75, 1.0]
exposure_type_list = ["Self-interested"]


def loop(k=0, K=0, exposure_type=None, socialization_freq=None, quality=None, openness=None, S_exposed_to_S=None,
         G_exposed_to_G=None, gs_proportion=None):
    # After simulators
    A_average_fitness_simulators = []
    A_average_fitness_rank_simulators = []
    B_surface_divergence_G_simulators = []
    B_surface_divergence_S_simulators = []
    C_surface_quality_G_simulators = []
    C_surface_quality_S_simulators = []
    D_surface_utilization_G_simulators = []
    D_surface_utilization_S_simulators = []

    for landscape_loop in range(landscape_iteration):
        simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration,
                              IM_type="Traditional Directed",
                              K=K, k=k, gs_proportion=gs_proportion, knowledge_num=knowledge_num,
                              exposure_type=exposure_type, openness=openness, quality=quality,
                              S_exposed_to_S=S_exposed_to_S, G_exposed_to_G=G_exposed_to_G)
        simulator.process(socialization_freq=socialization_freq)
        A_average_fitness_simulators.append(simulator.converged_fitness_landscape)
        A_average_fitness_rank_simulators.append(simulator.converged_fitness_rank_landscape)
        B_surface_divergence_G_simulators.append(simulator.surface_divergence_G_landscape)
        B_surface_divergence_S_simulators.append(simulator.surface_divergence_S_landscape)
        C_surface_quality_G_simulators.append(simulator.surface_quality_G_landscape)
        C_surface_quality_S_simulators.append(simulator.surface_quality_S_landscape)
        D_surface_utilization_G_simulators.append(simulator.surface_utilization_G_landscape)
        D_surface_utilization_S_simulators.append(simulator.surface_utilization_S_landscape)

    basic_file_name = 'N' + str(N) + '_K' + str(K) + '_E' + str(knowledge_num) + '_' + \
                      exposure_type + '_SS' + str(S_exposed_to_S) + '_GG' + str(G_exposed_to_G) + '_F' +\
                      str(socialization_freq) + '_Prop' + str(gs_proportion) + "_Q" + str(quality) + "_O" + str(openness) + "_"
    A_file_name_average = "1Average_" + basic_file_name
    A_file_name_average_rank = "2AverageRank_" + basic_file_name
    B_file_name_surface_divergence_G = "3DivergenceG_" + basic_file_name
    B_file_name_surface_divergence_S = "3DivergenceS_" + basic_file_name
    C_file_name_surface_quality_G = "4QualityG" + basic_file_name
    C_file_name_surface_quality_S = "4QualityS" + basic_file_name
    D_file_name_surface_utilization_G = "5UtilizationG" + basic_file_name
    D_file_name_surface_utilization_S = "5UtilizationS" + basic_file_name

    # outcome
    with open(A_file_name_average, 'wb') as out_file:
        pickle.dump(A_average_fitness_simulators, out_file)
    with open(A_file_name_average_rank, 'wb') as out_file:
        pickle.dump(A_average_fitness_rank_simulators, out_file)
    # mechanism
    with open(B_file_name_surface_divergence_G, 'wb') as out_file:
        pickle.dump(B_surface_divergence_G_simulators, out_file)
    with open(B_file_name_surface_divergence_S, 'wb') as out_file:
        pickle.dump(B_surface_divergence_S_simulators, out_file)

    with open(C_file_name_surface_quality_G, 'wb') as out_file:
        pickle.dump(C_surface_quality_G_simulators, out_file)
    with open(C_file_name_surface_quality_S, 'wb') as out_file:
        pickle.dump(C_surface_quality_S_simulators, out_file)

    with open(D_file_name_surface_utilization_G, 'wb') as out_file:
        pickle.dump(D_surface_utilization_G_simulators, out_file)
    with open(D_file_name_surface_utilization_S, 'wb') as out_file:
        pickle.dump(D_surface_utilization_S_simulators, out_file)


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
