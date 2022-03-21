# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import multiprocessing as mp
from Simulator_resiliance import Simulator


# Parameters
N = 10
state_num = 4
parent_iteration = 20
landscape_num = 200
agent_num = 200
search_iteration = 100
landscape_search_iteration = 100
k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
K_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
IM_type = "Traditional Directed"
knowledge_num = 20


# test
# N = 8
# state_num = 4
# parent_iteration = 1
# landscape_num = 2
# agent_num = 2
# search_iteration = 2
# landscape_search_iteration = 2
# K_list = [1]
# # IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
# IM_type = "Traditional Directed"
# knowledge_num = 16

def loop(k=0, K=0,):
    # Parent Landscape Level
    A_converged_potential_parent = []
    B_converged_fitness_parent = []  # for the final converged fitness after search
    C_IM_list_parent = []  # for the landscape IM
    D_row_match_parent = []  # for the weighted sum according to IM row
    D_column_match_parent = []  # for the weighted sum according to IM column

    for parent_loop in range(parent_iteration):
        simulator = Simulator(N=N, state_num=state_num, landscape_num=landscape_num, agent_num=agent_num,
                              search_iteration=search_iteration,
                              landscape_search_iteration=landscape_search_iteration, IM_type=IM_type, knowledge_num=knowledge_num)
        simulator.set_parent_landscape()
        simulator.process()
        print("Ending process")
        A_converged_potential_parent.append(simulator.potential_after_convergence_landscape)
        B_converged_fitness_parent.append(simulator.converged_fitness_landscape)
        C_IM_list_parent.append(simulator.IM_dynamics)
        D_row_match_parent.append(simulator.row_match_landscape)
        D_column_match_parent.append(simulator.colummn_match_landscape)

    if k < 10:
        basic_file_name =  IM_type + '_N' + str(N) + \
                          '_K' + str(K) + '_k0' + str(k) + '_E' + str(knowledge_num)
    else:
        basic_file_name = IM_type + '_N' + str(N) + \
                          '_K' + str(K) + '_k' + str(k) + '_E' + str(knowledge_num)

    A_file_name_potential = "1Potential_" + basic_file_name
    B_file_name_convergence = "2Convergence_" + basic_file_name
    C_file_IM_list_parent = "3IM_" + basic_file_name
    D_file_row_match_parent = "4RowMatch_" + basic_file_name
    D_file_column_match_parent = "4ColumnMatch_" + basic_file_name

    with open(A_file_name_potential, 'wb') as out_file:
        pickle.dump(A_converged_potential_parent, out_file)
    with open(B_file_name_convergence, 'wb') as out_file:
        pickle.dump(B_converged_fitness_parent, out_file)
    with open(C_file_IM_list_parent, 'wb') as out_file:
        pickle.dump(C_IM_list_parent, out_file)
    with open(D_file_row_match_parent, 'wb') as out_file:
        pickle.dump(D_row_match_parent, out_file)
    with open(D_file_column_match_parent, 'wb') as out_file:
        pickle.dump(D_column_match_parent, out_file)


if __name__ == '__main__':
    k = 0
    for K in K_list:
        p = mp.Process(target=loop, args=(k, K))
        p.start()
        # loop(k=k, K=K)