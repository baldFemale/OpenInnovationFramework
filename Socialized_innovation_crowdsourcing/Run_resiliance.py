# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import multiprocessing as mp
import math
from Simulator_resiliance import Simulator
from Socialized_Agent import Agent

# Parameters
N = 10
state_num = 4
parent_iteration = 20
landscape_iteration = 200
agent_iteration = 200
search_iteration = 100
k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
# generalist_list = [6, 0, 4, 2]
# specialist_list = [0, 3, 1, 2]
# totally 40 elements; try more knowledgeable setting: 36
generalist_list = [10, 0, 4, 6]
specialist_list = [0, 5, 3, 2]


def loop(k=0, K=0, IM_type=None, each_agent_type=None, generalist_num=None, specialist_num=None):
    """
    Task loop for multiple processing
    :param each_agent_type:
    :param generalist_num:
    :param specialist_num:
    :return: pickled data, 500*500 repetitions (i.e.,Landscape iteration and Agent iteration)
                for each setting (i.e., agent type and IM type) (around 10 Ks with 4 agent types)
                ultimately, data shape [10, 4, 500, 500]
    """

    simulator = Simulator(N=N, state_num=state_num)
    # Parent Landscape Level
    A_converged_potential_parent = []
    B_converged_fitness_parent = []  # for the final converged fitness after search
    C_row_match_parent = []  # for the weighted sum according to IM row
    C_column_match_parent = []  # for the weighted sum according to IM column
    D_IM_list_parent = []  # for the landscape IM
    E_knowledge_list_parent = []  # for the agent knowledge in case we will change the weighted sum algorithm
    for parent_loop in range(parent_iteration):
        simulator.set_parent_landscape()
        IM_dynamics_list = [] # only with the same parent landscape to generate dynamic landscape gradually
        A_converged_potential_landscape = []
        B_converged_fitness_landscape = []  # for the final converged fitness after search
        C_row_match_landscape = []  # for the weighted sum according to IM row
        C_column_match_landscape = []  # for the weighted sum according to IM column
        D_IM_list_landscape = []  # for the landscape IM -> one of the output files
        E_knowledge_list_landscape = []  # for the agent knowledge in case we will change the weighted sum algorithm
        for landscape_loop in range(landscape_iteration):
            if len(IM_dynamics_list) == 0:
                simulator.set_dynamic_landscape(K=K, k=k, IM_type=IM_type, factor_num=0, influential_num=0)
            else:
                simulator.set_dynamic_landscape(K=K, k=k, IM_type=IM_type, factor_num=0, influential_num=0,
                                                previous_IM=IM_dynamics_list[-1], IM_change_bit=1)
            IM_dynamics_list.append(simulator.landscape.IM)
            # Landscape Level Indicator
            A_converged_potential_agent = []
            B_converged_fitness_agent = []
            C_row_match_agent = []
            C_column_match_agent = []
            IM_ = simulator.landscape.IM.tolist()
            D_IM_list_landscape.append(IM_)  # only this doesn't need to get into Agent loop
            E_knowledge_list_agent = []

            for agent_loop in range(agent_iteration):
                simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num,
                                    specialist_num=specialist_num)
                for search_loop in range(search_iteration):
                    simulator.agent.cognitive_local_search()
                potential_after_convergence = simulator.landscape.query_potential_performance(cog_state=simulator.agent.cog_state, top=1)
                A_converged_potential_agent.append(potential_after_convergence)
                simulator.agent.state = simulator.agent.change_cog_state_to_state(
                    cog_state=simulator.agent.cog_state)
                simulator.agent.converge_fitness = simulator.landscape.query_fitness(state=simulator.agent.state)
                B_converged_fitness_agent.append(simulator.agent.converge_fitness)
                # Weighted sum for the match between landscape IM and agent knowledge
                C_row_match_temp = 0
                for row in range(simulator.N):
                    if row in simulator.agent.specialist_knowledge_domain:
                        C_row_match_temp += sum(IM[row]) * simulator.state_num
                    if row in simulator.agent.generalist_knowledge_domain:
                        C_row_match_temp += sum(IM[row]) * simulator.state_num * simulator.agent.gs_ratio
                C_column_match_temp = 0
                for column in range(simulator.N):
                    if column in simulator.agent.specialist_knowledge_domain:
                        C_column_match_temp += sum(IM[:][column]) * simulator.agent.state_num
                    if column in simulator.agent.generalist_knowledge_domain:
                        C_column_match_temp += sum(IM[:][column]) * simulator.agent.state_num * simulator.agent.gs_ratio

                C_row_match_agent.append(C_row_match_temp)
                C_column_match_agent.append(C_column_match_temp)
                E_knowledge_list_agent.append(
                    [simulator.agent.specialist_knowledge_domain, simulator.agent.generalist_knowledge_domain])
            A_converged_potential_landscape.append(A_converged_potential_agent)
            B_converged_fitness_landscape.append(B_converged_fitness_agent)
            C_row_match_landscape.append(C_row_match_agent)
            C_column_match_landscape.append(C_column_match_agent)
            E_knowledge_list_landscape.append(E_knowledge_list_agent)

        A_converged_potential_parent.append(A_converged_potential_landscape)
        B_converged_fitness_parent.append(B_converged_fitness_landscape)
        C_row_match_parent.append(C_row_match_landscape)
        C_column_match_parent.append(C_column_match_landscape)
        D_IM_list_parent.append(D_IM_list_landscape)
        E_knowledge_list_parent.append(E_knowledge_list_landscape)

    # Output file name  # fix the bug when evaluate the performance curve, the k=4 will go to position of the k=44
    if simulator.landscape.k < 10:
        basic_file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
                          '_K' + str(simulator.landscape.K) + '_k0' + str(simulator.landscape.k) + '_E' + str(
            simulator.agent.element_num) + \
                          '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num) \
                          + "_D"
    else:
        basic_file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
                          '_K' + str(simulator.landscape.K) + '_k' + str(simulator.landscape.k) + '_E' + str(
            simulator.agent.element_num) + \
                          '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num) \
                          + "_D"

    A_file_name_potential = "1Potential_" + basic_file_name
    B_file_name_convergence = "2Convergence_" + basic_file_name
    C_file_name_row_match = "3RowMatch_" + basic_file_name
    C_file_name_column_match = "4ColumnMatch_" + basic_file_name
    D_file_name_IM_information = "5IM_" + basic_file_name
    E_file_name_agent_knowledge = "6Knowledge_" + basic_file_name

    with open(A_file_name_potential, 'wb') as out_file:
        pickle.dump(A_converged_potential_parent, out_file)
    with open(B_file_name_convergence, 'wb') as out_file:
        pickle.dump(B_converged_fitness_parent, out_file)
    with open(C_file_name_row_match, 'wb') as out_file:
        pickle.dump(C_row_match_parent, out_file)
    with open(C_file_name_column_match, 'wb') as out_file:
        pickle.dump(C_column_match_parent, out_file)
    with open(D_file_name_IM_information, 'wb') as out_file:
        pickle.dump(D_IM_list_parent, out_file)
    with open(E_file_name_agent_knowledge, 'wb') as out_file:
        pickle.dump(E_knowledge_list_parent, out_file)


if __name__ == '__main__':
    K = 0
    for k in k_list:
        absolute_k = K if K else k // 10
        # the total number of alternative combination given N and K
        # each iteration, we choose one specific dependency distribution (e.g., [0,1,2,3]) and fix it.
        combination_count = int(math.factorial(N)/math.factorial(absolute_k)/math.factorial((N-absolute_k)))
        # add one more iteration level to get the relative but independent landscape
        for dynamic_flag in range(combination_count):
            for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list, ):
                p = mp.Process(target=loop, args=(k, K, each_agent_type, generalist_num, specialist_num, dynamic_flag))
                p.start()

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
