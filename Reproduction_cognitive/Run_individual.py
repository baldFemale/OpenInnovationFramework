# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:18
# @Author   : Junyi
# @FileName: Run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Simulator import Simulator
import pickle
import multiprocessing as mp

N = 10
state_num = 4
landscape_iteration = 500
agent_iteration = 500
search_iteration = 100
k_list = [14, 24, 34, 44, 54, 64, 74, 84]
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
generalist_list = [6, 0, 4, 2]
specialist_list = [0, 3, 1, 2]


def loop(k=0, K=0):
    for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list):
        simulator = Simulator(N=N, state_num=state_num)
        A_cog_fitness_landscape = []  # for the detailed fitness dynamic during search
        B_converged_fitness_landscape = []  # for the final converged fitness after search
        C_row_match_landscape = []  # for the weighted sum according to IM row
        C_column_match_landscape = [] # for the weighted sum according to IM column
        D_landscape_IM_list = []  # for the landscape IM
        E_knowledge_list_landscape = [] # for the agent knowledge in case we will change the weighted sum algorithm
        for landscape_loop in range(landscape_iteration):
            simulator.set_landscape(k=k, IM_type="Factor Directed",factor_num=0, influential_num=0)
            A_cog_fitness_agent = []
            B_converged_fitness_agent = []
            C_row_match_agent = []
            C_column_match_agent = []
            IM = simulator.landscape.IM.tolist()
            D_landscape_IM_list.append(IM)
            E_knowledge_list_agent = []
            for agent_loop in range(agent_iteration):
                simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num, specialist_num=specialist_num)
                A_cog_fitness_search = []
                for search_loop in range(search_iteration):
                    A_temp_cog_fitness = simulator.agent.cognitive_local_search()
                    A_cog_fitness_search.append(A_temp_cog_fitness)
                A_cog_fitness_agent.append(A_cog_fitness_search)
                # B_converged_fitness_agent.append(A_fitness_search[-1])  # wrong
                simulator.agent.state = simulator.agent.change_cog_state_to_state(cog_state=simulator.agent.cog_state)
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
                E_knowledge_list_agent.append([simulator.agent.specialist_knowledge_domain, simulator.agent.generalist_knowledge_domain])
                # print("Current landscape iteration: {0}; Agent iteration: {1}".format(landscape_loop, agent_loop))
            A_cog_fitness_landscape.append(A_cog_fitness_agent)
            B_converged_fitness_landscape.append(B_converged_fitness_agent)
            C_row_match_landscape.append(C_row_match_agent)
            C_column_match_landscape.append(C_column_match_agent)
            E_knowledge_list_landscape.append(E_knowledge_list_agent)

        # Output file name
        A_file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
                    '_K' + str(simulator.landscape.K) + '_k' + str(simulator.landscape.k) + '_E' + str(simulator.agent.element_num) + \
                    '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num)
        B_file_name_convergence = "2Convergence_" + A_file_name
        C_file_name_row_match = "3RowMatch_" + A_file_name
        C_file_name_column_match = "4ColumnMatch_" + A_file_name
        D_file_name_IM_information = "5IM_" + A_file_name
        E_file_name_agent_knowledge = "6Knowledge_" + A_file_name

        with open(A_file_name, 'wb') as out_file:
            pickle.dump(A_cog_fitness_landscape, out_file)
        with open(B_file_name_convergence, 'wb') as out_file:
            pickle.dump(B_converged_fitness_landscape, out_file)
        with open(C_file_name_row_match, 'wb') as out_file:
            pickle.dump(C_row_match_landscape, out_file)
        with open(C_file_name_column_match, 'wb') as out_file:
            pickle.dump(C_column_match_landscape, out_file)
        with open(D_file_name_IM_information, 'wb') as out_file:
            pickle.dump(D_landscape_IM_list, out_file)
        with open(E_file_name_agent_knowledge, 'wb') as out_file:
            pickle.dump(E_knowledge_list_landscape, out_file)

if __name__ == '__main__':
    for k in k_list:
        p = mp.Process(target=loop, args=(k,))
        p.start()