# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import random
import time
from Landscape import Landscape
from Agent import Agent


class Simulator:
    """For individual level"""
    def __init__(self, N=0, state_num=4, landscape_iteration=5, agent_iteration=200, search_iteration=100):
        self.N = N
        self.state_num = state_num
        self.landscape = None  # update this value after landscape setting
        self.agent = None  # update this value after agent setting
        self.fitness = []
        self.team_level = False
        self.landscape_iteration = landscape_iteration
        self.agent_iteration = agent_iteration
        self.search_iteration = search_iteration

        # Some indicators for evaluation
        #  Match indicators
        self.match_g = None  # the match degree between IM and agent/team generalist knowledge domain
        self.match_s = None  # the match degree between IM and agent/team specialist knowledge domain
        self.match_overall = None

        # Knowledge domain indicator
        self.generalist_knowledge_domain_list = []  # the order is consistent with self.converged_fitness
        self.specialist_knowledge_domain_list = []

        # Search indicators
        self.converged_fitness = []
        self.converged_state = []
        self.unique_fitness_list = []  # record the path variance; or for the rank-fitness transition
        self.change_count = 0  # record the number of state change towards the convergence
        self.jump_out_of_local_optimal = 0 # record the agents' capability of skipping local optimal, due to cognitive search

    def set_landscape(self, K=0, k=0, IM_type=None,factor_num=0, influential_num=0):
        self.landscape = Landscape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=IM_type, K=K, k=k, factor_num=factor_num, influential_num=influential_num)
        self.landscape.initialize()

    def set_agent(self, name="None", lr=0, generalist_num=0, specialist_num=0):
        self.agent = Agent(N=self.N, lr=0, landscape=self.landscape, state_num=self.state_num)
        self.agent.type(name=name, generalist_num=generalist_num, specialist_num=specialist_num)

    def individual_run(self):
        """
        Given the iteration parameters, conduct the individual search
        :return: the fitness list [L1[A1, A2, ... AN], L2, ..., LN]
        """
        if self.team_level:
            raise ValueError("This is only for individual level search")
        fitness_landscape = []
        for landscape_loop in range(self.landscape_iteration):
            landscape = Landscape(N=self.N, state_num=self.state_num)
            landscape.type(IM_type="Random Directed", k=66)
            fitness_agent = []

            for agent_loop in range(self.agent_iteration):
                # print("Current landscape iteration: {0}; Agent iteration: {1}".format(self.landscape_iteration, self.agent_iteration))
                for search_loop in range(self.search_iteration):
                    # print("Search Loop: ", search_loop)
                    temp_fitness = self.agent.independent_search()
                    fitness_agent.append(temp_fitness)
                self.converged_fitness.append(fitness_agent[-1])
                self.generalist_knowledge_domain_list.append(self.agent.generalist_knowledge_domain)

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
    landscape_iteration = 10
    agent_iteration = 200
    search_iteration = 100
    k_list = [23, 33, 43]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
    IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
    generalist_list = [6, 0, 4, 2]
    specialist_list = [0, 3, 1, 2]

    # state = 10, E = 12 (=2*6; 4*3; 2*4+ 4*1; 2*2+4*2)

    for K in K_list:
        for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list):
            simulator = Simulator(N=N, state_num=state_num)
            A_fitness_landscape = []  # for the detailed fitness dynamic during search
            B_converged_fitness_landscape = []  # for the final converged fitness after search
            C_row_match_landscape = []  # for the weighted sum according to IM row
            C_column_match_landscape = [] # for the weighted sum according to IM column
            D_landscape_IM_list = []  # for the landscape IM
            E_knowledge_list_landscape = [] # for the agent knowledge in case we will change the weighted sum algorithm
            for landscape_loop in range(landscape_iteration):
                simulator.set_landscape(K=K, IM_type="Traditional Directed",factor_num=0, influential_num=0)
                A_fitness_agent = []
                B_converged_fitness_agent = []
                C_row_match_agent = []
                C_column_match_agent = []
                IM = simulator.landscape.IM.tolist()
                D_landscape_IM_list.append(IM)
                E_knowledge_list_agent = []
                for agent_loop in range(agent_iteration):
                    simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num, specialist_num=specialist_num)
                    A_fitness_search = []
                    for search_loop in range(search_iteration):
                        A_temp_fitness = simulator.agent.cognitive_local_search()
                        A_fitness_search.append(A_temp_fitness)
                    A_fitness_agent.append(A_fitness_search)
                    # B_converged_fitness_agent.append(A_fitness_search[-1]) # wrong
                    simulator.agent.state = simulator.agent.change_cog_state_to_state(
                        cog_state=simulator.agent.cog_state)
                    simulator.agent.converged_fitness = simulator.landscape.query_fitness(state=simulator.agent.state)
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
                A_fitness_landscape.append(A_fitness_agent)
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
                pickle.dump(A_fitness_landscape, out_file)
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
    # end_time = time.time()
    # print("Time used: ", end_time-start_time)
        # plt.plot(np.mean(np.mean(np.array(fitness_landscape), axis=0), axis=0))
        # plt.legend()
        # plt.show()


