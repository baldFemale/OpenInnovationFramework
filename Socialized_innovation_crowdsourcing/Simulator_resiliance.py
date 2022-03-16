# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import matplotlib.pyplot as plt
import time
# new compositions for dynamic/socialization framework
from DyLandscape_2 import DyLandscape
from ParentLandscape import ParentLandscape
from Socialized_Agent import Agent


class Simulator:
    """One simulator represent one Parent Landscape iteration"""
    def __init__(self, N=10, state_num=4, landscape_num=200, agent_num=200, search_iteration=100,
                 landscape_search_iteration=100, IM_type=None, K=0, k=0, gs_proportion=0.5, knowledge_num=20):
        # Parent Landscape
        self.parent = None  # will be stable within one simulator
        self.N = N
        self.state_num = state_num

        # Chile Landscape
        self.landscapes = None  # will change over time
        self.landscape_num = landscape_num
        self.state_pool = []
        self.state_pool_rank = {}
        self.IM_dynamics = []
        self.IM_type = IM_type
        self.K = K
        self.k = k
        self.IM_change_bit = 1

        # Agent crowd
        self.agents = []  # will be stable; only the initial state and search will change
        self.agent_num = agent_num
        self.knowledge_num = knowledge_num
        self.gs_proportion = gs_proportion
        self.generalist_num = int(self.agent_num * gs_proportion)
        self.specialist_num = int(self.agent_num * (1-gs_proportion))

        # Cognitive Search
        self.search_iteration = search_iteration
        self.landscape_search_iteration = landscape_search_iteration

        # Some indicators for evaluation
        # convergence
        # list structure: agents -> multiple crowdsourcing (for one child landscape) -> multiple child landscape (for one parent)
        self.converged_fitness_agent = [] # temp
        self.converged_fitness_landscape = [] # child landscapes; or one parent
        self.potential_after_convergence_agent = [] # temp
        self.potential_after_convergence_landscape = []
        self.row_match_landscape = []
        self.colummn_match_landscape = []  # only in the child landscape level; match degree across agents is added up.

    def set_parent_landscape(self):
        self.parent = ParentLandscape(N=self.N, state_num=self.state_num)

    def set_child_landscape(self):
        if not self.parent:
            raise ValueError("Need to build parent landscape firstly")
        self.landscape = None  # delete the previous one
        self.landscape = DyLandscape(N=self.N, state_num=self.state_num, parent=self.parent)
        self.landscape.type(IM_type=self.IM_type, K=self.K, k=self.k,
                            previous_IM=self.IM_dynamics[-1], IM_change_bit=self.IM_change_bit)
        self.landscape.initialize()
        self.IM_dynamics.append(self.landscape.IM)  # extend the IM history

    def set_agent(self):
        """
        Fix the knowledge of Agents pool; but their initial state and cognitive fitness will change
        Initial state will change due to the socialization
        Cognitive fitness/search will change due to the dynamic landscape
        the cognitive fitness is assessed in the landscape level
        :param knowledge: agents have the same number of knowledge (this may change to make it closer to the real distribution)
        :param gs_proportion: the dynamic GS proportion, to some extend, reflects the composition dynamic of platform,
        even given no diversity of knowledge number
        :return: the agents with fixed knowledge distribution
        """
        if not self.parent:
            raise ValueError("Need to build parent landscape firstly")
        if not self.landscape:
            raise ValueError("Need to build child landscape secondly")
        for _ in range(self.generalist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            generalist_num = int(self.knowledge_num/2)
            specialist_num = 0
            agent = Agent(N=self.N, lr=0, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Generalist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)
        for _ in range(self.specialist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            specialist_num = int(self.knowledge_num/4)
            generalist_num = 0
            agent = Agent(N=self.N, lr=0, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Specialist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)
        # the knowledge distribution of the crowd is meaningless or unoriginal
        # in this paper, we focus on the proportion instead of diversity.
        # for agent in self.agents:
        #     self.knowledge_list_landscape.append(agent.generalist_knowledge_domain)

    def crowd_search_once(self):
        """
        For each landscape, there might need several iteration to make the crowd get convergency.
        For each search and learning, there will generate a list of performance (across agents) *Single list*
        But for Simulator, representing a Parent, the data will be two more levels
        *child level*: one child landscape, need several steps to converge; we just record the whole process toward convergency
        *parent level*: one parent level will include multiple child landscapes (e.g., 200)
        :return: the performance outcomes at the Simulator level (i.e., parent landscape level)
        """
        potential_after_convergence_agent = []
        converged_fitness_agent = []
        for agent in self.agents:
            for _ in range(self.search_iteration):
                agent.cognitive_local_search()
            potential_after_convergence = agent.landscape.query_potential_performance(cog_state=agent.cog_state, top=1)
            potential_after_convergence_agent.append(potential_after_convergence)
            agent.state = agent.change_cog_state_to_state(cog_state=agent.cog_state)
            agent.converge_fitness = agent.landscape.query_fitness(state=agent.state)
            converged_fitness_agent.append(agent.converge_fitness)
        self.converged_fitness_agent = converged_fitness_agent
        self.potential_after_convergence_agent = potential_after_convergence_agent

    def crowd_search_forward(self):
        """
        Difference: agents need to form the pool and update the initial state
        Corresponding to the child landscape level, where we can get the aggregated outcome variables.
        :return: accumulative outcome variables
        """
        for _ in range(self.landscape_search_iteration):
            self.creat_state_pool_and_rank()
            self.change_initial_state()
            self.crowd_search_once()
            # !!!
            # record the whole process toward child landscape convergence
            # may only need to record the last performance
            self.potential_after_convergence_landscape.append(self.potential_after_convergence_agent)
            self.converged_fitness_landscape.append(self.converged_fitness_agent)

    def creat_state_pool_and_rank(self):
        """
        after each crowd search, create the state pool and make the agents vote for it.
        :return: the crowd-level state pool and its overall rank
        """
        self.state_pool = [agent.state for agent in self.agents]
        self.state_pool = list(set(self.state_pool))
        overall_pool_rank = {}
        for state in self.state_pool:
            overall_pool_rank["".join(state)] = 0
        for agent in self.agents:
            agent.state_pool = self.state_pool
            personal_pool_rank = agent.vote_for_state_pool()
            # format: {"10102": 0.85} state string: cognitive fitness
            for key, value in personal_pool_rank.items():
                overall_pool_rank[key] += value
        self.state_pool_rank = overall_pool_rank

    def change_working_landscape(self):
        # change the child landscape, based on *previous IM*
        self.set_child_landscape()
        # assign the new landscape to all agents
        for agent in self.agents:
            agent.landscape = self.landscape

    def change_initial_state(self, exposure_type=None):
        """
        After we create the state pool and its rank, agents can update their initial state
        :param exposure_type: only three valid exposure types
        :return:all agents update their initial state
        """
        if not self.parent:
            raise ValueError("Need to build parent landscape firstly")
        if not self.landscape:
            raise ValueError("Need to build child landscape secondly")
        valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
        if exposure_type not in valid_exposure_type:
            raise ValueError("Only support: ", valid_exposure_type)
        if exposure_type == "Overall-ranking":
            self.creat_state_pool_and_rank()  # get the pool rank
            for agent in self.agents:
                agent.assigned_state_pool_rank = self.state_pool_rank
        for agent in self.agents:
            agent.update_state_from_exposure(exposure_type=exposure_type)

    def get_match(self):
        if not self.landscape:
            raise ValueError("NO landscape")
        if not self.agents:
            raise ValueError("NO agents")
        column_mach = 0
        row_match = 0
        for agent in self.agents:
            for column in range(self.N):
                if column in agent.specialist_knowledge_domain:
                    column_mach += sum(self.landscape.IM[:][column]) * agent.state_num
                elif column in agent.generalist_knowledge_domain:
                    column_mach += sum(self.landscape.IM[:][column]) * agent.state_num * 0.5
        for agent in self.agents:
            for row in range(self.N):
                if row in agent.specialist_knowledge_domain:
                    column_mach += sum(self.landscape.IM[row][:]) * agent.state_num
                elif row in agent.generalist_knowledge_domain:
                    column_mach += sum(self.landscape.IM[row][:]) * agent.state_num * 0.5
        self.row_match_landscape.append(int(row_match))
        self.colummn_match_landscape.append(int(row_match))

    def process(self):
        """
        Process to run one parent landscape loop
        :return: the outcomes in the parent level
        """
        self.set_parent_landscape()
        # the first child landscape have no previous reference; previous_IM is none.
        self.set_child_landscape()
        self.set_agent()
        self.get_match()
        # independent search without socialization
        self.crowd_search_once()
        # upcoming search toward the convergence regarding one child landscape
        # these upcoming search is socialized such that agents will update the initial state
        self.crowd_search_forward()
        for _ in range(self.landscape_num):
            # the upcoming child landscape will have previous child as reference
            # the upcoming child landscape only have one bit change regarding IM
            self.change_working_landscape()
            # one the landscape change, we need to re-calculate the match indicators
            self.get_match()
            # repeating the search toward to a convergence in this child landscape
            self.crowd_search_once()
            self.crowd_search_forward()



if __name__ == '__main__':
    # Test Example (Waiting for reshaping into class above)
    # The test code below works.
    start_time = time.time()
    N = 10
    state_num = 4
    parent_iteration = 100
    landscape_num = 200
    agent_num = 200
    search_iteration = 100
    IM_type = "Traditional Directed"
    k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
    # agent_name = ["Generalist"]
    # IM_type = ["Traditional Directed", "Factor Directed", "Influential Directed"]
    knowledge_num = 20
    





    # # state = 10, E = 12 (=2*6; 4*3; 2*4+ 4*1; 2*2+4*2)
    # K = 0
    # for k in k_list:
    #     for each_agent_type, generalist_num, specialist_num in zip(agent_name, generalist_list, specialist_list):
    #         simulator = Simulator(N=N, state_num=state_num)
    #         A_converged_potential_landscape = []
    #         B_converged_fitness_landscape = []  # for the final converged fitness after search
    #         C_row_match_landscape = []  # for the weighted sum according to IM row
    #         C_column_match_landscape = []  # for the weighted sum according to IM column
    #         D_landscape_IM_list = []  # for the landscape IM
    #         E_knowledge_list_landscape = []  # for the agent knowledge in case we will change the weighted sum algorithm
    #         for parent_loop in range(parent_iteration):
    #             simulator.set_parent_landscape(N=N, state_num=state_num)
    #             for landscape_loop in range(landscape_iteration):
    #                 simulator.set_dynamic_landscape(K=K, k=k, IM_type="Factor Directed", factor_num=0, influential_num=0)
    #                 A_converged_potential_agent = []
    #                 B_converged_fitness_agent = []
    #                 C_row_match_agent = []
    #                 C_column_match_agent = []
    #                 IM = simulator.landscape.IM.tolist()
    #                 D_landscape_IM_list.append(IM)
    #                 E_knowledge_list_agent = []
    #                 for agent_loop in range(agent_iteration):
    #                     simulator.set_agent(name=each_agent_type, lr=0, generalist_num=generalist_num,
    #                                         specialist_num=specialist_num)
    #                     for search_loop in range(search_iteration):
    #                         simulator.agent.cognitive_local_search()
    #                     potential_after_convergence = simulator.landscape.query_potential_performance(
    #                         cog_state=simulator.agent.cog_state, top=1)
    #                     A_converged_potential_agent.append(potential_after_convergence)
    #                     simulator.agent.state = simulator.agent.change_cog_state_to_state(
    #                         cog_state=simulator.agent.cog_state)
    #                     simulator.agent.converge_fitness = simulator.landscape.query_fitness(state=simulator.agent.state)
    #                     B_converged_fitness_agent.append(simulator.agent.converge_fitness)
    #                     # Weighted sum for the match between landscape IM and agent knowledge
    #                     C_row_match_temp = 0
    #                     for row in range(simulator.N):
    #                         if row in simulator.agent.specialist_knowledge_domain:
    #                             C_row_match_temp += sum(IM[row]) * simulator.state_num
    #                         if row in simulator.agent.generalist_knowledge_domain:
    #                             C_row_match_temp += sum(IM[row]) * simulator.state_num * simulator.agent.gs_ratio
    #                     C_column_match_temp = 0
    #                     for column in range(simulator.N):
    #                         if column in simulator.agent.specialist_knowledge_domain:
    #                             C_column_match_temp += sum(IM[:][column]) * simulator.agent.state_num
    #                         if column in simulator.agent.generalist_knowledge_domain:
    #                             C_column_match_temp += sum(
    #                                 IM[:][column]) * simulator.agent.state_num * simulator.agent.gs_ratio
    #
    #                     C_row_match_agent.append(C_row_match_temp)
    #                     C_column_match_agent.append(C_column_match_temp)
    #                     E_knowledge_list_agent.append(
    #                         [simulator.agent.specialist_knowledge_domain, simulator.agent.generalist_knowledge_domain])
    #                 A_converged_potential_landscape.append(A_converged_potential_agent)
    #                 B_converged_fitness_landscape.append(B_converged_fitness_agent)
    #                 C_row_match_landscape.append(C_row_match_agent)
    #                 C_column_match_landscape.append(C_column_match_agent)
    #                 E_knowledge_list_landscape.append(E_knowledge_list_agent)
    #
    #         # Output file name
    #         basic_file_name = simulator.agent.name + '_' + simulator.landscape.IM_type + '_N' + str(simulator.agent.N) + \
    #                           '_K' + str(simulator.landscape.K) + '_k' + str(simulator.landscape.k) + '_E' + str(
    #             simulator.agent.element_num) + \
    #                           '_G' + str(simulator.agent.generalist_num) + '_S' + str(simulator.agent.specialist_num)
    #         A_file_name_potential = "1Potential_" + basic_file_name
    #         B_file_name_convergence = "2Convergence_" + basic_file_name
    #         C_file_name_row_match = "3RowMatch_" + basic_file_name
    #         C_file_name_column_match = "4ColumnMatch_" + basic_file_name
    #         D_file_name_IM_information = "5IM_" + basic_file_name
    #         E_file_name_agent_knowledge = "6Knowledge_" + basic_file_name
    #
    #         with open(A_file_name_potential, 'wb') as out_file:
    #             pickle.dump(A_converged_potential_landscape, out_file)
    #         with open(B_file_name_convergence, 'wb') as out_file:
    #             pickle.dump(B_converged_fitness_landscape, out_file)
    #         with open(C_file_name_row_match, 'wb') as out_file:
    #             pickle.dump(C_row_match_landscape, out_file)
    #         with open(C_file_name_column_match, 'wb') as out_file:
    #             pickle.dump(C_column_match_landscape, out_file)
    #         with open(D_file_name_IM_information, 'wb') as out_file:
    #             pickle.dump(D_landscape_IM_list, out_file)
    #         with open(E_file_name_agent_knowledge, 'wb') as out_file:
    #             pickle.dump(E_knowledge_list_landscape, out_file)
    #
    #         # plt.plot(np.mean(np.mean(np.array(B_converged_fitness_landscape), axis=0), axis=0))
    #         # plt.legend()
    #         # plt.show()
    # end_time = time.time()
    # print("Time used: ", end_time-start_time)



