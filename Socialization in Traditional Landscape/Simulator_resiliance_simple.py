# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import time
from Landscape import Landscape
from Socialized_Agent import Agent


class Simulator:

    def __init__(self, N=10, state_num=4, agent_num=500, search_iteration=100,
                 landscape_search_iteration=100, IM_type=None, K=0, k=0, gs_proportion=0.5, knowledge_num=20,
                 exposure_type="Random"):
        self.N = N
        self.state_num = state_num

        # Landscape
        self.landscapes = None
        self.landscape_num = landscape_num
        self.state_pool = []
        self.state_pool_rank = {}
        self.IM_dynamics = [None]
        self.IM_type = IM_type
        self.K = K
        self.k = k
        self.IM_change_bit = 1

        # Agent crowd
        self.agents = []  # will be stable; only the initial state and search will change
        self.agent_num = agent_num
        self.knowledge_num = knowledge_num
        self.exposure_type = exposure_type
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

    def set_landscape(self):
        self.landscape = Landscape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=self.IM_type, K=self.K, k=self.k)
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
        for _ in range(self.generalist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            generalist_num = int(self.knowledge_num/2)
            specialist_num = 0
            print("generalist_num: ", generalist_num)
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Generalist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)
        for _ in range(self.specialist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            specialist_num = int(self.knowledge_num/4)
            generalist_num = 0
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Specialist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)

    def crowd_search_once(self):
        """
        Only one cognitive search loop, so we only change the cog_state, cog_fitness, and the potential_fitness
        The converged_fitness is not updated.
        :return: all the agents in the crowd search once.
        """
        for agent in self.agents:
            agent.cognitive_local_search()
            agent.potential_fitness = agent.landscape.query_potential_performance(cog_state=agent.cog_state, top=1)

    def change_initial_state(self, exposure_type="Random"):
        """
        After we create the state pool and its rank, agents can update their initial state
        :param exposure_type: only three valid exposure types
        :return:all agents update their initial state
        """
        valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
        if exposure_type not in valid_exposure_type:
            raise ValueError("Only support: ", valid_exposure_type)
        if exposure_type == "Overall-ranking":
            # only the overall rank need to shape the pool and assign it into crowd
            self.creat_state_pool_and_rank()  # get the pool rank
            for agent in self.agents:
                agent.assigned_state_pool_rank = self.state_pool_rank
        for agent in self.agents:
            agent.update_state_from_exposure(exposure_type=exposure_type)

    def crowd_search_forward(self):
        """
        Difference: agents need to form the pool and update the initial state
        Corresponding to the child landscape level, where we can get the aggregated outcome variables.
        :return: accumulative outcome variables
        """
        for _ in range(self.landscape_search_iteration):
            self.creat_state_pool_and_rank()
            self.change_initial_state(exposure_type=self.exposure_type)
            self.crowd_search_once()
            self.potential_after_convergence_landscape.append(self.potential_after_convergence_agent)
            self.converged_fitness_landscape.append(self.converged_fitness_agent)

    def creat_state_pool_and_rank(self):
        """
        after each crowd search, create the state pool and make the agents vote for it.
        :return: the crowd-level state pool and its overall rank
        """
        self.state_pool = ["".join(agent.state) for agent in self.agents]
        self.state_pool = list(set(self.state_pool))
        self.state_pool = [list(each) for each in self.state_pool]
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

    def get_match(self):
        """
        Either the agent or landscape change, match indicator need to be re-measured.
        :return: the match indicator, in child landscape level
        """
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
        self.set_landscape()
        self.set_agent()
        self.get_match()
        self.crowd_search_once()
        for _ in range(self.landscape_num):
            self.get_match()
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
    knowledge_num = 16
    exposure_type = "Random"
    # valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
    simulator = Simulator(N=8, state_num=4, landscape_num=2, agent_num=4, search_iteration=2,
                 landscape_search_iteration=1, IM_type="Traditional Directed", K=2, k=0, gs_proportion=0.5,
                          knowledge_num=knowledge_num, exposure_type=exposure_type)
    # simulator.set_parent_landscape()
    # simulator.set_child_landscape()
    # simulator.set_agent()
    simulator.process()
    print("END")