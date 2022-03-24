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

    def __init__(self, N=10, state_num=4, agent_num=500, search_iteration=100, IM_type=None,
                 K=0, k=0, gs_proportion=0.5, knowledge_num=20,
                 exposure_type="Self-interested", transparency_direction=None):
        self.N = N
        self.state_num = state_num
        # Landscape
        self.landscapes = None
        self.whole_state_pool = []
        self.G_state_pool = []
        self.S_state_pool = []
        self.whole_state_pool_rank = {}
        self.G_state_pool_rank = {}
        self.S_state_pool_rank = {}
        self.IM_type = IM_type
        self.K = K
        self.k = k
        # Agent crowd
        self.agents = []  # will be stable; only the initial state and search will change
        self.agent_num = agent_num
        self.knowledge_num = knowledge_num
        self.gs_proportion = gs_proportion
        self.generalist_num = int(self.agent_num * gs_proportion)
        self.specialist_num = int(self.agent_num * (1-gs_proportion))
        # Cognitive Search
        self.search_iteration = search_iteration
        self.exposure_type = exposure_type
        self.transparency_direction = transparency_direction
        valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
        valid_transparency_direction = ['G', "S", "A"]
        if self.exposure_type  not in valid_exposure_type:
            raise ValueError("Only support: ", valid_exposure_type)
        if self.transparency_direction  not in valid_transparency_direction:
            raise ValueError("Only support: ", valid_transparency_direction)

        # Outcome Variables
        self.converged_fitness_landscape = []
        self.potential_after_convergence_landscape = []
        self.unique_fitness_landscape = 0  # count the number of unique fitness in the top 10%

    def set_landscape(self):
        self.landscape = Landscape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=self.IM_type, K=self.K, k=self.k)
        self.landscape.initialize()

    def set_agent(self):
        for _ in range(self.generalist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            generalist_num = self.knowledge_num // 2
            specialist_num = 0
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Generalist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)
        for _ in range(self.specialist_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            specialist_num = self.knowledge_num // 4
            generalist_num = 0
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Specialist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)

    def create_state_pools(self):
        """
        after each crowd search, create the state pool and make the agents vote for it.
        :return: the crowd-level state pool and its overall rank
        """
        self.whole_state_pool = ["".join(agent.state) for agent in self.agents]
        self.whole_state_pool = list(set(self.whole_state_pool))
        self.whole_state_pool = [list(each) for each in self.whole_state_pool]

        self.G_state_pool = ["".join(agent.state) for agent in self.agents if agent.name == "Generalist"]
        self.G_state_pool = list(set(self.G_state_pool))
        self.G_state_pool = [list(each) for each in self.G_state_pool]

        self.S_state_pool = ["".join(agent.state) for agent in self.agents if agent.name == "Specialist"]
        self.S_state_pool = list(set(self.S_state_pool))
        self.S_state_pool = [list(each) for each in self.S_state_pool]

    def create_whole_pool_rank(self):
        overall_pool_rank = {}
        for state in self.whole_state_pool:
            overall_pool_rank["".join(state)] = 0
        for agent in self.agents:
            agent.state_pool = self.whole_state_pool
            personal_pool_rank = agent.vote_for_state_pool()
            # format: {"10102": 0.85} i.e., state string: cognitive fitness
            for key, value in personal_pool_rank.items():
                overall_pool_rank[key] += value
        self.whole_state_pool = overall_pool_rank

    def create_G_pool_rank(self):
        overall_pool_rank = {}
        for state in self.G_state_pool:
            overall_pool_rank["".join(state)] = 0
        for agent in self.agents:
            agent.state_pool = self.G_state_pool
            personal_pool_rank = agent.vote_for_state_pool()
            # format: {"10102": 0.85} state string: cognitive fitness
            for key, value in personal_pool_rank.items():
                overall_pool_rank[key] += value
        self.G_state_pool_rank = overall_pool_rank

    def create_S_pool_rank(self):
        overall_pool_rank = {}
        for state in self.S_state_pool:
            overall_pool_rank["".join(state)] = 0
        for agent in self.agents:
            agent.state_pool = self.S_state_pool
            personal_pool_rank = agent.vote_for_state_pool()
            # format: {"10102": 0.85} state string: cognitive fitness
            for key, value in personal_pool_rank.items():
                overall_pool_rank[key] += value
        self.S_state_pool_rank = overall_pool_rank

    def change_initial_state(self):
        """
        After we create the state pool and its rank, agents can update their initial state
        :param exposure_type: only three valid exposure types
        :return:all agents update their initial state
        """
        # For overall-ranking exposure, we create the overall rank and assign it to assigned_state_pool_rank
        if self.exposure_type == "Overall-ranking":
            if self.transparency_direction == "A":
                self.create_whole_pool_rank()
                for agent in self.agents:
                    agent.assigned_state_pool_rank = self.whole_state_pool_rank
            elif self.transparency_direction == "G":
                self.create_G_pool_rank()
                for agent in self.agents:
                    agent.assigned_state_pool_rank = self.G_state_pool_rank
            elif self.transparency_direction == "S":
                self.create_S_pool_rank()
                for agent in self.agents:
                    agent.assigned_state_pool_rank = self.S_state_pool_rank
        # For self-interested exposure, we assign the state_pool (no rank)
        for agent in self.agents:
            if self.transparency_direction == "A":
                agent.state_pool = self.whole_state_pool
            elif self.transparency_direction == "G":
                agent.state_pool = self.G_state_pool
            elif self.transparency_direction == "S":
                agent.state_pool = self.S_state_pool
        # update the initial state
        success_count = 0
        for agent in self.agents:
            success_count += agent.update_state_from_exposure(exposure_type=self.exposure_type)
        print("success_count: ", success_count)

    def process(self, socialization_freq=1):
        # first search is independent
        self.set_landscape()
        self.set_agent()
        agent0 = self.agents[0]
        for agent in self.agents:
            # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
            agent.cognitive_local_search()
        print(agent0.cog_state, agent0.cog_fitness, agent0.state)
        # socialized search
        self.create_state_pools()
        for i in range(self.search_iteration):
            print(agent0.cog_state, agent0.cog_fitness, agent0.state)
            if i % socialization_freq == 0:
                self.change_initial_state()
            for agent in self.agents:
                # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
                agent.cognitive_local_search()
            self.create_state_pools()

        # converge and analysis
        for agent in self.agents:
            agent.converged_fitness = self.landscape.query_fitness(state=agent.state)
            agent.potential_fitness = self.landscape.query_potential_performance(cog_state=agent.cog_state, top=1)
            self.converged_fitness_landscape.append(agent.converged_fitness)
            self.potential_after_convergence_landscape.append(agent.potential_fitness)
        unique_fitness_list = list(set(self.converged_fitness_landscape))
        for fitness in unique_fitness_list:
            if self.landscape.fitness_to_rank_dict[fitness] < 0.1 * self.state_num ** self.N:
                self.unique_fitness_landscape += 1


if __name__ == '__main__':
    # Test Example
    N = 8
    state_num = 4
    K = 2
    k = 0
    IM_type = "Traditional Directed"
    agent_num = 200
    search_iteration = 100
    knowledge_num = 16
    exposure_type = "Random"
    transparency_direction = "A"
    simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration, IM_type=IM_type,
                 K=K, k=k, gs_proportion=0.5, knowledge_num=knowledge_num,
                 exposure_type=exposure_type, transparency_direction=transparency_direction)
    simulator.process()
    print("END")