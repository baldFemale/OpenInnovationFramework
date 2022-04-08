# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import time
from Landscape import Landscape
from Socialized_Agent import Agent
import numpy as np


class Simulator:

    def __init__(self, N=10, state_num=4, agent_num=500, search_iteration=100, IM_type=None,
                 K=0, k=0, gs_proportion=0.5, knowledge_num=20,
                 exposure_type="Self-interested", openness=None, frequency=None,
                 quality=1.0, S_exposed_to_S=None, G_exposed_to_G=None):
        self.N = N
        self.state_num = state_num
        # Landscape
        self.landscapes = None
        self.whole_state_pool = []
        self.G_state_pool = []
        self.S_state_pool = []
        self.whole_state_pool_rank = []  # only use the list instead of dict
        self.G_state_pool_rank = []  # and we keep the order consistent with state pool list
        self.S_state_pool_rank = []
        self.IM_type = IM_type
        self.K = K
        self.k = k
        # Agent crowd
        self.agents = []  # will be stable; only the initial state and search will change
        self.agent_num = agent_num
        self.knowledge_num = knowledge_num
        self.gs_proportion = gs_proportion
        self.generalist_agent_num = int(self.agent_num * gs_proportion)
        self.specialist_agent_num = int(self.agent_num * (1-gs_proportion))
        # Cognitive Search
        self.search_iteration = search_iteration

        # Transparency parameters
        # Frequency (receiver side: how frequent to copy) & Openness (sender side: how frequent to disclose)
        # This indicator can be furth divided into G_openness and S_openness
        self.openness = openness
        self.frequency = frequency
        # Quality -> (how much information/state length can be copied)
        self.quality = quality
        # Direction
        self.G_exposed_to_G = G_exposed_to_G
        self.G_exposed_to_S = 1 - G_exposed_to_G
        self.S_exposed_to_S = S_exposed_to_S
        self.S_exposed_to_G = 1 - S_exposed_to_S
        # exposure type-> how the agent rank the state pool (self-interested or overall rank)
        self.exposure_type = exposure_type
        valid_exposure_type = ["Self-interested", "Overall-ranking", "Random"]
        # valid_transparency_direction = ['G', "S", "A", "GS", "SG", "Inverse"]
        if self.exposure_type not in valid_exposure_type:
            raise ValueError("Only support: ", valid_exposure_type)
        if (self.openness < 0) or (self.openness > 1):
            raise ValueError("Openness should be between 0-1")
        # if self.transparency_direction not in valid_transparency_direction:
        #     raise ValueError("Only support: ", valid_transparency_direction)

        # Outcome Variables
        self.converged_fitness_landscape = []
        self.potential_after_convergence_landscape = []
        self.unique_fitness_landscape = 0  # count the number of unique fitness in the top 10%

    def set_landscape(self):
        self.landscape = Landscape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=self.IM_type, K=self.K, k=self.k)
        self.landscape.initialize()

    def set_agent(self):
        for _ in range(self.generalist_agent_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            generalist_num = self.knowledge_num // 2
            specialist_num = 0
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Generalist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)
        for _ in range(self.specialist_agent_num):
            # fix the knowledge of each Agent
            # fix the composition of agent crowd (e.g., the proportion of GS)
            specialist_num = self.knowledge_num // 4
            generalist_num = 0
            agent = Agent(N=self.N, landscape=self.landscape, state_num=self.state_num)
            agent.type(name="Specialist", generalist_num=generalist_num, specialist_num=specialist_num)
            self.agents.append(agent)

    def create_state_pools(self):
        """
        There are two factors affecting the state pool generation.
        1) the sharing willingness, captured by self.openness (the likelihood of releasing solutions)
        2) the information quality, captured by self.quality (how much of information to be shared)
        :return: three kinds of state pools
        """
        # clear the pool cache from last time
        self.G_state_pool = []
        self.S_state_pool = []
        self.whole_state_pool = []
        for agent in self.agents:
            flag = np.random.choice((0, 1), p=[1-self.openness, self.openness])
            if (flag == 1) and (self.quality == 1.0):
                if agent.name == "Generalist":
                    self.G_state_pool.append(agent.state)
                elif agent.name == "Specialist":
                    self.S_state_pool.append(agent.state)
            elif (flag == 1) and (self.quality < 1):
                biased_state = list(agent.state)
                inconsistent_len = self.N - int(self.quality * self.N)
                inconsistent_index = np.random.choice(range(self.N), inconsistent_len, replace=False)
                for index in inconsistent_index:
                    biased_state[index] = str(np.random.choice(range(self.state_num)))
                if agent.name == "Generalist":
                    self.G_state_pool.append(biased_state)
                elif agent.name == "Specialist":
                    self.S_state_pool.append(biased_state)
            elif flag == 0:
                continue
            else:
                raise ValueError("Unsupported cases")
        # remove the repeated agent
        self.G_state_pool = ["".join(each) for each in self.G_state_pool]
        self.G_state_pool = list(set(self.G_state_pool))
        self.G_state_pool = [list(each) for each in self.G_state_pool]

        self.S_state_pool = ["".join(each) for each in self.S_state_pool]
        self.S_state_pool = list(set(self.S_state_pool))
        self.S_state_pool = [list(each) for each in self.S_state_pool]

        self.whole_state_pool = self.G_state_pool + self.S_state_pool
        self.whole_state_pool = ["".join(each) for each in self.whole_state_pool]
        self.whole_state_pool = list(set(self.whole_state_pool))
        self.whole_state_pool = [list(each) for each in self.whole_state_pool]

    def create_overall_rank(self, which="A"):
        # in this function, we use local temp to update the pool rank.
        # in last function of state pool generation, we clear pool each time we re-create pool.
        # both work to avoid the pool cache and incorrect accumulation
        overall_pool_rank = {}
        temp = []
        # clear the cache
        self.whole_state_pool_rank = []
        self.S_state_pool_rank = []
        self.G_state_pool_rank = []

        if (which == "All") or (which == "A"):
            for state in self.whole_state_pool:
                overall_pool_rank["".join(state)] = 0
        elif which == 'G':
            for state in self.G_state_pool:
                overall_pool_rank["".join(state)] = 0
        elif which == 'S':
            for state in self.S_state_pool:
                overall_pool_rank["".join(state)] = 0
        else:
            raise ValueError("Unsupported which type: ", which)

        for agent in self.agents:
            if (which == "All") or (which == "A"):
                agent.state_pool_all = self.whole_state_pool
                personal_pool_rank = agent.vote_for_state_pool(which="A")
                agent.state_pool_all = []  # clear the memory after voting
            elif which == 'G':
                agent.state_pool_G = self.G_state_pool
                personal_pool_rank = agent.vote_for_state_pool(which="G")
                agent.state_pool_G = []
            elif which == 'S':
                agent.state_pool_S = self.S_state_pool
                personal_pool_rank = agent.vote_for_state_pool(which="S")
                agent.state_pool_S = []
            else:
                raise ValueError("Unsupported which type: ", which)
            for key, value in personal_pool_rank.items():
                overall_pool_rank[key] += value
        if (which == "All") or (which == "A"):
            for state in self.whole_state_pool:
                temp.append(overall_pool_rank["".join(state)])
        elif which == "G":
            for state in self.G_state_pool:
                temp.append(overall_pool_rank["".join(state)])
        elif which == "S":
            for state in self.S_state_pool:
                temp.append(overall_pool_rank["".join(state)])
        temp = [i/sum(temp) for i in temp]
        if (which == "All") or (which == "A"):
            self.whole_state_pool_rank = temp
        elif which == "G":
            self.G_state_pool_rank = temp
            if len(self.G_state_pool_rank) != len(self.G_state_pool):
                raise ValueError("rank length: {0}, pool length: {1}".format(len(self.G_state_pool_rank), len(self.G_state_pool)))
        elif which == "S":
            self.S_state_pool_rank = temp
            if len(self.S_state_pool_rank) != len(self.S_state_pool):
                raise ValueError("rank length: {0}, pool length: {1}".format(len(self.S_state_pool_rank), len(self.S_state_pool)))

    def change_initial_state(self):
        """
        After we create the state pool and its rank, agents can update their initial state
        :param exposure_type: only three valid exposure types
        :return:all agents update their initial state
        """
        # Prepare the rank information for each kind of exposure
        if self.exposure_type == "Overall-ranking":
            if (not self.S_exposed_to_S) and (not self.G_exposed_to_G):
                self.create_overall_rank(which="A")
                for agent in self.agents:
                    agent.overall_state_pool_rank = self.whole_state_pool_rank
            else:
                self.create_overall_rank(which="G")
                self.create_overall_rank(which="S")
                for agent in self.agents:
                    agent.state_pool_G = self.G_state_pool
                    agent.state_pool_S = self.S_state_pool
                    agent.overall_state_pool_rank_G = self.G_state_pool_rank
                    agent.overall_state_pool_rank_S = self.S_state_pool_rank
        elif self.exposure_type == "Self-interested":
            # update the state_pool after last search
            for agent in self.agents:
                agent.state_pool_G = self.G_state_pool
                agent.state_pool_S = self.S_state_pool
                agent.state_pool_all = self.whole_state_pool
        elif self.exposure_type == "Random":
            for agent in self.agents:
                agent.state_pool_all = self.whole_state_pool
        else:
            raise ValueError("Unsupported exposure_type {0}".format(self.exposure_type))
        success_count = 0
        for agent in self.agents:
            success_count += agent.update_state_from_exposure(exposure_type=self.exposure_type, G_exposed_to_G=self.G_exposed_to_G,
                                                              S_exposed_to_S=self.S_exposed_to_S)
        # print("success_count: ", success_count)

    def tail_recursion(self):
        """
        Finally update the simulator's pool rank information
        This doesn't impact the main body, just for bug check.
        :return:
        """
        if self.exposure_type == "Overall-ranking":
            if (not self.S_exposed_to_S) and (not self.G_exposed_to_G):
                self.create_overall_rank(which="A")
                for agent in self.agents:
                    agent.overall_state_pool_rank = self.whole_state_pool_rank
            else:
                self.create_overall_rank(which="G")
                self.create_overall_rank(which="S")
                for agent in self.agents:
                    agent.state_pool_G = self.G_state_pool
                    agent.state_pool_S = self.S_state_pool
                    agent.overall_state_pool_rank_G = self.G_state_pool_rank
                    agent.overall_state_pool_rank_S = self.S_state_pool_rank
        elif self.exposure_type == "Self-interested":
            # update the state_pool after last search
            for agent in self.agents:
                agent.state_pool_G = self.G_state_pool
                agent.state_pool_S = self.S_state_pool
                agent.state_pool_all = self.whole_state_pool
        elif self.exposure_type == "Random":
            for agent in self.agents:
                agent.state_pool_all = self.whole_state_pool

    def process(self, socialization_freq=1):
        # first search is independent
        self.set_landscape()
        self.set_agent()
        agent0 = self.agents[0]
        # No socialization
        if (self.openness == 0) or (self.quality == 0) or (self.frequency == 0) or \
                ((self.G_exposed_to_G == 0) and (self.S_exposed_to_S == 0)):
            print("Independent Search")
            for agent in self.agents:
                for _ in range(self.search_iteration):
                    agent.cognitive_local_search()
        else:
            for agent in self.agents:
                # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
                agent.cognitive_local_search()
            # print(agent0.cog_state, agent0.cog_fitness, agent0.state)
            # socialized search
            self.create_state_pools()  # <- pool generation, agent's pool will be reset to none, cutting off the link and avoiding wrong pointer
            # print(simulator.whole_state_pool, simulator.G_state_pool, simulator.S_state_pool)
            for i in range(self.search_iteration):
                print(agent0.cog_state, agent0.cog_fitness, agent0.state)
                if i % socialization_freq == 0:
                    self.change_initial_state()  # <- rank generation
                for agent in self.agents:
                    # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
                    agent.cognitive_local_search()
                self.create_state_pools()  # <- pool generation  -> there is a mis-match between rank and pool
            self.tail_recursion()  # This is for information consistency in the Simulator class. For bug control.
            # without tail recursion, the last loop would not update the rank, but its state pool is updated.
            # Thus, the length of state pool and pool rank is unequal. It would be confusing for code review and bug check.
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
    N = 6
    state_num = 4
    K = 2
    k = 0
    IM_type = "Traditional Directed"
    openness = 0.5
    quality = 0.5
    S_exposed_to_S = 0
    G_exposed_to_G = 0.5
    agent_num = 500
    search_iteration = 20
    knowledge_num = 12
    # exposure_type = "Overall-ranking"
    exposure_type = "Self-interested"
    # exposure_type = "Random"
    # if S_exposed_to_S and G_exposed_to_G are None, then it refers to whole state pool,
    # could be either self-interested rank or overall rank on the whole state pool
    simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration, IM_type=IM_type,
                 K=K, k=k, gs_proportion=0.5, knowledge_num=knowledge_num,
                 exposure_type=exposure_type, openness=openness, quality=quality,
                          S_exposed_to_S=S_exposed_to_S, G_exposed_to_G=G_exposed_to_G)
    simulator.process(socialization_freq=1)
    print("END")

