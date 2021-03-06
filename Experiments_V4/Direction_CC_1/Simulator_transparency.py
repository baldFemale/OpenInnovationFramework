# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
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
        # Pool state is for surface divergence/distance
        self.whole_state_pool = []
        self.G_state_pool = []
        self.S_state_pool = []
        # Surface divergence
        self.S_state_pool_divergence = 0
        self.G_state_pool_divergence = 0
        # Surface potential
        self.whole_state_pool_potential = []  # not used so far; is a redundant variable
        self.G_state_pool_potential = []  # to measured the surface quality, measured when agent share
        self.S_state_pool_potential = []
        # Surface utilization
        self.whole_state_pool_utilization = 0
        self.G_state_pool_utilization = 0  # measured by the (true_fitness / max_potential)
        self.S_state_pool_utilization = 0
        # Pool rank is for exposure likelihood
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
        self.S_exposed_to_S = S_exposed_to_S
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
        self.converged_fitness_rank_landscape = []
        # self.potential_fitness_landscape = []

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
        # clear the pool cache from last time to avoid the accumulation
        self.whole_state_pool = []
        self.G_state_pool = []  # to measure the surface diversity
        self.S_state_pool = []  # and also used during the socialization

        self.whole_state_pool_potential = []
        self.G_state_pool_potential = []  # to measure the surface quality
        self.S_state_pool_potential = []

        for agent in self.agents:
            if (agent.fixed_openness_flag == 1) and (self.quality == 1.0):
                if agent.name == "Generalist":
                    self.G_state_pool.append(agent.state)
                    self.G_state_pool_potential.append(agent.potential_fitness)
                elif agent.name == "Specialist":
                    self.S_state_pool.append(agent.state)
                    self.S_state_pool_potential.append(agent.potential_fitness)
            elif (agent.fixed_openness_flag == 1) and (self.quality < 1):
                biased_state = list(agent.state)
                inconsistent_len = self.N - int(self.quality * self.N)
                inconsistent_index = np.random.choice(range(self.N), inconsistent_len, replace=False)
                for index in inconsistent_index:
                    original_element = agent.state[index]
                    freedom_space = [i for i in range(self.state_num) if str(i) != original_element]
                    biased_state[index] = str(np.random.choice(freedom_space))
                if agent.name == "Generalist":
                    self.G_state_pool.append(biased_state)
                    self.G_state_pool_potential.append(agent.potential_fitness)
                elif agent.name == "Specialist":
                    self.S_state_pool.append(biased_state)
                    self.S_state_pool_potential.append(agent.potential_fitness)
            elif agent.fixed_openness_flag == 0:
                continue
            else:
                raise ValueError("Unsupported cases")
        # before unique, measure the utilization degree; the utilization could include the repeated
        # (to keep the sequence order)

        # print("before_average_len: ", len(ave_fitness_list_G), len(ave_fitness_list_S))
        # print("before_potential_len: ", len(self.G_state_pool_potential), len(self.S_state_pool_potential))
        # remove all the operation, because agent could get to convergence
        # the repeated solutions are more likely to be used; but the repeated number is small
        # self.G_state_pool = ["".join(each) for each in self.G_state_pool]
        # self.G_state_pool = list(set(self.G_state_pool))
        # self.G_state_pool = [list(each) for each in self.G_state_pool]
        #
        # self.S_state_pool = ["".join(each) for each in self.S_state_pool]
        # self.S_state_pool = list(set(self.S_state_pool))
        # self.S_state_pool = [list(each) for each in self.S_state_pool]
        #
        # self.whole_state_pool = self.G_state_pool + self.S_state_pool
        # self.whole_state_pool = ["".join(each) for each in self.whole_state_pool]
        # self.whole_state_pool = list(set(self.whole_state_pool))
        # self.whole_state_pool = [list(each) for each in self.whole_state_pool]

        # the potential is to measure the surface quality, so it could be repeated and cannot remove the repeatation
        # Because different surface could lead to the same potential position
        # self.whole_state_pool_potential = list(set(self.whole_state_pool_potential))
        # self.G_state_pool_potential = list(set(self.G_state_pool_potential))
        # self.S_state_pool_potential = list(set(self.S_state_pool_potential))

        # print("after_state_pool: ", len(self.G_state_pool), len(self.S_state_pool))
        # print("after_potential: ", len(self.G_state_pool_potential), len(self.S_state_pool_potential))

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

    def change_initial_state(self, footprint=False):
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
        if footprint:
            print("success_count: ", success_count)

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

    def process(self, socialization_freq=1, footprint=False):
        # first search is independent
        self.set_landscape()
        self.set_agent()
        agent0 = self.agents[0]
        # No socialization
        if (self.openness == 0) or (self.quality == 0) or (self.frequency == 0):
            print("Independent Search")
            for agent in self.agents:
                for _ in range(self.search_iteration):
                    agent.cognitive_local_search()
        else:
            # fix the exposure pool across search iterations
            # fix the willingness to share
            for agent in self.agents:
                if agent.name == "Generalist":
                    selected_pool_index = np.random.choice((0, 1), p=[self.G_exposed_to_G, 1-self.G_exposed_to_G])  # 0 refers to G pool, while 1 refers to S pool
                elif agent.name == "Specialist":
                    selected_pool_index = np.random.choice((0, 1), p=[1-self.S_exposed_to_S, self.S_exposed_to_S])
                else:
                    raise ValueError("Outlier of agent name: {0}".format(agent.name))
                agent.fixed_state_pool = selected_pool_index
                agent.fixed_openness_flag = np.random.choice((0,1), p=[1-self.openness, self.openness])
            for agent in self.agents:
                # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
                agent.cognitive_local_search()
            # print(agent0.cog_state, agent0.cog_fitness, agent0.state)
            # socialized search
            self.create_state_pools()  # <- pool generation, agent's pool will be reset to none, cutting off the link and avoiding wrong pointer
            # print(simulator.whole_state_pool, simulator.G_state_pool, simulator.S_state_pool)
            for i in range(self.search_iteration):
                if footprint:
                    print(agent0.cog_state, agent0.cog_fitness, agent0.state)
                if i % socialization_freq == 0:
                    self.change_initial_state(footprint=footprint)  # <- rank generation
                for agent in self.agents:
                    # once search, agent will update its cog_state, state, cog_fitness; but not for fitness
                    agent.cognitive_local_search()
                self.create_state_pools()  # <- pool generation  -> there is a mis-match between rank and pool
            self.tail_recursion()  # This is for information consistency in the Simulator class. For bug control.
            # without tail recursion, the last loop would not update the rank, but its state pool is updated.
            # Thus, the length of state pool and pool rank is unequal. It would be confusing for code review and bug check.
        # Only record the surface feature after convergence
        for agent in self.agents:
            agent.converged_fitness = self.landscape.query_fitness(state=agent.state)
            agent.converged_fitness_rank = self.landscape.fitness_to_rank_dict[agent.converged_fitness]
            # agent.potential_fitness = self.landscape.query_potential_fitness(cog_state=agent.cog_state, top=1)
            self.converged_fitness_landscape.append(agent.converged_fitness)
            self.converged_fitness_rank_landscape.append(agent.converged_fitness_rank)
        if self.gs_proportion != 0:  # all the S, len(G_pool) = 0 and cannot be divided.
            ave_fitness_list_G = [self.landscape.query_fitness(state=state) for state in self.G_state_pool]
            self.G_state_pool_utilization = [ave/potential for ave, potential in zip(ave_fitness_list_G, self.G_state_pool_potential)]
            self.G_state_pool_utilization = sum(self.G_state_pool_utilization)/(len(self.G_state_pool) + 1)
        if self.gs_proportion != 1:  # all the G, len(S_pool) = 0 and cannot be divided.
            ave_fitness_list_S = [self.landscape.query_fitness(state=state) for state in self.S_state_pool]
            self.S_state_pool_utilization = [ave/potential for ave, potential in zip(ave_fitness_list_S, self.S_state_pool_potential)]
            self.S_state_pool_utilization = sum(self.S_state_pool_utilization)/(len(self.S_state_pool_utilization) + 1)

        # calculate the pair-wise distance to measure the surface divergence
        if self.gs_proportion != 0:
            self.G_state_pool_divergence = self.pair_wise_distance(state_pool=self.G_state_pool)
        if self.gs_proportion != 1:
            self.S_state_pool_divergence = self.pair_wise_distance(state_pool=self.S_state_pool)

    def pair_wise_distance(self, state_pool=None):
        distance = 0
        for state in state_pool:
            distance += sum([self.count_divergence(state, next_) for next_ in state_pool]) / (len(state_pool) + 1)
        distance = distance / (len(state_pool) + 1)
        return distance

    def count_divergence(self, state_1=None, state_2=None):
        divergence = 0
        for i in range(len(state_1)):
            if state_1[i] != state_2[i]:
                divergence += 1
        return divergence



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
    search_iteration = 50
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
    simulator.process(socialization_freq=1, footprint=False)
    count_GS = 0
    count_GG = 0
    count_SS = 0
    count_SG = 0
    count_open_G, count_open_S = 0, 0
    for agent in simulator.agents:
        if agent.name == "Generalist":
            if agent.fixed_state_pool == 1:
                count_GS += 1
            else:
                count_GG += 1
            if agent.fixed_openness_flag == 1:
                count_open_G += 1
        else:
            if agent.fixed_state_pool == 1:
                count_SS += 1
            else:
                count_SG += 1
            if agent.fixed_openness_flag == 1:
                count_open_S += 1
    print("GG, GS: ", count_GG, count_GS)
    print("SS, SG", count_SS, count_SG)
    print("Openness G, S: ", count_open_G, count_open_S)
    surface_quality_G, surface_quality_S = [], []
    for each_qualities in simulator.surface_quality_G_landscape:
        surface_quality_G.append(np.mean(np.array(each_qualities, dtype=object), axis=0))
    print("surface_quality_G: ", surface_quality_G)
    for each_qualities in simulator.surface_quality_S_landscape:
        surface_quality_S.append(np.mean(np.array(each_qualities, dtype=object), axis=0))
    print("surface_quality_S: ", surface_quality_S)
    print("END")

