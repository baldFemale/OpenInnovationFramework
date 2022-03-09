# -*- coding: utf-8 -*-
# @Time     : 12/15/2021 20:11
# @Author   : Junyi
# @FileName: Team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import random
from Agent import Agent
from Landscape import Landscape
import pickle


class Team:
    """For team level, compared to Simulator()"""
    def __init__(self, members):
        self.members = members  # agent member
        self.agent_num = len(members)
        self.N = self.members[0].N
        self.gs_ratio = 0.5
        self.state_num = self.members[0].state_num
        self.landscape = self.members[0].landscape

        self.knowledge_domain = []
        self.generalist_knowledge_domain = []
        self.specialist_knowledge_domain = []
        self.decision_space = []
        self.freedom_space = []

        for agent in members:
            self.knowledge_domain += agent.knowledge_domain
            self.decision_space += agent.decision_space
            self.generalist_knowledge_domain += agent.generalist_knowledge_domain
            self.specialist_knowledge_domain += agent.specialist_knowledge_domain
        self.knowledge_domain = list(set(self.knowledge_domain))
        self.decision_space = list(set(self.decision_space))
        self.generalist_domain = list(set(self.generalist_knowledge_domain))
        self.specialist_domain = list(set(self.specialist_knowledge_domain))

        # the joint state list for team members to optimize (which might be unknown to some agents)
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.state = [str(i) for i in self.state]

        # allocate the true state to agents
        for agent in self.members:
            agent.state = self.state
            agent.cog_state = agent.change_state_to_cog_state(state=agent.state)
        # collect the cognitive state from agent(s)
        self.cog_state = []
        # update the freedom space whenever the current state changes
        for agent in self.members:
            agent.update_freedom_space()

        for agent in members:
            if agent.state_num != self.state_num:
                raise ValueError("Agents should have the same state number")
            if agent.N != self.N:
                raise ValueError("Agents should have the same N")
            if agent.landscape != self.landscape:
                raise ValueError("Agents should have the same landscape")

    def collect_cog_state(self):
        for agent in self.members:
            self.cog_state.append(agent.cog_state)
        self.cog_state = list(set(self.cog_state))

    def serial_search(self, search_iteration=100):
        """
        Agent always trusts the decision from prior agent
        :param search_iteration:
        :return:the cognitive fitness list
        """
        serial_search_cog_fitness = []
        for search_loop in range(search_iteration):
            pass

    def parallel_search(self, search_iteration=100):
        """
        Negotiation between two agents
        :return:the cognitive fitness process
        """
        parallel_search_result = []
        for search_loop in range(search_iteration):
            self.cog_state = None

    def describe(self):
        self.members[0].landscape.describe()
        print("-------------------Team Member Information-------------------")
        for i in range(self.agent_num):
            self.members[i].describe()
        print("-------------------Team Level Information-------------------")
        print("Team state: ", self.state)
        print("Team knowledge: {0}, length of {1}".format(self.knowledge_domain, len(self.knowledge_domain)))
        print("Team generalist domain: ", self.generalist_knowledge_domain)
        print("Team specialist domain: ", self.specialist_knowledge_domain)
        len_ = [len(agent.knowledge_domain) for agent in self.members]
        print("Team knowledge overlap: {0}".format(sum(len_)-len(self.knowledge_domain)))
        print("Team freedom space: ", self.freedom_space)
        print("Team state occupation")
        print("Team decision space: ", self.decision_space)


if __name__ == '__main__':
    # Test Example
    random.seed(1024)
    np.random.seed(1024)
    N = 10
    k = 0
    K_list = [2]
    state_num = 4
    landscape_iteration = 5
    agent_iteration = 200
    search_iteration = 100
    generalist_list = [6, 0, 4, 2]
    specialist_list = [0, 3, 1, 2]

    for K in K_list:
        fitness_landscape = []
        for landscape_loop in range(landscape_iteration):
            landscape = Landscape(N=N, state_num=state_num)
            landscape.type(IM_type="Traditional Directed", K=K)
            landscape.initialize()
            fitness_agent = []
            for agent_loop in range(agent_iteration):
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                team_gs = Team(members=[agent_g, agent_s])
                fitness_search = team_gs.serial_search(search_iteration=search_iteration)
                fitness_agent.append(fitness_search)
                print("Current landscape iteration: {0}; Agent iteration: {1}".format(landscape_loop, agent_loop))
            fitness_landscape.append(fitness_agent)

        file_name = "GS" + '_' + "Traditional Directed" + '_N' + str(N) + \
                    '_K' + str(K) + '_k' + str(k) + '_E12' + "G6_S3"
        with open(file_name, 'wb') as out_file:
            pickle.dump(fitness_landscape, out_file)
