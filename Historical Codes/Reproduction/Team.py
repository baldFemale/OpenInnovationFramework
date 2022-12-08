# -*- coding: utf-8 -*-
# @Time     : 12/15/2021 20:11
# @Author   : Junyi
# @FileName: Team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import random
from Agent import Agent
from MultiStateInfluentialLandscape import LandScape
import pickle

class Team:
    def __init__(self, members):
        self.members = members  # agent member
        self.agent_num = len(members)
        self.N = self.members[0].N
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

        self.decision_space_dict = {}
        for agent in members:
            for key, value in agent.decision_space_dict.items():
                if key not in self.decision_space_dict.keys():
                    self.decision_space_dict[key] = value
                else:
                    self.decision_space_dict[key] += value
                    self.decision_space_dict[key] = list(set(self.decision_space_dict[key]))

        # the joint state list for team members to optimize (which might be unknown to some agents)
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        # adjustment of the initial joint state
        self.adjust_initial_state()
        # update the freedom space whenever the current state changes
        self.update_freedom_space()

        for agent in members:
            if agent.state_num != self.state_num:
                raise ValueError("Agents should have the same state number")
            if agent.N != self.N:
                raise ValueError("Agents should have the same N")
            if agent.landscape != self.landscape:
                raise ValueError("Agents should have the same landscape")

        # Coordination of the members' state; they start from the *same* team-level state
        for agent in members:
            agent.state = self.state
            agent.update_freedom_space()
            agent.fitness = agent.landscape.query_fitness(self.state)

    def adjust_initial_state(self):
        for i, bit in enumerate(self.state):
            if i in self.knowledge_domain:  # only change the knowledge domain; the unknown domain will stay still
                if i*self.state_num+bit not in self.decision_space:
                    print("Initial adjustment only for *team-level* generalist knowledge domains")
                    self.state[i] = random.choice(self.decision_space_dict[i])
        # there is a special case where G and S have an *overlap*.
        # In such an overlap domain, team level state will not be adjusted.
        # However, for G, this domain originally need to be changed due to outside issue.
        # We assume that G can get an accurate fitness within overlap domains due to the existing of S.

    def update_freedom_space(self):
        state_occupation = [i * self.N + j for i, j in enumerate(self.state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]

    def serial_search(self, search_iteration=100):
        """
        Serial team search based on the joint cognitive fitness given a team state
        Serial means agent will conduct all the search iteration (e.g., 100 steps) before sending the state to another agent
        :param priority: several types pf search order/priority
        :return: updated team state (self.state); updated agent cognitive state (self.member[i].cog_state, masked)
                    updated team fitness (self.fitness); updated agent fitness (self.member[i].fitness)
        """
        # default: follow the list order
        serial_search_result = []
        for search_loop in range(search_iteration):
            temp_fitness = self.members[0].independent_search()
            serial_search_result.append(temp_fitness)
            print("First member search: {0}, {1}".format(self.members[0].state, self.members[0].fitness))
        self.state = self.members[0].state
        self.update_freedom_space()

        self.members[1].state = self.state
        self.members[1].update_freedom_space()

        for search_loop in range(search_iteration):
            temp_fitness = self.members[1].independent_search()
            serial_search_result.append(temp_fitness)
            print("Second member search: {0}, {1}".format(self.members[1].state, self.members[1].fitness))
        self.state = self.members[1].state
        self.update_freedom_space()
        return serial_search_result

    def parallel_search(self, search_iteration=100):
        """
        Parallel search where agents search one step and send the state into another agent.
        Parallel mens each agent only conduct one step.
        :param priority:
        :return: the fitness list
        """
        parallel_search_result = []
        for search_loop in range(search_iteration):
            for agent in self.members:
                agent.state = self.state
                agent.update_freedom_space()
                agent.independent_search()
                temp = agent.fitness
                parallel_search_result.append(temp)
                self.state = agent.state
                print("member {2} search: {0}, {1}".format(agent.state, agent.fitness, agent.name))
        self.update_freedom_space()
        return parallel_search_result

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
        print("Team decision space dict: ", self.decision_space_dict)


if __name__ == '__main__':
    # Test Example
    random.seed(1024)
    np.random.seed(1024)
    N = 10
    k = 0
    K_list = [2, 4, 6, 8, 10]
    state_num = 4
    landscape_iteration = 5
    agent_iteration = 200
    search_iteration = 100
    generalist_list = [6, 0, 4, 2]
    specialist_list = [0, 3, 1, 2]

    for K in K_list:
        fitness_landscape = []
        for landscape_loop in range(landscape_iteration):
            landscape = LandScape(N=N, state_num=state_num)
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
