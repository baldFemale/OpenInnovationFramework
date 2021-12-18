# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle


class Simulator:

    def __init__(self, landscape=None, agent=None, agents=None):
        self.landscape = landscape
        self.agent = agent  # for individual search
        self.agents = agents  # for team-level coordination
        if agents:
            self.agent_num = len(agents)
        else:
            self.agent_num = 1
        self.agent_iteration = None  # repeat the agent or team search
        self.landscape_iteration = None
        self.period = None  # the max iteration of agent search toward convergence
        self.fitness_list = []  # the returned performance path
        if (agent != None) and (agents != None):
            raise ValueError("Agent is for individual search; Agents is for team search!")

    def type(self, agent_iteration=2, landscape_iteration=2, period=100):
        """
        Setting the simulator iteration features (loop number)
        """
        self.agent_iteration = agent_iteration
        self.landscape_iteration = landscape_iteration
        self.period = period

    def describe(self):
        print("*********Simulator information********* ")
        if self.agent_num == 1:
            print("Coordination form: Individual Search")
        else:
            print("Coordination form: Team Search with {0} members.".format(self.agent_num))
        print("Max convergence period: ", self.period)
        print("Landscape Iteration: ", self.landscape_iteration)
        print("Agent(s) Iteration: ".format(self.agent_iteration))

    def individual_run(self):
        fitness_landscape = []
        for landscape_loop in range(self.landscape_iteration):
            fitness_agent = []
            for agent_loop in range(self.agent_iteration):
                print("Current landscape iteration: {0}; Agent iteration: {1}".format(self.landscape_iteration, self.agent_iteration))
                for search_loop in range(self.period):
                    print("Search Loop: ", search_loop)
                    temp_fitness = self.agent.independent_search()
                    fitness_agent.append(temp_fitness)
            fitness_landscape.append(fitness_agent)

        file_name = self.agent.name + '_N' + str(self.agent.N) + '_K' + str(self.landscape.K) + '_k' + str(self.landscape.k) + '_E' + str(self.agent.element_num)
        with open(file_name,'w') as out_file:
            pickle.dump(fitness_landscape, out_file)


if __name__ == '__main__':
    # Test Example
    from MultiStateInfluentialLandscape import LandScape
    from Agent import Agent

    N = 10
    state_num = 4
    landscape_iteration = 2
    agent_iteration = 2
    k_list = [2]

    landscape = LandScape(N=N,state_num=state_num)
    landscape.type(IM_type="Random Directed", k=22)
    generalist = Agent(N=N, landscape=landscape)
    generalist.type(name="Generalist", generalist_num=4)
    generalist.describe()
    # specialist = Agent(N=N, landscape=landscape)
    # T_shape = Agent(N=N, landscape=landscape)
    # T_shape.type(name="T shape", generalist_num=2, specialist_num=2)
    simulator = Simulator(landscape=landscape, agent=generalist)
    simulator.type(agent_iteration=2, landscape_iteration=2, period=20)
    simulator.individual_run()


