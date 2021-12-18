# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style

class Simulator:

    def __init__(self, landscape=None, agent=None):
        self.landscape = landscape
        self.agent = agent
        self.agent_num = len(agent)
        self.agent_iteration = None  # repeat the agent or team search
        self.landscape_iteration = None
        self.period = None  # the max iteration of agent search toward convergence
        self.fitness_list = []  # the returned performance path

    def type(self, agent_iteration=2, landscape_iteration=2, period=100):
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



    def run(self, k, K, ):
        for each_landscape in range(self.landscape_iteration):
            fitness_L = []
            for each_agent in range(self.agent_iteration):
                fitness_A = []
                print("Current landscape iteration: {0}; Agent iteration: {1}".format(self.landscape_iteration, self.agent_iteration))


if __name__ == '__main__':
    # Test Example
    from MultiStateInfluentialLandscape import LandScape
    from Agent import Agent

    N = 6
    state_num = 4
    landscape_iteration = 2
    agent_iteration = 2
    k_list = [2]

    landscape = LandScape(N=N,state_num=state_num)
    generalist = Agent(N=N, landscape=landscape)
    generalist.type(name="generalist", generalist_num=4)
    generalist.describe()
    specialist = Agent(N=N, landscape=landscape)
    T_shape = Agent(N=N, landscape=landscape)
    T_shape.type(name="T shape", generalist_num=2, specialist_num=2)
    simulator = Simulator(landscape=landscape, agent=[generalist,specialist,T_shape])


