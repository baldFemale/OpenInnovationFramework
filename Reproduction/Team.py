# -*- coding: utf-8 -*-
# @Time     : 12/15/2021 20:11
# @Author   : Junyi
# @FileName: Team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np


class Team:
    def __init__(self, members):
        self.members = members  # agent member
        self.agent_num = len(members)
        self.N = self.members[0].N
        self.state_num = self.members[0].state_num
        self.landscape = self.members[0].landscape
        self.decision_space = []
        self.freedom_space = []
        for agent in members:
            self.decision_space += agent.decision_space
        self.decision_space = list(set(self.decision_space))
        for agent in members:
            self.freedom_space += agent.freedom_space
        # the joint state list for team members to optimize (which might be unknown to some agents)
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.first_time = True
        self.priority = None

        for i in range(self.agent_num):
            if self.members[i].state_num != self.state_num:
                raise ValueError("Agents should have the same state number")
            if self.members[i].N != self.N:
                raise ValueError("Agents should have the same N")
            if self.members[i].landscape != self.landscape:
                raise ValueError("Agents should have the same landscape")

        # Coordination of the members' state
        for i in range(self.agent_num):
            self.members[i].state = self.state
            self.members[i].fitness = self.members[i].landscape.query_fitness(self.state)

    def initialize(self):
        for i in range(self.agent_num):
            if self.members[i].name in ["Generalist", "T shape"]:
                # need to adjust the initialization
                pass

    def set_landscape(self, K=0, k=0, IM_type=None, factor_num=0, influential_num=0):
        self.landscape = LandScape(N=self.N, state_num=self.state_num)
        self.landscape.type(IM_type=IM_type, K=K, k=k, factor_num=factor_num,
                            influential_num=influential_num)
        self.landscape.initialize()
        self.landscape.describe()

    def set_agent(self, name="None", lr=0, generalist_num=0, specialist_num=0):
        self.agent = Agent(N=self.N, lr=0, landscape=self.landscape, state_num=self.state_num)
        self.agent.type(name=name, generalist_num=generalist_num, specialist_num=specialist_num)
        self.agent.describe()

    def cog_fitness_cache(self):
        pass

    def serial_search(self, priority=None):
        """
        Serial team search based on the joint cognitive fitness given a team state
        Serial means agent will conduct all the search iteration (e.g., 100 steps) before sending the state to another agent
        :param priority: several types pf search order/priority
        :return: updated team state (self.state); updated agent cognitive state (self.member[i].cog_state, masked)
                    updated team fitness (self.fitness); updated agent fitness (self.member[i].fitness)
        """
        # Defaults to the given order in member list
        valid_priority = ["Generalist first", "Specialist first", "T shape first", None]
        if priority not in valid_priority:
            raise ValueError("Unsupported priority type, only support {0}".format(valid_priority))

    def parallel_search(self, priority=None):
        """
        Parallel search where agents search one step and send the state into another agent.
        Parallel mens each agent only conduct one step.
        :param priority:
        :return:
        """
        valid_priority = ["Generalist first", "Specialist first", "T shape first", None]
        if priority not in valid_priority:
            raise ValueError("Unsupported priority type, only support {0}".format(valid_priority))

    def describe(self):
        self.members[0].landscape.describe()
        print("-------------------Team Member Information-------------------")
        for i in range(self.agent_num):
            print("------------{0} out of {1} members-----------".format(i+1, self.agent_num))
            self.members[i].describe()


if __name__ == '__main__':
    # Test Example
    from Agent import Agent
    from MultiStateInfluentialLandscape import LandScape
    # initialize the landscape
    landscape = LandScape(N=10, state_num=4)
    landscape.type(IM_type="Traditional Directed", K=4)
    landscape.initialize()

    # initialize the agent members (total element number is 12)
    agent_s = Agent(N=10, lr=0, landscape=landscape, state_num=4, team_flag=True)
    agent_s.type(name="Specialist", generalist_num=0, specialist_num=3,)

    agent_g = Agent(N=10, lr=0, landscape=landscape, state_num=4, team_flag=True)
    agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)

    agent_t = Agent(N=10, lr=0, landscape=landscape, state_num=4, team_flag=True)
    agent_t.type(name="T shape", specialist_num=2, generalist_num=2)

    # make up the team
    team_gs = Team(members=[agent_g, agent_s])
    team_gs.describe()
