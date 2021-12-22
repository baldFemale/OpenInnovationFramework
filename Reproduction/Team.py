# -*- coding: utf-8 -*-
# @Time     : 12/15/2021 20:11
# @Author   : Junyi
# @FileName: Team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
from collections import defaultdict
from tools import *


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
        # the joint state list for team members to optimize
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()

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

    def cog_fitness_cache(self):
        pass

    def cog_team_search(self, priority=None):
        """
        Team search based on the joint cognitive fitness given a team state
        :param priority: several types pf search order/priority
        :return: updated team state (self.state); updated agent cognitive state (self.member[i].cog_state, masked)
                    updated team fitness (self.fitness); updated agent fitness (self.member[i].fitness)
        """
        # Defaults to the given order in member list
        valid_priority = ["Generalist first", "Specialist first", "T shape first", None]
        if priority not in valid_priority:
            raise ValueError("Unsupported priority type, only support {0}".format(valid_priority))


    def get_cognitive_state(self):
        """
        Transfer the real state string into a cognitive/ masked state string using "*"
        :return: the cognitive state stored in Agent.cog_state (i.e., stored in each member's memory, rather than the Team class)
        """
        for i in range(self.agent_num):
            for index, bit in enumerate(self.state):
                if index*self.state_num + bit not in self.members[i].decision_space:
                    self.members[i].cog_state.append("*")
                else:
                    self.members[i].cog_state.append(self.state[index])

    def describe(self):
        self.members[0].landscape.describe()
        for i in range(self.agent_num):
            print("-------------------Team Member Information-------------------")
            self.members[i].describe()


if __name__ == '__main__':
    # Test Example
    from Agent import Agent
    from MultiStateInfluentialLandscape import LandScape
    landscape = LandScape(N=10, state_num=4)
    landscape.type(IM_type="Traditional Directed", K=4)
    landscape.initialize()

    agent_s = Agent(N=10, lr=0, landscape=landscape, state_num=4, team_flag=True)
    agent_s.type(name="Specialist", specialist_num=2, generalist_num=0)

    agent_g = Agent(N=10, lr=0, landscape=landscape, state_num=4, team_flag=True)
    agent_g.type(name="Generalist", specialist_num=0, generalist_num=4)

    team_a = Team(members=[agent_s, agent_g])
    team_a.describe()
