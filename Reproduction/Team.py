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

    def __init__(self, members, ):
        self.members = members  # agent member
        self.decision_space = []
        self.freedom_space = []
        for agent in members:
            self.decision_space += agent.decision_space
        self.decision_space = list(set(self.decision_space))

        self.state = []  # the joint state list for team members to search







if __name__ == '__main__':
    # Test Example
    from Agent import Agent
    from MultiStateInfluentialLandscape import landscape
    landscape = landscape(N=8, K=4, IM_type="random", IM_random_ratio=None, state_num=4)
    agent_s = Agent(N=10, knowledge_num=8, lr=0, landscape=None, state_num=4)
    agent_s.type(name="Specialist", specialist_num=2)
    agent_g = Agent(N=10, knowledge_num=8, lr=0, landscape=None, state_num=4)
    agent_g.type(name="Generalist", specialist_num=0)
    team_a = Team(members=[agent_s, agent_g])