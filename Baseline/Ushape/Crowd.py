# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from BinaryAgent import BinaryAgent


class Crowd:
    def __init__(self, N: int, agent_num: int, expertise_amount: int, landscape: object):
        self.agent_num = agent_num
        self.agents = []
        for _ in range(agent_num):
            agent = BinaryAgent(N=N, landscape=landscape, expertise_amount=expertise_amount)
            self.agents.append(agent)
        self.solutions = []

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.evaluate(cur_state=cur_state,
                                   next_state=next_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2
