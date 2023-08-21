# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Generalist import Generalist
from Specialist import Specialist


class Crowd:
    def __init__(self, N: int, agent_num: int, generalist_expertise: int, specialist_expertise: int,
                 landscape: object, state_num: int, label: str):
        self.agent_num = agent_num
        self.agents = []
        for _ in range(agent_num):
            if label == "G":
                agent = Generalist(N=N, landscape=landscape, state_num=state_num,
                               generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
                self.agents.append(agent)
            elif label == "S":
                agent = Specialist(N=N, landscape=landscape, state_num=state_num,
                               generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
                self.agents.append(agent)
        self.solutions = []

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.evaluate(cur_state=cur_state,
                                   next_state=next_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2
