# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Generalist import Generalist
from Specialist import Specialist
import numpy as np


class Crowd:
    def __init__(self, N: int, agent_num: int, generalist_expertise: int, specialist_expertise: int,
                 landscape: object, state_num: int, label: str, share_prob: float):
        self.agent_num = agent_num
        self.agents = []
        self.share_prob = share_prob
        for _ in range(agent_num):
            if label == "G":
                agent = Generalist(N=N, landscape=landscape, state_num=state_num, generalist_expertise=generalist_expertise)
                self.agents.append(agent)
            elif label == "S":
                agent = Specialist(N=N, landscape=landscape, state_num=state_num, specialist_expertise=specialist_expertise)
                self.agents.append(agent)
        self.solutions = []

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.public_evaluate(cur_state=cur_state,
                                   next_state=next_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2

    def private_evaluate(self, cur_cog_state: list, next_cog_state: list) -> bool:
        opinions = [agent.private_evaluate(cur_cog_state=cur_cog_state,
                                   next_cog_state=next_cog_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2

    def get_shared_state_pool(self):
        shared_states = []
        for agent in self.agents:
            if np.random.uniform(0, 1) < self.share_prob:
                shared_states.append(agent.state)
        return shared_states


if __name__ == '__main__':
    # Test why Generalists provide a poor feedback
    from Landscape import Landscape
    N, K, state_num = 9, 3, 4
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                  generalist_expertise=12, specialist_expertise=0, label="G")
    generalist = Generalist(N=N, landscape=landscape, state_num=state_num, crowd=crowd,
                            generalist_expertise=12)
    for _ in range(100):
        generalist.feedback_search(roll_back_ratio=1, roll_forward_ratio=1)
        print(generalist.state, generalist.cog_fitness, generalist.fitness)
    import matplotlib.pyplot as plt
    x = range(len(generalist.fitness_across_time))
    plt.plot(x, generalist.fitness_across_time, "k-", label="Fitness")
    plt.plot(x, generalist.cog_fitness_across_time, "k--", label="Cognitive Fitness")
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    # plt.savefig("T_performance.png", transparent=True, dpi=200)
    plt.show()