# -*- coding: utf-8 -*-
# @Time     : 6/22/2023 20:46
# @Author   : Junyi
# @FileName: Crowd.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent


class Crowd:
    def __init__(self, N: int, agent_num: int, generalist_expertise: int, specialist_expertise: int,
                 landscape: object, state_num: int, ):
        self.agent_num = agent_num
        self.agents = []
        self.N = N
        self.K = K
        self.state_num = state_num
        self.landscape = landscape
        for _ in range(agent_num):
            agent = Agent(N=N, landscape=landscape, state_num=state_num,
                           generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
            self.agents.append(agent)
        self.solutions = []

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.evaluate(cur_state=cur_state,
                                   next_state=next_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2

    def form_connections(self, group_size: int = 7):
        for agent in self.agents:
            agent.connections = np.random.choice(range(self.agent_num), group_size, replace=False)

    def imitate_from_internal_connections(self, lr: float = 0.3) -> None:
        """
        Imperfect imitation: one can only learn from limited number of connections,
        can only learn from single reference with relatively high performance (from one's perception)
        can only partially (i.e., lr) learn from the superficial solution, without awareness of the entire search path,
        and without awareness of the expertise of the imitatee
        :param lr: imitation rate
        """
        for agent in self.agents:
            peers = []
            for index in agent.connections:
                peers.append(self.agents[index])
            peer_solutions = [one.state for one in peers]
            peer_performance = [agent.get_cog_fitness(state=state) for state in peer_solutions]
            reference = peer_solutions[peer_performance.index(max(peer_performance))]
            for i in range(self.N):
                if np.random.uniform(0, 1) < lr:
                    agent.state[i] = reference[i]
            agent.cog_state = agent.state_2_cog_state(state=agent.state)
            agent.cog_fitness = agent.get_cog_fitness(state=agent.state)
            agent.fitness = self.landscape.query_second_fitness(state=agent.state)

    def imitate_from_external_connections(self, lr: float = 0.3, crowd: object = None) -> None:
        """
        Imperfect imitation: one can only learn from limited number of connections,
        can only learn from single reference with relatively high performance (from one's perception)
        can only partially (i.e., lr) learn from the superficial solution, without awareness of the entire search path,
        and without awareness of the expertise of the imitatee
        :param lr: imitation rate
        """
        for agent in self.agents:
            peers = []
            for index in agent.connections:
                peers.append(crowd.agents[index])
            peer_solutions = [one.state for one in peers]
            peer_performance = [agent.get_cog_fitness(state=state) for state in peer_solutions]
            reference = peer_solutions[peer_performance.index(max(peer_performance))]
            for i in range(self.N):
                if np.random.uniform(0, 1) < lr:
                    agent.state[i] = reference[i]
            agent.cog_state = agent.state_2_cog_state(state=agent.state)
            agent.cog_fitness = agent.get_cog_fitness(state=agent.state)
            agent.fitness = self.landscape.query_second_fitness(state=agent.state)


if __name__ == '__main__':
    from Landscape import Landscape
    np.random.seed(100)
    N = 9
    K = 0
    state_num = 4
    agent_num = 50
    landscape = Landscape(N=N, K=K, state_num=state_num)
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0)

    crowd.form_connections(group_size=7)
    for agent in crowd.agents:
        print(agent.connections)
