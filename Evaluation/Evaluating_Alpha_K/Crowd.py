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
                 landscape: object, state_num: int, label: str):
        self.agent_num = agent_num
        self.agents = []
        self.share_prob = 1
        self.lr = 1
        for _ in range(agent_num):
            if label == "G":
                agent = Generalist(N=N, landscape=landscape, state_num=state_num, generalist_expertise=generalist_expertise)
                self.agents.append(agent)
            elif label == "S":
                agent = Specialist(N=N, landscape=landscape, state_num=state_num, specialist_expertise=specialist_expertise)
                self.agents.append(agent)
        self.solution_pool = []

    def search(self):
        for agent in self.agents:
            agent.search()

    def get_shared_pool(self):
        self.solution_pool = []  # reset the solution pool
        if self.share_prob == 1:
            for agent in self.agents:
                expertise = agent.generalist_domain.copy() + agent.specialist_domain.copy()
                partial_solution = [agent.state[index] for index in expertise]
                self.solution_pool.append([expertise, partial_solution])
        else:
            for agent in self.agents:
                if np.random.uniform(0, 1) < self.share_prob:
                    domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
                    partial_solution = [agent.state[index] for index in domains]
                    self.solution_pool.append([domains, partial_solution])
        np.random.shuffle(self.solution_pool)  # shuffle the order

    def learn_from_shared_pool(self):
        if self.lr < 1:
            for agent in self.agents:
                if np.random.uniform(0, 1) < self.lr:  # some agents are willing to learn
                    for domains, solution in self.solution_pool:
                        learnt_solution = agent.state.copy()
                        for domain, bit in zip(domains, solution):
                            learnt_solution[domain] = bit
                        cog_solution = agent.state_2_cog_state(state=learnt_solution)
                        perception = agent.get_cog_fitness(cog_state=cog_solution, state=learnt_solution)
                        if perception > agent.cog_fitness:
                            agent.state = solution
                            agent.cog_state = cog_solution
                            agent.cog_fitness = perception
                            agent.fitness = agent.landscape.query_second_fitness(state=solution)
                            break
        else:  # agents always are willing to learn
            for agent in self.agents:
                for domains, solution in self.solution_pool:
                    learnt_solution = agent.state.copy()
                    for domain, bit in zip(domains, solution):
                        learnt_solution[domain] = bit
                    cog_solution = agent.state_2_cog_state(state=learnt_solution)
                    perception = agent.get_cog_fitness(cog_state=cog_solution, state=learnt_solution)
                    if perception > agent.cog_fitness:
                        agent.state = learnt_solution
                        agent.cog_state = cog_solution
                        agent.cog_fitness = perception
                        agent.fitness = agent.landscape.query_second_fitness(state=learnt_solution)
                        break

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.partial_evaluate(cur_state=cur_state, next_state=next_state,
                                   visible_scope=agent.generalist_domain+agent.specialist_domain) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2


if __name__ == '__main__':
    # Test why Generalists provide a poor feedback
    from Landscape import Landscape
    N, K, state_num = 9, 3, 4
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                  generalist_expertise=12, specialist_expertise=0, label="G")
    generalist = Generalist(N=N, landscape=landscape, state_num=state_num, crowd=crowd,
                            generalist_expertise=12)
    fitness_across_time, cog_fitness_across_time = [], []
    for _ in range(100):
        generalist.feedback_search(roll_back_ratio=1, roll_forward_ratio=1)
        print(generalist.state, generalist.cog_fitness, generalist.fitness)
        fitness_across_time.append(generalist.fitness)
        cog_fitness_across_time.append(generalist.cog_fitness)
    import matplotlib.pyplot as plt
    x = range(len(fitness_across_time))
    plt.plot(x, fitness_across_time, "k-", label="Fitness")
    plt.plot(x, cog_fitness_across_time, "k--", label="Cognitive Fitness")
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    # plt.savefig("T_performance.png", transparent=True, dpi=200)
    plt.show()