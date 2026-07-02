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
        self.N = N
        self.agent_num = agent_num
        self.agents = []
        # Visibility sharing mode:
        # - "full": agents disclose their whole solution string.
        # - "partial": agents disclose only the solution components in their knowledge domains.
        # The default is set to "full" to operationalize visibility as whole-solution visibility.
        self.share_mode = "full"

        # self.lr = 1  # theoretically overlap with share_prob
        for _ in range(agent_num):
            if label == "G":
                agent = Generalist(N=N, landscape=landscape, state_num=state_num,
                                   generalist_expertise=generalist_expertise)
                self.agents.append(agent)
            elif label == "S":
                agent = Specialist(N=N, landscape=landscape, state_num=state_num,
                                   specialist_expertise=specialist_expertise)
                self.agents.append(agent)
        self.solution_pool = []

    def search(self):
        for agent in self.agents:
            agent.search()

    def set_visibility_status(self, visibility_prob: float):
        """
        Fix solver-level visibility status for the whole experiment.

        visibility_prob is interpreted as the proportion of solvers whose
        solutions are structurally visible. Once assigned, visibility_status
        does not change across visibility periods unless this method is called
        again.
        """
        if visibility_prob < 0 or visibility_prob > 1:
            raise ValueError("visibility_prob must be between 0 and 1.")

        visible_num = int(round(visibility_prob * self.agent_num))
        visible_indices = np.random.choice(
            range(self.agent_num), size=visible_num, replace=False
        ).tolist()
        visible_indices = set(visible_indices)

        for index, agent in enumerate(self.agents):
            agent.visibility_status = index in visible_indices

    def get_visible_pool(self, visible_mode: str = None):
        """
        Construct the visible solution pool.

        Parameters
        ----------
        visible_mode : str, optional
            - "full": share the sender's whole solution string.
            - "partial": share only the sender's known domains and corresponding partial solution.

        Notes
        -----
        The solution pool keeps the same internal format for both modes:
            [domains, solution]
        where domains identifies the indices to be copied and solution stores the corresponding bits.
        Under full sharing, domains = [0, 1, ..., N-1] and solution = full state.
        """
        if visible_mode is None:
            visible_mode = self.visible_mode

        if visible_mode not in ["full", "partial"]:
            raise ValueError("visible_mode must be either 'full' or 'partial'.")

        self.solution_pool = []  # reset the solution pool
        for agent in self.agents:
            if agent.visibility_status:
                if visible_mode == "full":
                    domains = list(range(self.N))
                    solution = agent.state.copy()
                else:
                    domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
                    solution = [agent.state[index] for index in domains]
                self.solution_pool.append([domains, solution])
        np.random.shuffle(self.solution_pool)  # shuffle the order; randomly imitate

    def learn_from_visible_pool(self):
        # remove the lr parameter; all agents will learn if shared
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

    def calculate_pairwise_solution_distance(self):
        """Average pairwise normalized Hamming distance across complete solutions."""
        states = [agent.state for agent in self.agents]
        if len(states) <= 1:
            return 0

        distance_list = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                distance = (
                    sum(
                        1
                        for bit_i, bit_j in zip(states[i], states[j])
                        if bit_i != bit_j
                    )
                    / self.N
                )
                distance_list.append(distance)

        return np.mean(distance_list)

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        opinions = [agent.public_evaluate(cur_state=cur_state,
                                   next_state=next_state) for agent in self.agents]
        true_count = sum(1 for item in opinions if item)
        return true_count > self.agent_num / 2

    def private_evaluate(self, cur_cog_state: list, next_cog_state: list) -> bool:
        opinions = [agent.private_evaluate(cur_cog_state=cur_cog_state,
                                   next_cog_state=next_cog_state) for agent in self.agents]
        true_count = sum(1 for item in opinions)
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
