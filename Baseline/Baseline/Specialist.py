# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape


class Agent:
    def __init__(self, N=None, landscape=None, state_num=4, crowd=None,
                 generalist_expertise=None, specialist_expertise=None):
        """
        :param N: problem dimension
        :param landscape: assigned landscape
        :param state_num: state number for each dimension
        :param generalist_expertise: the amount of G knowledge
        :param specialist_expertise: the amount of S knowledge
        """
        self.landscape = landscape
        self.crowd = crowd
        self.N = N
        self.state_num = state_num
        self.generalist_domain = []
        self.specialist_domain = np.random.choice(range(self.N), specialist_expertise // 4, replace=False).tolist()
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: a list of string
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.get_cog_fitness(state=self.state)
        self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.cog_fitness_across_time, self.fitness_across_time = [], []
        self.cog_cache = {}
        if specialist_expertise and generalist_expertise:
            if generalist_expertise // 2 + specialist_expertise // 4 > self.N:
                raise ValueError("Entire Expertise Exceed N")
        if generalist_expertise and (generalist_expertise % 2 != 0):
            raise ValueError("Problematic G Expertise")
        if specialist_expertise and (specialist_expertise % 4 != 0):
            raise ValueError("Problematic S Expertise")

    def get_cog_fitness(self, state: list) -> float:
        """
        If Full S, it can perceive the real fitness on the deep landscape
        Otherwise, it can only perceive partial fitness
        """
        if len(self.specialist_domain) == self.N:  # iff full S
            cog_fitness = self.landscape.query_second_fitness(state=state)
        else:
            cog_state = self.state_2_cog_state(state=state)
            cog_fitness = self.landscape.query_scoped_second_fitness(cog_state=cog_state, state=state)
        return cog_fitness

    def search(self) -> None:
        next_state = self.state.copy()
        index = np.random.choice(self.generalist_domain + self.specialist_domain)
        # index = np.random.choice(range(self.N))  # if mindset changes; if environmental turbulence arise outside one's knowledge
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.get_cog_fitness(state=next_state)
        if next_cog_fitness > self.cog_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.fitness_across_time.append(self.fitness)
        self.cog_fitness_across_time.append(self.cog_fitness)

    def feedback_search(self, roll_back_ratio: float, roll_forward_ratio: float) -> None:
        next_state = self.state.copy()
        # index = np.random.choice(self.generalist_domain + self.specialist_domain)
        index = np.random.choice(range(self.N))  # if mindset changes; if environmental turbulence arise outside one's knowledge
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.get_cog_fitness(state=next_state)
        feedback = self.crowd.evaluate(cur_state=self.state, next_state=next_state)
        if next_cog_fitness >= self.cog_fitness:  # focal perception is positive
            if feedback:  # peer feedback is also positive
                self.state = next_state
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                self.fitness = self.landscape.query_second_fitness(state=self.state)
            else:  # feedback is negative
                if np.random.uniform(0, 1) < roll_back_ratio:  # 1st conflict: self "+" and peer "-"
                    pass
                    # roll back; follow the peer feedback and refuse; bounded rationality:
                    # 1) biased perception and thus biased feedback from peers;
                    # 2) generate imperfect solutions and thus received imperfect solutions from peers.
                else:
                    self.state = next_state
                    self.cog_state = next_cog_state
                    self.cog_fitness = next_cog_fitness
                    self.fitness = self.landscape.query_second_fitness(state=self.state)
        else:
            if feedback:  # 2nd conflict: self "-" and peer "+"
                if np.random.uniform(0, 1) < roll_forward_ratio:
                # roll forward; follow the positive feedback and accept;
                    self.state = next_state
                    self.cog_state = next_cog_state
                    self.cog_fitness = next_cog_fitness
                    self.fitness = self.landscape.query_second_fitness(state=self.state)
                else:
                    pass
        self.fitness_across_time.append(self.fitness)
        self.cog_fitness_across_time.append(self.cog_fitness)

    def state_2_cog_state(self, state: list) -> list:
        """
        For Full G, it perceives the real fitness in the shallow landscape
        Similarly, for Full S, it also perceives the real fitness in the deep landscape
        :param state: the real state
        :return: the cognitive state is only a state with unknown shelter
        """
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if index in self.specialist_domain:
                pass
            else:
                cog_state[index] = "*"
        return cog_state

    def shared_cog_state_2_cog_state(self, cog_state: list) -> list:
        """
        The shared state include the "*" -> it is the perceived solution from the sharer
        UNKNOWN domain: one cannot perceive and express the solution accurately
        Resonating with the literature on imperfect socialization (e.g., imperfect imitation, imperfect convey, imperfect acquisition)
        :param cog_state: shared cog_state for evaluation
        :return:self-perceived cog_state
        """
        self_cog_state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if index in self.specialist_domain:
                if bit_value == "*":
                    self_cog_state[index] = self.state[index]
                elif bit_value == "A":
                    self_cog_state[index] = np.random.choice(["0", "1"])
                elif bit_value == "B":
                    self_cog_state[index] = np.random.choice(["2", "3"])
                else:
                    pass
            else:
                self_cog_state[index] = "*"
        return self_cog_state

    def evaluate(self, cur_state: list, next_state: list) -> bool:
        cur_cog_fitness = self.shared_cog_state_2_cog_state(cog_state=cur_state)
        next_cog_fitness = self.shared_cog_state_2_cog_state(cog_state=next_state)
        if next_cog_fitness > cur_cog_fitness:
            return True
        else:
            return False

    # def cog_state_2_state(self, cog_state=None):
    #     state = cog_state.copy()
    #     for index, bit_value in enumerate(cog_state):
    #         # if (index not in self.generalist_domain) and (index not in self.specialist_domain):
    #         #     state[index] = str(random.choice(range(self.state_num)))
    #         if index in self.generalist_domain:
    #             if bit_value == "A":
    #                 state[index] = random.choice(["0", "1"])
    #             elif bit_value == "B":
    #                 state[index] = random.choice(["2", "3"])
    #             else:
    #                 raise ValueError("Unsupported state element: ", bit_value)
    #         else:
    #             pass
    #     return state

    def describe(self) -> None:
        print("Agent of G/S Domain: ", self.generalist_domain, self.specialist_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive State: {0}, Cognitive Fitness: {1}".format(self.cog_state, self.cog_fitness))


if __name__ == '__main__':
    # Test Example
    import time
    t0 = time.time()
    np.random.seed(1000)
    search_iteration = 100
    N = 9
    K = 8
    state_num = 4
    generalist_expertise = 0
    specialist_expertise = 36
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.5)

    # landscape.describe()
    agent = Agent(N=N, landscape=landscape, state_num=state_num,
                    generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
    # agent.describe()
    for _ in range(search_iteration):
        agent.search()
    import matplotlib.pyplot as plt
    x = range(len(agent.fitness_across_time))
    plt.plot(x, agent.fitness_across_time, "k-", label="Fitness")
    plt.plot(x, agent.cog_fitness_across_time, "k--", label="Cognitive Fitness")
    plt.title('Performance at N={0}, K={1}, G={2}, S={3}'.format(N, K, generalist_expertise, specialist_expertise))
    plt.xlabel('Iteration', fontweight='bold', fontsize=10)
    plt.ylabel('Performance', fontweight='bold', fontsize=10)
    # plt.xticks(x)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig("T_performance.png", transparent=True, dpi=200)
    plt.show()
    plt.clf()
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

