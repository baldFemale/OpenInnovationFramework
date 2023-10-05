# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape


class Generalist:
    def __init__(self, N=None, landscape=None, state_num=4, crowd=None, generalist_expertise=None):
        """
        :param N: problem dimension
        :param landscape: assigned landscape
        :param state_num: state number for each dimension
        :param generalist_expertise: the amount of G knowledge
        :param specialist_expertise: the amount of S knowledge
        cog_state of G: only has "A", "B", and "*"
        cog_state of S: only has "0123" and "*"
        """
        self.landscape = landscape
        self.crowd = crowd
        self.N = N
        self.state_num = state_num
        self.generalist_domain = np.random.choice(range(self.N),  generalist_expertise // 2, replace=False).tolist()
        self.specialist_domain = []
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: a list of string
        self.cog_state = self.state_2_cog_state(state=self.state)
        self.cog_fitness = self.get_cog_fitness(cog_state=self.cog_state, state=self.state)
        self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.cog_fitness_across_time, self.fitness_across_time = [self.cog_fitness], [self.fitness]
        self.cog_cache = {}

    def get_cog_fitness(self, cog_state: list, state: list) -> float:
        """
        If Full G, it can perceive the real fitness on the shallow landscape
        Otherwise, it can only perceive partial fitness
        """
        if len(self.generalist_domain) == self.N:  # iff full G
            cog_fitness = self.landscape.query_first_fitness(state=cog_state)  # use "AB"
        else:
            cog_fitness = self.landscape.query_scoped_first_fitness(cog_state=cog_state, state=state)  # use "AB*" and "0123"
        return cog_fitness

    def search(self) -> None:
        next_state = self.state.copy()
        index = np.random.choice(self.generalist_domain + self.specialist_domain)
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=next_state)
        if next_cog_fitness > self.cog_fitness:
            self.state = next_state
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.fitness_across_time.append(self.fitness)
        self.cog_fitness_across_time.append(self.cog_fitness)

    def feedback_search(self, roll_back_ratio: float, roll_forward_ratio: float) -> None:
        next_state = self.state.copy()
        index = np.random.choice(self.generalist_domain + self.specialist_domain)
        # index = np.random.choice(range(self.N))  # if mindset changes; if environmental turbulence arise outside one's knowledge
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=next_state)
        feedback = self.crowd.evaluate(cur_state=self.state, next_state=next_state)

        if next_cog_fitness > self.cog_fitness:  # focal perception is positive
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
        :return: "0213" -> "AB**"
        """
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if index in self.generalist_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                else:
                    cog_state[index] = "B"
            else:
                cog_state[index] = "*"
        return cog_state

    def cog_state_2_state(self, cog_state: list) -> list:
        """
        For Full G, it perceives the real fitness in the shallow landscape
        Similarly, for Full S, it also perceives the real fitness in the deep landscape
        :param state: the real state
        :return: "0213" -> "AB**"
        """
        state = []
        for index, bit_value in enumerate(cog_state):
            if bit_value == "A":
                state.append(np.random.choice(["0", "1"]))
            elif bit_value == "B":
                state.append(np.random.choice(["2", "3"]))
            elif bit_value == "*":
                state.append(np.random.choice(["0", "1", "2", "3"]))
            elif bit_value in ["0", "1", "2", "3"]:
                state.append(bit_value)
            else:
                raise ValueError("Unsupported Bit of {0}".format(bit_value))
        return state

    def public_evaluate(self, cur_state: list, next_state: list) -> bool:
        """
        Use the explicit state information; only utilize the knowledge domain, not mindset
        """
        cur_cog_state = self.state_2_cog_state(state=cur_state)
        next_cog_state = self.state_2_cog_state(state=next_state)
        cur_cog_fitness = self.get_cog_fitness(cog_state=cur_cog_state, state=cur_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=next_state)
        if next_cog_fitness > cur_cog_fitness:
            return True
        else:
            return False

    def private_evaluate(self, cur_cog_state: list, next_cog_state: list) -> bool:
        """
        With additional "*" shelter due to inexplicit expression and acquisition
        """
        aligned_cur_cog_state, aligned_next_cog_state = cur_cog_state.copy(), next_cog_state.copy()
        # aligned as the G cog_state: only has "A", "B", and "*"
        for index in range(self.N):
            if index in self.generalist_domain:
                if aligned_cur_cog_state[index] in ["0", "1"]:
                    aligned_cur_cog_state[index] = "A"
                elif aligned_cur_cog_state[index] in ["2", "3"]:
                    aligned_cur_cog_state[index] = "B"
                elif aligned_cur_cog_state[index] == "*":
                    aligned_cur_cog_state[index] = self.cog_state[index]
                else:
                    pass

                if aligned_next_cog_state[index] in ["0", "1"]:
                    aligned_next_cog_state[index] = "A"
                elif aligned_next_cog_state[index] in ["2", "3"]:
                    aligned_next_cog_state[index] = "B"
                elif aligned_next_cog_state[index] == "*":
                    aligned_next_cog_state[index] = self.cog_state[index]
                else:
                    pass
            else:
                aligned_cur_cog_state[index] = "*"
                aligned_next_cog_state[index] = "*"
        aligned_cur_state = self.cog_state_2_state(cog_state=aligned_cur_cog_state)
        aligned_next_state = self.cog_state_2_state(cog_state=aligned_next_cog_state)
        # the randomness in reverse mapping could produce additional difference; but it is random
        cur_cog_fitness = self.get_cog_fitness(cog_state=aligned_cur_cog_state, state=aligned_cur_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=aligned_next_state)
        if next_cog_fitness > cur_cog_fitness:
            return True
        else:
            return False

    def is_local_optima(self, state: list) -> bool:
        """
        This is for joint confusin vs. mutual climb mechanism
        For simplification, we only consider the public evaluation mode
        The ambiguity in expression/acquisition is neglected
        :param state:
        :return:
        """
        neighbor_states = []
        for index in range(self.N):
            for bit in ["0", "1", "2", "3"]:
                new_state = state.copy()
                if bit != state[index]:
                    new_state[index] = bit
                    neighbor_states.append(new_state)
        for neighbor in neighbor_states:
            if self.public_evaluate(cur_state=state, next_state=neighbor):
                return False
        return True

    def suggest_better_state(self, state: list) -> list:
        """
        This is for joint confusin vs. mutual climb mechanism
        For simplification, we only consider the public evaluation mode
        The ambiguity in expression/acquisition is neglected
        :param state:
        :return:
        """
        neighbor_states = []
        for index in range(self.N):
            for bit in ["0", "1", "2", "3"]:
                new_state = state.copy()
                if bit != state[index]:
                    new_state[index] = bit
                    neighbor_states.append(new_state)
        for neighbor in neighbor_states:
            if self.public_evaluate(cur_state=state, next_state=neighbor):
                return neighbor
        return []

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
    K = 0
    state_num = 4
    generalist_expertise = 12
    specialist_expertise = 0
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.2)

    # landscape.describe()
    agent = Generalist(N=N, landscape=landscape, state_num=state_num, generalist_expertise=generalist_expertise)
    # agent.describe()
    for _ in range(search_iteration):
        agent.search()
        print(agent.cog_state, agent.state, agent.cog_fitness)
    last_state = agent.state
    last_state[-1] = "1"
    print("Local Optimal: ", last_state, agent.is_local_optima(state=last_state))
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

