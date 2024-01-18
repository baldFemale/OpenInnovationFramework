# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape


class Specialist:
    def __init__(self, N=None, landscape=None, state_num=4, crowd=None, specialist_expertise=None):
        """
        :param N: problem dimension
        :param landscape: assigned landscape
        :param state_num: state number for each dimension
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
        self.cog_fitness = self.get_cog_fitness(cog_state=self.cog_state, state=self.state)
        self.fitness = self.landscape.query_second_fitness(state=self.state)
        self.cog_fitness_across_time, self.fitness_across_time = [self.cog_fitness], [self.fitness]
        self.cog_cache = {}

    def get_cog_fitness(self, state: list, cog_state: list) -> float:
        """
        If Full S, it can perceive the real fitness on the deep landscape
        Otherwise, it can only perceive partial fitness
        """
        if len(self.specialist_domain) == self.N:  # iff full S
            cog_fitness = self.landscape.query_second_fitness(state=cog_state)
        else:
            cog_fitness = self.landscape.query_scoped_second_fitness(cog_state=cog_state, state=state)
        return cog_fitness

    def search(self) -> None:
        next_state = self.state.copy()
        # index = np.random.choice(self.generalist_domain + self.specialist_domain)
        index = np.random.choice(range(self.N))  # if mindset changes; if environmental turbulence arise outside one's knowledge
        free_space = ["0", "1", "2", "3"]
        free_space.remove(next_state[index])
        next_state[index] = np.random.choice(free_space)
        next_cog_state = self.state_2_cog_state(state=next_state)
        next_cog_fitness = self.get_cog_fitness(state=next_state, cog_state=next_cog_state)
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
        next_cog_fitness = self.get_cog_fitness(state=next_state, cog_state=self.cog_state)
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
        :return: the cognitive state for S is only a state with unknown shelter
        """
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if index in self.specialist_domain:
                pass
            else:
                cog_state[index] = "*"
        return cog_state

    def cog_state_2_state(self, cog_state: list) -> list:
        """
        For Full G, it perceives the real fitness in the shallow landscape
        Similarly, for Full S, it also perceives the real fitness in the deep landscape
        :param cog_state:
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

    def full_evaluate(self, cur_state: list, next_state: list) -> bool:
        """
        Use the explicit state information; only utilize the knowledge domain, not mindset
        """
        if (len(cur_state) == 0) or (len(next_state) == 0):
            raise ValueError("Blank State List")
        # Evaluator can only access the full solution
        cur_cog_state = self.state_2_cog_state(state=cur_state)
        next_cog_state = self.state_2_cog_state(state=next_state)
        cur_cog_fitness = self.get_cog_fitness(cog_state=cur_cog_state, state=cur_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=next_state)
        if next_cog_fitness > cur_cog_fitness:
            return True
        else:
            return False

    def partial_evaluate(self, cur_state: list, next_state: list, visible_scope: list) -> bool:
        """
        Use the explicit state information; only utilize the knowledge domain, not mindset
        """
        if (len(cur_state) == 0) or (len(next_state) == 0):
            raise ValueError("Blank State List")
        # Evaluator can only access the solution fragments within sharers' expertise
        evaluated_cur_state = self.state.copy()
        for index in visible_scope:
            evaluated_cur_state[index] = cur_state[index]
        evaluated_next_state = self.state.copy()
        for index in visible_scope:
            evaluated_next_state[index] = next_state[index]
        cur_cog_state = self.state_2_cog_state(state=evaluated_cur_state)
        next_cog_state = self.state_2_cog_state(state=evaluated_next_state)
        cur_cog_fitness = self.get_cog_fitness(cog_state=cur_cog_state, state=cur_state)
        next_cog_fitness = self.get_cog_fitness(cog_state=next_cog_state, state=next_state)
        if next_cog_fitness > cur_cog_fitness:
            return True
        else:
            return False

    def is_local_optima(self, state: list) -> bool:
        """
        This is for joint confusion vs. mutual climb mechanism
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
            if self.full_evaluate(cur_state=state, next_state=neighbor):
                return False
        return True

    def suggest_better_state_from_expertise(self, state: list) -> list:
        """
        This is for joint confusion vs. mutual climb mechanism
        For simplification, we only consider the public evaluation mode
        The ambiguity in expression/acquisition is neglected
        :param state:
        :return:
        """
        suggestions = []
        neighbor_states = []
        for index in self.specialist_domain:  # only from within expertise domains
            for bit in ["0", "1", "2", "3"]:
                new_state = state.copy()
                if bit != state[index]:
                    new_state[index] = bit
                    neighbor_states.append(new_state)
        if len(neighbor_states) == 0:
            return []
        for neighbor in neighbor_states:
            if self.full_evaluate(cur_state=state, next_state=neighbor):
                suggestions.append(neighbor)
        return suggestions

    def suggest_better_state_from_all(self, state: list) -> list:
        """
        This is for joint confusin vs. mutual climb mechanism
        For simplification, we only consider the public evaluation mode
        The ambiguity in expression/acquisition is neglected
        :param state:
        :return:
        """
        suggestions = []
        neighbor_states = []
        for index in range(self.N):
            for bit in ["0", "1", "2", "3"]:
                new_state = state.copy()
                if bit != state[index]:
                    new_state[index] = bit
                    neighbor_states.append(new_state)
        for neighbor in neighbor_states:
            if self.full_evaluate(cur_state=state, next_state=neighbor):
                suggestions.append(neighbor)
        return suggestions

    def describe(self) -> None:
        print("Agent of G/S Domain: ", self.generalist_domain, self.specialist_domain)
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Cognitive State: {0}, Cognitive Fitness: {1}".format(self.cog_state, self.cog_fitness))


if __name__ == '__main__':
    # Test Example
    import time
    from Crowd import Crowd
    t0 = time.time()
    np.random.seed(110)
    search_iteration = 100
    N = 9
    K = 0
    state_num = 4
    generalist_expertise = 0
    specialist_expertise = 12
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    agent = Specialist(N=N, landscape=landscape, state_num=state_num, specialist_expertise=specialist_expertise, crowd=crowd)
    for _ in range(search_iteration):
        agent.feedback_search(roll_back_ratio=0, roll_forward_ratio=0.5)
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

