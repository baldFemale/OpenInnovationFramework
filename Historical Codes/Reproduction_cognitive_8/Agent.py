# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np


class Agent:

    def __init__(self, N=10, lr=0, landscape=None, state_num=8, LL_ratio=0.5):
        self.landscape = landscape
        self.N = N
        self.name = None
        self.state_num = state_num

        # store the true state list at the initialization and the search ending
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.state = [str(i) for i in self.state]
        # store the cognitive state list during the search
        self.cog_state = self.state.copy()

        self.L1_knowledge_representation = ["A", "B"]
        self.L2_knowledge_representation = ["a", "b", 'c', "d"]  # [a, b] is for A; [c, d] is for B

        self.L1_num = 0
        self.L2_num = 0
        self.L3_num = 0
        self.L1_domain = []
        self.L2_domain = []
        self.L3_domain = []
        self.knowledge_domain = []  # L1+L2+L3=knowledge domain

        self.LL_ratio = LL_ratio  # generalist know half of the knowledge compared to specialist
        self.lr = lr

        self.decision_space = []  # all manageable elements' index: ['02', '03', '11', '13', '30', '31']
        self.freedom_space = []  # the alternatives for next step random selection: ['02', '13', '31'] given the current state '310*******'
        # decision_space will not change over time,
        # while freedom_space keep changing due to the current state occupation
        self.cog_fitness = 0  # store the cog_fitness value for each step in search
        self.converge_fitness = 0  # in the final search loop, record the *true* fitness compared to the cog_fitness

        self.valid_state_bit = list(range(self.N))
        self.valid_state_bit += ["A", "B", "a", "b", 'c', "d", "*"]  # for cognitive representation

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")

    def type(self, name=None, L1_num=0, L2_num=0, L3_num=0, element_num=0, LL_ratio=0.5):
        """
        Set the role of agent
        :param name: agent name
        :param L1_num: the number of  L1 domains
        :param L2_num: the number of L2 domains
        :param L3_num: the number of L3 domains (i.e., Specialist domains)
        :param element_num: the total element number
        :param LL_ratio: the ratio of knowledge between to level, 2 by default
        :return: define the agent object
        """
        if name:
            self.name = name
        else:
            self.name = str(L1_num) + "_" + str(L2_num) + "_" + str(L3_num)
        self.LL_ratio = LL_ratio
        self.L1_num = L1_num
        self.L2_num = L2_num
        self.L3_num = L3_num

        domain_number = self.L1_num + self.L2_num + self.L3_num
        self.knowledge_domain = np.random.choice(self.N, domain_number, replace=False).tolist()
        self.L1_domain = np.random.choice(self.knowledge_domain, self.L1_num, replace=False).tolist()
        left_knowledge_domain = [each for each in self.knowledge_domain if each not in self.L1_domain]
        self.L2_domain = np.random.choice(left_knowledge_domain, self.L2_num, replace=False).tolist()
        self.L3_domain = [each for each in left_knowledge_domain if each not in self.L2_domain]

        if element_num:
            self.element_num = element_num  # record the total number of element that agents can change
        else:
            self.element_num = int(2 * self.L1_num + 4 * self.L2_num + 8 * self.L3_num)

        self.cog_state = self.change_state_to_cog_state(state=self.state)
        for cur in self.L1_domain:
            self.decision_space += [str(cur) + i for i in self.L1_knowledge_representation]
        for cur in self.L2_domain:
            self.decision_space += [str(cur) + i for i in self.L2_knowledge_representation]
        for cur in self.L3_domain:
            self.decision_space += [str(cur) + str(i) for i in range(self.state_num)]
        self.update_freedom_space()

        # check the freedom space
        if len(self.freedom_space) != 7 * self.L3_num + 3 * self.L2_num + self.L1_num:
            raise ValueError("Freedom space has a mismatch with type {0}".format(name))

    def update_freedom_space(self):
        state_occupation = [str(i) + j for i, j in enumerate(self.cog_state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]

    def randomly_select_one(self):
        """
        Randomly select one element in the freedom space
        :return: selected position and corresponding value for state list/string
        """
        if len(self.decision_space) == 0:
            raise ValueError("Haven't initialize the decision space; Need to run type() function first")
        next_step = random.choice(self.freedom_space)
        cur_i, cur_j = next_step[0], next_step[1]
        return cur_i, cur_j

    def cognitive_local_search(self):
        """
        Local search in the cognitive neighborhood (one-bit stride)
        :return: Cognitive fitness
        """
        next_cog_state = self.cog_state.copy()
        updated_position, updated_value = self.randomly_select_one()
        next_cog_state[int(updated_position)] = updated_value
        next_cog_fitness = self.landscape.query_cog_fitness(next_cog_state)
        current_cog_fitness = self.landscape.query_cog_fitness(self.cog_state)
        # print("Cur: {0}; Next: {1}".format(current_cog_fitness, next_cog_fitness))
        # print("Cur_state: {0}; Next_state: {1}".format(self.cog_state, next_cog_state))
        if next_cog_fitness > current_cog_fitness:
            self.cog_state = next_cog_state
            self.cog_fitness = next_cog_fitness
            self.update_freedom_space()  # whenever state change, freedom space need to be changed
            return next_cog_fitness
        else:
            self.cog_fitness = current_cog_fitness
            return current_cog_fitness

    def change_state_to_cog_state(self, state):
        cog_state = state.copy()
        for index, bit_value in enumerate(state):
            if index in self.L1_domain:
                if bit_value in ["0", "1", "2", "3"]:
                    cog_state[index] = "A"
                elif bit_value in ["4", "5", "6", "7"]:
                    cog_state[index] = "B"
            elif index in self.L2_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "a"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "b"
                elif bit_value in ["4", "5"]:
                    cog_state[index] = "c"
                elif bit_value in ["6", "7"]:
                    cog_state[index] = "d"
                else:
                    raise ValueError("Unsupported bit value: ", bit_value)
        for index, bit_value in enumerate(self.state):
            if index not in self.knowledge_domain:
                cog_state[index] = '*'
        return cog_state

    def change_cog_state_to_state(self, cog_state=None):
        """
        After concergence, we need to mirror the cognitive fitness into true fitness
        :return: randomly get a true state
        """
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if index not in self.knowledge_domain:
                state[index] = str(random.choice(range(self.state_num)))
            if index in self.L1_domain:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1", "2", "3"])
                elif bit_value == "B":
                    state[index] = random.choice(["4", "5", "6", "7"])
            elif index in self.L2_domain:
                if bit_value == "a":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "b":
                    state[index] = random.choice(["2", "3"])
                elif bit_value == "c":
                    state[index] = random.choice(["4", "5"])
                elif bit_value == "d":
                    state[index] = random.choice(["6", "7"])
                else:
                    raise ValueError("Unsupported bit value: ", bit_value)
        return state

    def state_legitimacy_check(self):
        for index, bit_value in enumerate(self.state):
            if bit_value not in self.valid_state_bit:
                raise ValueError("Current state element/bit is not legitimate")

    def learn(self, target_):
        pass

    def describe(self):
        print("*********Agent information********* ")
        print("Agent type: ", self.name)
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state list: ", self.state)
        print("Current cognitive state list: ", self.cog_state)
        print("Current cognitive fitness: ", self.cog_fitness)
        print("Converged fitness: ", self.converge_fitness)
        print("Knowledge/Manageable domain: ", self.knowledge_domain)
        print("L1 knowledge domain: ", self.L1_domain)
        print("L2 knowledge domain: ", self.L2_domain)
        print("L3 knowledge domain: ", self.L3_domain)
        print("Element number: ", self.element_num)
        print("Freedom space: ", self.freedom_space)
        print("Cognitive state occupation: ", [str(i)+str(j) for i, j in enumerate(self.cog_state)])
        print("Decision space: ", self.decision_space)
        print("********************************")


if __name__ == '__main__':
    # Test Example
    from Landscape import Landscape
    landscape = Landscape(N=8, state_num=8)
    landscape.type(IM_type="Traditional Directed", K=2, k=0)
    landscape.initialize()

    agent = Agent(N=8, lr=0, landscape=landscape, state_num=8)
    agent.type(L1_num=4, L2_num=0, L3_num=0)
    agent.describe()
    for _ in range(100):
        agent.cognitive_local_search()
    agent.state = agent.change_cog_state_to_state(cog_state=agent.cog_state)
    agent.describe()
    agent.converge_fitness = agent.landscape.query_fitness(state=agent.state)
    agent.describe()
