# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape

class Agent:

    def __init__(self, N, lr=0, landscape=None, state_num=4, gs_ratio=0.5):
        self.landscape = landscape
        self.N = N
        self.name = "None"
        self.state_num = state_num

        # Agent will have both unmasked and masked state representation system
        # store the true state list at the initialization and the search ending
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.state = [str(i) for i in self.state]
        # store the cognitive state list during the search
        self.cog_state = self.state.copy()

        self.generalist_knowledge_representation = ["A", "B"]  # for state_num=6, use ["A", "B", "C"]
        self.knowledge_domain = []
        self.generalist_num = 0
        self.specialist_num = 0
        self.specialist_knowledge_domain = []
        self.generalist_knowledge_domain = []  # G_domain + S_domain =knowledge_domain
        self.default_elements_in_unknown_domain = []  # record the unknown domain '*'

        self.gs_ratio = gs_ratio  # generalist know half of the knowledge compared to specialist
        self.lr = lr

        self.decision_space = []  # all manageable elements' index: ['02', '03', '11', '13', '30', '31']
        self.freedom_space = []  # the alternatives for next step random selection: ['02', '13', '31'] given the current state '310*******'
        # self.freedom_space_g = [] # for long jump search; [[], []]
        # self.freedom_space_s = []
        # decision_space will not change over time,
        # while freedom_space keep changing due to the current state occupation
        self.cog_fitness = 0  # store the cog_fitness value for each step in search
        self.converge_fitness = 0  # in the final search loop, record the true fitness compared to the cog_fitness
        self.potential = 0  # record the potential achievement; the position advantage to achieve a higher future performance

        self.first_time = True  # flag for the state adjustment of random initialization, as some element may be outside of the agent knowledge

        self.valid_state_bit = list(range(self.N))
        self.valid_state_bit += ["A", "B", "*"]  # for cognitive representation

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")

    def type(self, name=None, specialist_num=0, generalist_num=0, element_num=0, gs_ratio=0.5):
        """
        Allocate one certain type to the agent
        :param name: the agent role
        :param specialist_num: the
        :param gs_ratio: the ratio of knowledge between G and S
        :return: Updating the agent characters
        """
        valid_types = ['Generalist', 'Specialist', 'T shape', "None"]
        if name not in valid_types:
            raise ValueError("Only support 4 types: Generalist, Specialist, T shape, and None.")
        if (name == "Generalist") & (specialist_num != 0):
            raise ValueError("Generalist cannot have specialist number")
        if (name == "Specialist") & (generalist_num != 0):
            raise ValueError("Generalist cannot have specialist number")
        if (name == "T shape") & (generalist_num*specialist_num == 0):
            raise ValueError("T shape mush have both generalist number and specialist number")

        self.name = name
        self.gs_ratio = gs_ratio
        self.generalist_num = generalist_num
        self.specialist_num = specialist_num
        domain_number = self.generalist_num + self.specialist_num
        self.knowledge_domain = np.random.choice(self.N, domain_number, replace=False).tolist()
        self.default_elements_in_unknown_domain = [str(index) + str(value) for index, value in enumerate(self.state)
                                                   if index not in self.knowledge_domain]
        if element_num:
            self.element_num = element_num  # record the total number of element that agents can change

            if element_num % int(self.gs_ratio * self.state_num) != 0:
                raise ValueError("Element number ({0}) cannot be exactly divided by {1}".format(element_num,
                                                                                                int(self.gs_ratio * self.state_num)))
            if element_num < int(self.gs_ratio * self.state_num) * domain_number:
                raise ValueError(
                    "Element number ({0}) is not greater than {1}, which is the minimum for generalist type".format(
                        element_num, int(self.gs_ratio * self.state_num) * domain_number))
            if element_num > self.state_num * domain_number:
                raise ValueError(
                    "Element number ({0}) is not smaller than {1}, which is the maximun for specialist type".format(
                        element_num, self.state_num * domain_number))
            if self.name == "T shape":
                if element_num == int(self.gs_ratio * self.state_num) * domain_number:
                    raise ValueError("T shape without enough elements is the same as generalist")
                elif element_num == self.state_num * domain_number:
                    raise ValueError("T shape with full element occupation is the same as specialist")
        else:
            self.element_num = int(self.state_num * self.generalist_num * self.gs_ratio + self.state_num * self.specialist_num)

        self.specialist_knowledge_domain = np.random.choice(self.knowledge_domain, specialist_num, replace=False).tolist()
        self.generalist_knowledge_domain = [
            cur for cur in self.knowledge_domain if cur not in self.specialist_knowledge_domain
        ]
        self.cog_state = self.change_state_to_cog_state(state=self.state)
        for cur in self.specialist_knowledge_domain:
            self.decision_space += [str(cur) + str(i) for i in range(self.state_num)]
        for cur in self.generalist_knowledge_domain:
            self.decision_space += [str(cur) + i for i in self.generalist_knowledge_representation]
        self.update_freedom_space()

        # check the freedom space
        if len(self.freedom_space) != (self.state_num - 1) * self.specialist_num + self.generalist_num:
            raise ValueError("Freedom space has a mismatch with type {0}".format(name))

    def update_freedom_space(self):
        state_occupation = [str(i) + j for i, j in enumerate(self.cog_state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]

    def randomly_select_one(self):
        """
        Randomly select one element in the freedom space
        Local search for Specialist domains
        :return: selected position and corresponding value for state list/string
        """
        if len(self.decision_space) == 0:
            raise ValueError("Haven't initialize the decision space; Need to run type() function first")
        next_step = random.choice(self.freedom_space)
        cur_i, cur_j = next_step[0], next_step[1]
        return int(cur_i), cur_j

    def cognitive_local_search(self):
        updated_position, updated_value = self.randomly_select_one()
        if updated_position in self.generalist_knowledge_domain:
            next_cog_state = self.cog_state.copy()
            next_cog_state[updated_position] = updated_value
            current_cog_fitness, cur_max_potential_rank = self.landscape.query_cog_fitness(self.cog_state)
            next_cog_fitness, next_max_potential_rank = self.landscape.query_cog_fitness(next_cog_state)
            if next_cog_fitness > current_cog_fitness:
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
                self.potential = next_max_potential_rank
            else:
                self.cog_fitness = current_cog_fitness
                self.potential = cur_max_potential_rank
        elif updated_position in self.specialist_knowledge_domain:
            cur_cog_state_with_default = self.cog_state.copy()
            next_cog_state = self.cog_state.copy()
            next_cog_state[updated_position] = updated_value
            next_cog_state_with_default = next_cog_state.copy()
            for default_mindset in self.default_elements_in_unknown_domain:
                cur_cog_state_with_default[int(default_mindset[0])] = default_mindset[1]
                next_cog_state_with_default[int(default_mindset[0])] = default_mindset[1]
            current_cog_fitness, cur_max_potential_rank = self.landscape.query_cog_fitness(cur_cog_state_with_default)
            next_cog_fitness, next_max_potential_rank = self.landscape.query_cog_fitness(next_cog_state_with_default)
            if next_cog_fitness > current_cog_fitness:
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
                self.potential = next_max_potential_rank
            else:
                self.cog_fitness = current_cog_fitness
                self.potential = cur_max_potential_rank
        else:
            raise ValueError("The picked next step go outside of G/S knowledge domain")

    def change_state_to_cog_state(self, state):
        """
        There are two kinds of unknown elements:
            1) unknown in the width (outside of knowledge domain)-> masked with '*'
            2) unknown in the depth (outside of professional level)-> agents cannot select independently (i.e., masked by freedom space)
                For team level, there could be some kind of mapping, learning, or communication to improve the cognitive accuracy.
        :param state: the intact state list
        :return: masked state list
        """
        cog_state = self.state.copy()
        if self.generalist_num != 0:  # there are some unknown depth (i.e., with G domain)
            for index, bit_value in enumerate(state):
                if index in self.generalist_knowledge_domain:
                    if bit_value in ["0", "1"]:
                        cog_state[index] = "A"
                    elif bit_value in ["2", "3"]:
                        cog_state[index] = "B"
                    else:
                        raise ValueError("Only support for state number = 4")
        if len(self.knowledge_domain) < self.N:  # there are some unknown domains
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
            if index in self.generalist_knowledge_domain:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "B":
                    state[index] = random.choice(["2", "3"])
                else:
                    raise ValueError("Unsupported state element: ", bit_value)
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
        print("Specialist knowledge domain: ", self.specialist_knowledge_domain)
        print("Generalist knowledge domain: ", self.generalist_knowledge_domain)
        print("Element number: ", self.element_num)
        print("Learning rate: ", self.lr)
        print("G/S knowledge ratio: ", self.gs_ratio)
        print("Freedom space: ", self.freedom_space)
        print("Cognitive state occupation: ", [str(i)+str(j) for i, j in enumerate(self.cog_state)])
        print("Decision space: ", self.decision_space)
        print("Unknown knowledge as default: ", self.default_elements_in_unknown_domain)
        print("********************************")


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    landscape.type(IM_type="Random Directed", K=0, k=22)
    landscape.initialize()

    agent = Agent(N=8, lr=0, landscape=landscape, state_num=4)
    agent.type(name="T shape", generalist_num=4, specialist_num=2)
    agent.describe()
    for _ in range(100):
        agent.cognitive_local_search()
    agent.state = agent.change_cog_state_to_state(cog_state=agent.cog_state)
    agent.converge_fitness = agent.landscape.query_fitness(state=agent.state)
    agent.describe()

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]
