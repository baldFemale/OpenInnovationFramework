# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
from collections import defaultdict
from tools import *


class Agent:

    def __init__(self, N, lr=0, landscape=None, state_num=4, gs_ratio=0.5):
        self.landscape = landscape
        self.N = N
        self.name = "None"
        self.state_num = state_num
        # intact state string -> may be outside of the agent's capability
        # -> Does this initialization matter? Or we may start from the manageable elements
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        # masked state string -> using '*' to represent the agent perceived state
        self.cog_state = None

        self.knowledge_domain = None
        self.generalist_num = None
        self.specialist_num = None
        self.specialist_knowledge_domain = None
        self.generalist_knowledge_domain = None  # G_domain + S_domain =knowledge_domain
        self.gs_ratio = gs_ratio  # generalist know half of the knowledge compared to specialist
        self.lr = lr
        self.decision_space = []  # all manageable elements' index: ['02', '03', '11', '13', '30', '31']
        self.freedom_space = []  # the alternatives for next step random selection: ['02', '13', '31'] given the current state '310*******'
        # decision_space will not change over time, while freedom_space will change due to the current state occupation

        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape mismatch. Please check your N and state number")


    def type(self, name=None, specialist_num=0, generalist_num=0, element_num=None, gs_ratio=0.5):
        """
        Allocate one certain type to the agent
        :param name: the agent role
        :param specialist_num: the
        :param gs_ratio: the ratio of knowledge between G and S
        :return: Updating the agent characters
        """
        self.name = name
        self.gs_ratio = gs_ratio
        self.generalist_num = generalist_num
        self.specialist_num = specialist_num
        domain_number = self.generalist_num + self.specialist_num
        valid_types = ['generalist', 'specialist', 'T shape', "None"]
        if name not in valid_types:
            raise ValueError("Only support 4 types: generalist, specialist, T shape, and None.")
        if (self.name == "Generalist") & (specialist_num != 0):
            raise ValueError("Generalist cannot have specialist number")
        if (self.name == "Specialist") & (generalist_num != 0):
            raise ValueError("Generalist cannot have specialist number")
        if (self.name == "T shape") & (generalist_num*specialist_num == 0):
            raise ValueError("T shape mush have both generalist number and specialist number")
        if element_num % int(self.gs_ratio*self.state_num) != 0:
            raise ValueError("Element number ({0}) cannot be exactly divided by {1}".format(element_num, int(self.gs_ratio*self.state_num)))
        if element_num < int(self.gs_ratio*self.state_num) * domain_number:
            raise ValueError("Element number ({0}) is not greater than {1}, which is the minimum for generalist type".format(element_num, int(self.gs_ratio*self.state_num) * domain_number))

        self.knowledge_domain = np.random.choice(self.N, domain_number, replace=False).tolist()
        if element_num:
            self.element_num = element_num  # record the total number of element that agents can change
        else:
            self.element_num = int(self.state_num * self.generalist_num * self.gs_ratio + self.state_num * self.specialist_num)

        self.specialist_knowledge_domain = np.random.choice(self.knowledge_domain, specialist_num, replace=False).tolist()
        self.generalist_knowledge_domain = [
            cur for cur in self.knowledge_domain if cur not in self.specialist_knowledge_domain
        ]

        if self.name == "Specialist":
            if len(self.generalist_knowledge_domain) != 0:
                raise ValueError("Specialist cannot have any generalist domain.")
            self.decision_space = [str(i) + str(j) for i in self.specialist_knowledge_domain for j in range(self.state_num)]
        elif self.name == "Generalist":
            if len(self.specialist_knowledge_domain) != 0:
                raise ValueError("Specialist cannot have any specialist domain.")
            for cur in self.generalist_knowledge_domain:
                full_states = list(range(self.state_num))
                random.shuffle(full_states)
                random_half_depth = full_states[:int(self.state_num*self.gs_ratio)]
                self.decision_space += [str(cur) + str(j) for j in random_half_depth]
        elif self.name == 'T shape':
            if len(self.specialist_knowledge_domain)*len(self.generalist_knowledge_domain) == 0:
                raise ValueError("T shape should have both specialist and generalist domain.")
            self.decision_space += [str(i) + str(j) for i in self.specialist_knowledge_domain for j in range(self.state_num)]
            for cur in self.generalist_knowledge_domain:
                full_states = list(range(self.state_num))
                random.shuffle(full_states)
                random_half_depth = full_states[:int(self.state_num*self.gs_ratio)]
                self.decision_space += [str(cur) + str(j) for j in random_half_depth]
        else:
            pass

        state_occupation = [str(i) + str(j) for i,j in enumerate(self.state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]

    def random_select_gst(self):
        """
        Randomly select one element in the freedom space
        :return: updated position and corresponding value for state list
        """
        if len(self.decision_space) == 0:
            raise ValueError("Haven't initialize the decision space; Need to run type() function first")
        state_occupation = [str(i) + str(j) for i,j in enumerate(self.state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]
        next_step = int(random.choice(self.freedom_space))
        cur_i, cur_j = next_step//10, next_step%10
        return cur_i, cur_j

    def independent_search(self, ):
        """
        Greedy search in the neighborhood (one-bit stride)
        :return: Updating the state list heading for a higher cognitive fitness
        """
        current_state = self.state.copy()
        updated_position, updated_value = self.random_select_gst()
        self.state[updated_position] = updated_value

        cognitive_updated_state = self.change_state_to_cog_state(self.state)
        cognitive_current_state = self.change_state_to_cog_state(current_state)

        if self.landscape.query_cog_fitness_gst(
            cognitive_updated_state, self.generalist_knowledge_domain, self.specialist_knowledge_domain
        ) > self.landscape.query_cog_fitness_gst(
            cognitive_current_state, self.generalist_knowledge_domain, self.specialist_knowledge_domain
        ):
            pass
        else:
            self.state = current_state  # roll back

    def change_state_to_cog_state(self, state):
        """
        There are two kinds of unknown elements:
            1) unknown in the width (outside of knowledge domain)-> masked with '*'
            2) unknown in the depth (outside of professional level)-> agents cannot select independently;
                For team level, there could be some kind of mapping, learning, or communication to improve the cognitive accuracy.
        :param state: the intact state list
        :return: masked state list
        """


    def learn(self, target_):
        pass

    def describe(self):
        print("*********Agent information********* ")
        print("Agent type: ", self.name)
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state list: ", self.state)
        print("Knowledge/Manageable domain: ", self.knowledge_domain)
        print("Specialist knowledge domain: ", self.specialist_knowledge_domain)
        print("Generalist knowledge domain: ", self.generalist_knowledge_domain)
        print("Element number: ", self.element_num)
        print("Learning rate: ", self.lr)
        print("G/S knowledge ratio: ", self.gs_ratio)
        print("Decision space: ", self.decision_space)
        print("Freedom space: ", self.freedom_space)
        print("********************************")


if __name__ == '__main__':
    # Test Example
    from MultiStateInfluentialLandscape import LandScape
    landscape = LandScape(N=10, K=4, IM_type="random", IM_random_ratio=None, state_num=4)
    agent = Agent(N=10, lr=0, landscape=landscape, state_num=4)
    agent.type(name="T shape", generalist_num=4,element_num=24,specialist_num=2)
    agent.describe()
