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
        self.decision_space_dict = {}
        self.freedom_space = []  # the alternatives for next step random selection: ['02', '13', '31'] given the current state '310*******'
        # decision_space will not change over time, while freedom_space will change due to the current state occupation

        self.first_search = True
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
        valid_types = ['generalist', 'specialist', 'T shape', "None"]
        if name not in valid_types:
            raise ValueError("Only support 4 types: generalist, specialist, T shape, and None.")
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

        if self.name == "Specialist":
            for cur in self.specialist_knowledge_domain:
                self.decision_space += [cur*self.N+j for j in range(self.state_num)]
                self.decision_space_dict[cur] = list(range(self.state_num))
        elif self.name == "Generalist":
            for cur in self.generalist_knowledge_domain:
                full_states = list(range(self.state_num))
                random.shuffle(full_states)
                random_half_depth = full_states[:int(self.state_num*self.gs_ratio)]
                self.decision_space += [cur*self.N+j for j in random_half_depth]
                self.decision_space_dict[cur] = random_half_depth
        elif self.name == 'T shape':
            for cur in self.specialist_knowledge_domain:
                self.decision_space += [cur*self.N+j for j in range(self.state_num)]
                self.decision_space_dict[cur] = list(range(self.state_num))
            for cur in self.generalist_knowledge_domain:
                full_states = list(range(self.state_num))
                random.shuffle(full_states)
                random_half_depth = full_states[:int(self.state_num*self.gs_ratio)]
                self.decision_space += [cur*self.N+j for j in random_half_depth]
                self.decision_space_dict[cur] = random_half_depth
        else:
            pass

        state_occupation = [i*self.N + j for i,j in enumerate(self.state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]

    def random_select_gst(self):
        """
        Randomly select one element in the freedom space
        :return: updated position and corresponding value for state list
        """
        if len(self.decision_space) == 0:
            raise ValueError("Haven't initialize the decision space; Need to run type() function first")
        state_occupation = [i*self.N + j for i,j in enumerate(self.state)]
        self.freedom_space = [each for each in self.decision_space if each not in state_occupation]
        next_step = int(random.choice(self.freedom_space))
        cur_i, cur_j = next_step // self.N, next_step % self.N
        return cur_i, cur_j

    def independent_search(self,):
        """
        Greedy search in the neighborhood (one-bit stride)
        :return: Updating the state list heading for a higher cognitive fitness
        """
        # Adjust the random initialization of state list
        if self.first_search:
            for i in range(self.N):
                # unknown domain depth will be randomly adjusted to the familiar depth
                if str(i) + str(self.state[i]) not in self.decision_space:
                    self.state[i] = random.choice(self.decision_space_dict[i])
                # unknown domain will not change, or not have a chance to be updated
                # if i not in self.knowledge_domain:
                #     pass
        self.first_search = False
        next_state = self.state.copy()
        updated_position, updated_value = self.random_select_gst()
        next_state[updated_position] = updated_value

        # we don't need to mask the state string, since the unknown domain will not change
        # cognitive_updated_state = self.change_state_to_cog_state(self.state)
        # cognitive_current_state = self.change_state_to_cog_state(current_state)

        # For individual level:
        # the state list is still intact, so we can query the full cache
        # Only when it is team level should we use cognitive fitness to overcome the unknown element. (both unknown depths and unknown domains)
        if self.landscape.query_fitness(next_state) > self.landscape.query_fitness(self.state):
            self.state = next_state
        else:
            pass

    # def change_state_to_cog_state(self, state):
    #     """
    #     There are two kinds of unknown elements:
    #         1) unknown in the width (outside of knowledge domain)-> masked with '*'
    #         2) unknown in the depth (outside of professional level)-> agents cannot select independently (i.e., masked by freedom space)
    #             For team level, there could be some kind of mapping, learning, or communication to improve the cognitive accuracy.
    #     :param state: the intact state list
    #     :return: masked state list
    #     """
    #     temp_state = self.state.copy()
    #     for i in range(self.N):
    #         if i not in self.knowledge_domain:
    #             temp_state[i] = '*'
    #     return temp_state

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
        print("Decision space dict: ", self.decision_space_dict)
        print("Freedom space: ", self.freedom_space)
        print("********************************")


if __name__ == '__main__':
    # Test Example
    from MultiStateInfluentialLandscape import LandScape
    landscape = LandScape(N=10, IM_type="random", IM_random_ratio=None, state_num=4)
    agent = Agent(N=10, lr=0, landscape=landscape, state_num=4)
    agent.type(name="T shape", generalist_num=4,specialist_num=2)
    agent.describe()
