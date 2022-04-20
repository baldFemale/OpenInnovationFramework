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

    def __init__(self, N, landscape=None, state_num=4, gs_ratio=0.5):
        """
        The difference between original one and the socialized one:
        1. copied_state: enable the angens to polish the existing ideas/solutions
        2. state_pool: the existing solutions in the community
        3. state_pool_rank: the personal rank regarding the existing solutions
        :param landscape:
        :param state_num:
        :param gs_ratio:
        :param cope_state: the observed state according to different exposure mechanisms
        """
        self.landscape = landscape
        self.N = N
        self.name = "None"
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.state = [str(i) for i in self.state]  # ensure the format of state
        # store the cognitive state list during the search
        self.cog_state = None
        # for different transparency directions, the exposed pool will be controlled in the simulator level
        self.state_pool_G = []  # assigned from externality
        self.state_pool_S = []  # assigned from externality
        self.state_pool_all = [] # assigned from externality
        self.personal_state_pool_rank_G = []  # self-generated after the pool assignment
        self.personal_state_pool_rank_S = []  # self-generated after the pool assignment
        self.personal_state_pool_rank_all = []
        self.fixed_state_pool = None  # fix the pool over time; agent will not pick a pool every time
        self.fixed_openness_flag = None
        self.overall_state_pool_rank_all = []  # assigned from externality; pay attention to the order
        self.overall_state_pool_rank_G = []
        self.overall_state_pool_rank_S = []
        # the order should be consistent in terms of state string and fitness

        self.generalist_knowledge_representation = ["A", "B"]  # for state_num=6, use ["A", "B", "C"]
        self.knowledge_domain = []
        self.generalist_num = 0
        self.specialist_num = 0
        self.specialist_knowledge_domain = []
        self.generalist_knowledge_domain = []  # G_domain + S_domain =knowledge_domain
        self.default_elements_in_unknown_domain = []  # record the unknown domain '*'

        self.gs_ratio = gs_ratio  # generalist know half of the knowledge compared to specialist

        self.decision_space = []  # all manageable elements' index: ['02', '03', '11', '13', '30', '31']
        self.freedom_space = []  # the alternatives for next step random selection: ['02', '13', '31'] given the current state '310*******'
        # self.freedom_space_g = [] # for long jump search; [[], []]
        # self.freedom_space_s = []
        # decision_space will not change over time,
        # while freedom_space keep changing due to the current state occupation
        self.cog_fitness = 0  # store the cog_fitness value for each step in search
        self.converged_fitness = 0  # in the final search loop, record the true fitness compared to the cog_fitness
        self.converged_fitness_rank = 0
        self.potential_fitness = 0  # record the potential achievement; the position advantage to achieve a higher future performance
        self.potential_fitness_rank = 0

        self.valid_state_bit = list(range(self.N))
        self.valid_state_bit += ["A", "B", "*"]  # for cognitive representation

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")

    def update_state_from_exposure(self, exposure_type="Self-interested", G_exposed_to_G=None, S_exposed_to_S=None):
        """
        Re-assign the initial state for upcoming cognitive search
        :return:update the initial state
        """
        success = 0
        if exposure_type == "Random":
            index = np.random.choice(len(self.state_pool_all))
            comparison_state = self.state_pool_all[index]
            comparison_cog_state = self.change_state_to_cog_state(state=comparison_state)
            comparison_cog_fitness = self.landscape.query_cog_fitness(cog_state=comparison_cog_state)
            if comparison_cog_fitness > self.cog_fitness:
                success = 1
                self.state = comparison_state
                self.cog_state = comparison_cog_state
                self.cog_fitness = comparison_cog_fitness
                self.update_freedom_space()
        elif exposure_type == "Self-interested":
            # if (not S_exposed_to_S) and (not G_exposed_to_G):
            #     # when these two parameter are None, refers to the whole state pool
            #     selected_pool_index = -1
            # else:
            #     if self.name == "Generalist":
            #         selected_pool_index = np.random.choice((0, 1), p=[G_exposed_to_G, 1-G_exposed_to_G])  # 0 refers to G pool, while 1 refers to S pool
            #     elif self.name == "Specialist":
            #         selected_pool_index = np.random.choice((0, 1), p=[1-S_exposed_to_S, S_exposed_to_S])
            #     else:
            #         raise ValueError("Outlier of agent name: {0}".format(agent.name))
            selected_pool_index = self.fixed_state_pool
            # fix bugs for gs_proportion = 0 (all G) and 1 (all S)
            if (len(self.state_pool_G) == 0) and (selected_pool_index == 0):
                selected_pool_index = 1
            elif (len(self.state_pool_S) == 0) and (selected_pool_index == 1):
                selected_pool_index = 0

            if selected_pool_index == 0:
                self.create_personal_state_pool_rank(which="G")
                if len(self.state_pool_G) != len(self.personal_state_pool_rank_G):
                    raise ValueError("Length of state_pool_G ({0}) is not equal to length of rank list ({1})"
                                     .format(len(self.state_pool_G), len(self.personal_state_pool_rank_G)))
                selected_state = np.random.choice(len(self.state_pool_G), p=self.personal_state_pool_rank_G)
                selected_state = self.state_pool_G[selected_state]
            elif selected_pool_index == 1:
                self.create_personal_state_pool_rank(which="S")
                if len(self.state_pool_S) != len(self.personal_state_pool_rank_S):
                    raise ValueError("Length of state_pool_S ({0}) is not equal to length of rank list ({1})"
                                     .format(len(self.state_pool_S), len(self.personal_state_pool_rank_S)))
                selected_state = np.random.choice(len(self.state_pool_S), p=self.personal_state_pool_rank_S)
                selected_state = self.state_pool_S[selected_state]
            elif selected_pool_index == -1:
                self.create_personal_state_pool_rank(which="A")
                if len(self.state_pool_all) != len(self.personal_state_pool_rank_all):
                    raise ValueError("Length of state_pool_all ({0}) is not equal to length of rank list ({1})"
                                     .format(len(self.state_pool_all), len(self.personal_state_pool_rank_all)))
                selected_state = np.random.choice(len(self.state_pool_all), p=self.personal_state_pool_rank_all)
                selected_state = self.state_pool_S[selected_state]
            else:
                raise ValueError("Unsupported selected_pool_index: ", selected_pool_index)
            cog_pool_idea = self.change_state_to_cog_state(state=selected_state)
            cog_fitness_pool_idea = self.landscape.query_cog_fitness(cog_state=cog_pool_idea)
            if cog_fitness_pool_idea > self.cog_fitness:
                success = 1
                self.state = selected_state
                self.cog_state = cog_pool_idea
                self.cog_fitness = cog_fitness_pool_idea
                self.update_freedom_space()
        elif exposure_type == "Overall-ranking":
            if (not G_exposed_to_G) and (not S_exposed_to_S):
                selected_pool_index = -1
            else:
                if self.name == "Generalist":
                    selected_pool_index = np.random.choice((0, 1), p=[G_exposed_to_G, 1-G_exposed_to_G])  # 0 refers to G pool, while 1 refers to S pool
                elif self.name == "Specialist":
                    selected_pool_index = np.random.choice((0, 1), p=[1-S_exposed_to_S, S_exposed_to_S])
                else:
                    raise ValueError("Outlier of agent name: {0}".format(agent.name))
            if selected_pool_index == 0:
                if len(self.state_pool_G) != len(self.overall_state_pool_rank_G):
                    raise ValueError("len_1 {0} and len_2 {1}".format(len(self.state_pool_G), len(self.overall_state_pool_rank_G)))
                selected_state = np.random.choice(len(self.state_pool_G), p=self.overall_state_pool_rank_G)
                selected_state = self.state_pool_G[selected_state]
            elif selected_pool_index == 1:
                if len(self.state_pool_S) != len(self.overall_state_pool_rank_S):
                    raise ValueError("len_1 {0} and len_2 {1}".format(len(self.state_pool_G), len(self.overall_state_pool_rank_S)))
                selected_state = np.random.choice(len(self.state_pool_S), p=self.overall_state_pool_rank_S)
                selected_state = self.state_pool_S[selected_state]
            elif selected_pool_index == -1:
                if len(self.state_pool_all) != len(self.overall_state_pool_rank_all):
                    raise ValueError("len_1 {0} and len_2 {1}".format(len(self.state_pool_all), len(self.overall_state_pool_rank_all)))
                selected_state = np.random.choice(len(self.state_pool_all), p=self.overall_state_pool_rank_all)
            else:
                raise ValueError("Unsupported selected_pool_index")
            cog_pool_idea = self.change_state_to_cog_state(state=selected_state)
            cog_fitness_pool_idea = self.landscape.query_cog_fitness(cog_state=cog_pool_idea)
            if cog_fitness_pool_idea > self.cog_fitness:
                success = 1
                self.state = selected_state
                self.cog_state = cog_pool_idea
                self.cog_fitness = cog_fitness_pool_idea
                self.update_freedom_space()
        return success

    def vote_for_state_pool(self, which="All"):
        """
        Vote for the whole pool; the weights are measured by each cognitive fitness
        voting does not update the rank list, just return the rank for aggregated rank in crowd level
        :return:the dict of each state with its cog_fitness
        """
        personal_pool_rank = {}
        if (which == "All") or (which == "A"):
            if not self.state_pool_all:
                raise ValueError("Need state_pool_all assigned by the simulator."
                                 " state_pool should be G+S, with consistent order of pairs of state and fitness")
            for each_state in self.state_pool_all:
                cog_state = self.change_state_to_cog_state(state=each_state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state)
                personal_pool_rank[''.join(each_state)] = perceived_fitness
        elif which == "S":
            if not self.state_pool_S:
                raise ValueError("Need state_pool_S assigned by the simulator.")
            for each_state in self.state_pool_S:
                cog_state = self.change_state_to_cog_state(state=each_state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state)
                personal_pool_rank[''.join(each_state)] = perceived_fitness
        elif which == "G":
            if not self.state_pool_G:
                raise ValueError("Need state_pool_G assigned by the simulator.")
            for each_state in self.state_pool_G:
                cog_state = self.change_state_to_cog_state(state=each_state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state)
                personal_pool_rank[''.join(each_state)] = perceived_fitness
        return personal_pool_rank

    def create_personal_state_pool_rank(self, which="A"):
        """
        In vote, we generate the dict with state string and its fitness, because we need to sum acrom agents
        In create, we generate the list, which has the same order as the pool list
        :param which: A, G, and S
        :return: the fitness list
        """
        # the state_pool corresponds to the state_pool_rank, although they are both list.
        rank_temp = []
        if (which == "All") or (which == "A"):
            if len(self.state_pool_all) == 0:
                self.state_pool_all = self.state_pool_G + self.state_pool_S
                if len(self.state_pool_all) == 0:
                    raise ValueError("Need to assign state pool first")
            for state in self.state_pool_all:
                cog_state_solution = self.change_state_to_cog_state(state=state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state=cog_state_solution)
                rank_temp.append(perceived_fitness)
            rank_temp = [i/sum(rank_temp) for i in rank_temp]
            self.personal_state_pool_rank_all = rank_temp
        elif which == "G":
            if len(self.state_pool_G) == 0:
                if self.generalist_num != 0:
                    raise ValueError("Need to assign the G state pool first")
            for state in self.state_pool_G:
                cog_state_solution = self.change_state_to_cog_state(state=state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state=cog_state_solution)
                rank_temp.append(perceived_fitness)
            rank_temp = [i/sum(rank_temp) for i in rank_temp]
            self.personal_state_pool_rank_G = rank_temp
        elif which == "S":
            if len(self.state_pool_S) == 0:
                if self.specialist_num != 0:
                    raise ValueError("Need to assign the S state pool first")
            for state in self.state_pool_S:
                cog_state_solution = self.change_state_to_cog_state(state=state)
                perceived_fitness = self.landscape.query_cog_fitness(cog_state=cog_state_solution)
                rank_temp.append(perceived_fitness)
            rank_temp = [i/sum(rank_temp) for i in rank_temp]
            self.personal_state_pool_rank_S = rank_temp
        else:
            raise ValueError("Unsupported which type of {0}".format(which))

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
        self.cog_fitness = self.landscape.query_cog_fitness(cog_state=self.cog_state)
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
        """
        The core of this model where we define a consistent cognitive search framework for G/S role
        The Generalist domain follows the average pooling search
        The Specialist domain follows the mindset search
        There is a final random mapping after cognitive convergence, to map a vague state into a definite state
        """
        updated_position, updated_value = self.randomly_select_one()
        if updated_position in self.generalist_knowledge_domain:
            next_cog_state = self.cog_state.copy()
            next_cog_state[updated_position] = updated_value
            current_cog_fitness = self.landscape.query_cog_fitness(self.cog_state)
            next_cog_fitness = self.landscape.query_cog_fitness(next_cog_state)
            if next_cog_fitness > current_cog_fitness:
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                # add the mapping during the search because we need the imitation
                self.state = self.change_cog_state_to_state(cog_state=self.cog_state)
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
            else:
                self.cog_fitness = current_cog_fitness
        elif updated_position in self.specialist_knowledge_domain:
            cur_cog_state_with_default = self.cog_state.copy()
            next_cog_state = self.cog_state.copy()
            next_cog_state[updated_position] = updated_value
            next_cog_state_with_default = next_cog_state.copy()
            # replace the * with default value, that is, mindset
            for default_mindset in self.default_elements_in_unknown_domain:
                # default_mindset "32" refers to "2" in location "3"
                cur_cog_state_with_default[int(default_mindset[0])] = default_mindset[1]
                next_cog_state_with_default[int(default_mindset[0])] = default_mindset[1]
            current_cog_fitness = self.landscape.query_cog_fitness(cur_cog_state_with_default)
            next_cog_fitness = self.landscape.query_cog_fitness(next_cog_state_with_default)
            if next_cog_fitness > current_cog_fitness:
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                self.state = self.change_cog_state_to_state(cog_state=self.cog_state)
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
            else:
                self.cog_fitness = current_cog_fitness
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

    def describe(self):
        print("*********Agent information********* ")
        print("Agent type: ", self.name)
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state list: ", self.state)
        print("Current cognitive state list: ", self.cog_state)
        print("Current cognitive fitness: ", self.cog_fitness)
        print("Converged fitness: ", self.converged_fitness)
        print("Knowledge/Manageable domain: ", self.knowledge_domain)
        print("Specialist knowledge domain: ", self.specialist_knowledge_domain)
        print("Generalist knowledge domain: ", self.generalist_knowledge_domain)
        print("Element number: ", self.element_num)
        print("G/S knowledge ratio: ", self.gs_ratio)
        print("Freedom space: ", self.freedom_space)
        print("Cognitive state occupation: ", [str(i)+str(j) for i, j in enumerate(self.cog_state)])
        print("Decision space: ", self.decision_space)
        print("Unknown knowledge as default: ", self.default_elements_in_unknown_domain)
        print("********************************")


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    landscape.type(IM_type="Factor Directed", K=0, k=42)
    landscape.initialize()

    agent = Agent(N=8, landscape=landscape, state_num=4)
    agent.type(name="T shape", generalist_num=4, specialist_num=2)
    agent.describe()
    for _ in range(100):
        agent.cognitive_local_search()
    agent.state = agent.change_cog_state_to_state(cog_state=agent.cog_state)
    agent.converge_fitness = agent.landscape.query_fitness(state=agent.state)
    agent.describe()
    print("END")

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]


