# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 19:59
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape


class Generalist:
    def __init__(self, N=None, landscape=None, state_num=4, expertise_amount=None):
        self.landscape = landscape
        self.N = N
        self.state_num = state_num
        self.state = np.random.choice(range(self.state_num), self.N).tolist()
        self.state = [str(i) for i in self.state]  # state format: string
        self.generalist_knowledge_representation = ["A", "B"]
        self.expertise_domain = np.random.choice(range(self.N), expertise_amount / 2).tolist()
        self.cog_state = self.state_2_cog_state()
        self.cog_fitness = None
        self.fitness = None

        if not self.landscape:
            raise ValueError("Agent need to be assigned a landscape")
        if (self.N != landscape.N) or (self.state_num != landscape.state_num):
            raise ValueError("Agent-Landscape Mismatch: please check your N and state number.")

    def leaning_from_exposure(self, pool=None):
        pass

    def cognitive_local_search(self):
        """
        The core of this model where we define a consistent cognitive search framework for G/S role
        The Generalist domain follows the average pooling search
        The Specialist domain follows the mindset search
        There is a final random mapping after cognitive convergence, to map a vague state into a definite state
        """
        if len(self.decision_space) == 0:
            raise ValueError("Haven't initialize the decision space; Need to run type() function first")
        next_step = random.choice(self.freedom_space)
        updated_index, updated_value = next_step[0], next_step[1]  # both are string
        if updated_index in self.generalist_domain:
            next_cog_state = self.cog_state.copy()
            next_cog_state[int(updated_index)] = updated_value
            current_cog_fitness = self.landscape.query_cog_fitness(self.cog_state)
            next_cog_fitness = self.landscape.query_cog_fitness(next_cog_state)
            if next_cog_fitness > current_cog_fitness:
                self.cog_state = next_cog_state
                self.cog_fitness = next_cog_fitness
                # add the mapping during the search because we need the imitation
                self.state = self.change_cog_state_to_state(cog_state=self.cog_state)
                self.potential_fitness = self.landscape.query_potential_fitness(cog_state=self.cog_state)
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
        elif updated_index in self.specialist_domain:
            cur_cog_state_with_default = self.cog_state.copy()
            next_cog_state = self.cog_state.copy()
            next_cog_state[int(updated_index)] = updated_value
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
                self.potential_fitness = self.landscape.query_potential_fitness(cog_state=self.cog_state)
                self.update_freedom_space()  # whenever state change, freedom space need to be changed
            else:
                self.cog_fitness = current_cog_fitness
        else:
            raise ValueError("The picked next step go outside of G/S knowledge domain")

    def state_2_cog_state(self, state=None):
        cog_state = self.state.copy()
        for index, bit_value in enumerate(state):
            if index in self.expertise_domain:
                if bit_value in ["0", "1"]:
                    cog_state[index] = "A"
                elif bit_value in ["2", "3"]:
                    cog_state[index] = "B"
                else:
                    raise ValueError("Only support for state number = 4")
            else:
                cog_state[index] = "*"
        return cog_state

    def cog_state_2_state(self, cog_state=None):
        state = cog_state.copy()
        for index, bit_value in enumerate(cog_state):
            if index not in self.expertise_domain:
                state[index] = str(random.choice(range(self.state_num)))
            else:
                if bit_value == "A":
                    state[index] = random.choice(["0", "1"])
                elif bit_value == "B":
                    state[index] = random.choice(["2", "3"])
                else:
                    raise ValueError("Unsupported state element: ", bit_value)
        return state

    def describe(self):
        print("N: ", self.N)
        print("State number: ", self.state_num)
        print("Current state list: ", self.state)
        print("Current cognitive state list: ", self.cog_state)
        print("Current cognitive fitness: ", self.cog_fitness)
        print("Converged fitness: ", self.fitness)
        print("Expertise domain: ", self.expertise_domain)


if __name__ == '__main__':
    # Test Example
    landscape = Landscape(N=8, state_num=4)
    landscape.type(IM_type="Factor Directed", K=0, k=42)
    landscape.initialize()

    agent = Agent(N=8, landscape=landscape, state_num=4)
    agent.type(name="T shape", generalist_num=1, specialist_num=7)
    agent.describe()
    for _ in range(100):
        agent.cognitive_local_search()
    agent.state = agent.change_cog_state_to_state(cog_state=agent.cog_state)
    agent.converged_fitness = agent.landscape.query_fitness(state=agent.state)
    agent.describe()
    print("END")

# does this search space or freedom space is too small and easy to memory for individuals??
# because if we limit their knowledge, their search space is also limited.
# Compared to the original setting of full-known knowledge, their search space is limited.
# Thus, we can increase the knowledge number to make it comparable to the original full-knowledge setting.
# [0, 0, 1, 3, 3, 3, 2, 0, 1, 0]
# [2, 1, 1, 1, 3, 3, 0, 3, 1, 0]


