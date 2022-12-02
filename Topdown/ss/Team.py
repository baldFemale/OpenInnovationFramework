# -*- coding: utf-8 -*-
# @Time     : 2/12/2022 21:22
# @Author   : Junyi
# @FileName: Team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Tshape import Tshape
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Pool
from multiprocessing import Semaphore
import pickle
import math


class Team:
    def __init__(self, agent_1=None, agent_2=None, N=None, state_num=None):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.solution = np.random.choice(range(state_num), N).tolist()
        self.solution = [str(i) for i in self.solution]  # state format: string
        self.agent_1.align_default_state(state=self.solution)
        self.agent_2.align_default_state(state=self.solution)

    def search(self):
        # Agent 1 search
        self.agent_1.align_default_state(state=self.solution)
        next_cog_state = self.agent_1.cog_state.copy()
        index = np.random.choice(self.agent_1.expertise_domain)
        if next_cog_state[index] == "A":
            next_cog_state[index] = "B"
        else:
            next_cog_state[index] = "A"
        next_cog_fitness = self.agent_1.landscape.query_cog_fitness_partial(cog_state=next_cog_state,
                                                                            expertise_domain=self.agent_1.expertise_domain)
        if next_cog_fitness > self.agent_1.cog_fitness:
            self.agent_1.cog_state = next_cog_state
            self.agent_1.cog_fitness = next_cog_fitness
            self.agent_1.solution[index] = self.agent_1.cog_state[index]

        # Agent 2 search
        self.agent_2.align_default_state(state=self.solution)
        next_cog_state = self.agent_2.cog_state.copy()
        index = np.random.choice(self.agent_2.expertise_domain)
        if next_cog_state[index] == "A":
            next_cog_state[index] = "B"
        else:
            next_cog_state[index] = "A"
        next_cog_fitness = self.agent_2.landscape.query_cog_fitness_partial(cog_state=next_cog_state,
                                                                            expertise_domain=self.agent_2.expertise_domain)
        if next_cog_fitness > self.agent_2.cog_fitness:
            self.agent_2.cog_state = next_cog_state
            self.agent_2.cog_fitness = next_cog_fitness
            self.solution[index] = self.agent_2.cog_state[index]