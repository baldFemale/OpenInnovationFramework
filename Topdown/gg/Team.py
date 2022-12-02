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
