# -*- coding: utf-8 -*-
# @Time     : 10/29/2022 21:42
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
    def __init__(self, proposer=None, evaluator=None):
        self.proposer = proposer
        self.evaluator = evaluator
        self.gg_overlap = None
        self.gs_overlap = None
        self.ss_overlap = None


    def search(self):
        next_cog_state = self.proposer.cog_state.copy()
        index = np.random.choice(self.proposer.expertise_domain)  # ensure the change will not arise from the unknown domains
        space = ["0", "1", "2", "3"]

        if next_cog_state[index] == "A":
            next_cog_state[index] = "B"
        elif next_cog_state[index] == "B":
            next_cog_state[index] = "A"
        elif next_cog_state[index] in ["0", "1", "2", "3"]:
            space.remove(next_cog_state[index])
            next_cog_state[index] = np.random.choice(space)
