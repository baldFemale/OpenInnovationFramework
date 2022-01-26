# -*- coding: utf-8 -*-
# @Time     : 1/26/2022 17:11
# @Author   : Junyi
# @FileName: Socialized Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape
from Agent import Agent


class Network():
    def __init__(self):
        self.crowd_size = 0
        self.connections = []  # size of crowd num * dynamic links between agents
        self.crowd = []  # a crowd of agent
        self.learn = False  # during the search process
        self.mirror = False  # observe the convergence of others;
        # i.e., copy their pieces of optimization as their initiali opsition\
        self.centralization = None


    def type(self, crowd_size=None, centralization=None):
        self.crowd_size = crowd_size
        valid_centralization_type = ["High", "Middle", "Low"]
        if centralization not in valid_centralization_type:
            raise ValueError("Invalid centralization type")
        self.centralization = centralization
        valid_organization_type = ["Centralized around Generalist", "Centralized around T shape", "Centralized around Specialist"]
        # to check the seed crystal effect
        # GST might have different potential in terms of their ability to inspire others
        # not only idea generation: the seed crystal effect in crowd-based innovation
        # to make it different from the recombination or teaming-up
        # we need to go against the S's super power in attraction, and encourage more centralization degree around the generalist
        # the reason is that the final outcome of such a sub-group or core is less than that aroup generalist
        # traditional opinion is the innovation comes from the margin of a network but also need some legitimacy authorization
        # so they need some impactive node to spread into a right area where nodes would more likely to benefit from that

        # randomly select some hot topic to be exposed to other sub-group can performance better.









