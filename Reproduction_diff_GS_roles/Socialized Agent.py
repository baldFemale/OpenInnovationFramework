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
        self.crowd_size = 0  # e.g., 10 million individuals
        self.connections = []  # size of crowd num * dynamic links between agents
        self.crowd = []  # a crowd of agent
        self.learn = False  # during the search process
        self.mirror = False  # observe the convergence of others;
        # i.e., copy their pieces of optimization as their initial position
        self.centralization = None
        self.seed = None  # from several seed or high exposure individuals we start the crowd search
        self.exposure_probabilities = []  # for each individual, they will have some probability to be exposed.
        # (e.g., in the top of platform)
        self.impact_distance = 5  # on average

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

    def create_seed(self, seed_ratio=0.01, seed_g=0, seed_s=0):
        """
        Generate some convergence point, as the initial seed pool for the crowd to learn
        """
        if seed_g + seed_s == 0:
            raise ValueError("Need to specify the initial seed characters.")


        for i in range(seed_ratio*self.crowd_size):
            agent_i = Agent()

    def assemble(self, gravitation=10):
        """
        First assemble process around the initial seed
        :param gravitation: the number of crowd attracted by the central seed
        :return:
        """

    def random_blossom(self, gravitation=10):
        """
        Overflowed crowd randomly initialize new seed and self-assemble, leading to a developed community
        :param gravitation:
        :return:
        """

    def rank_directed_blossom(self, gravitation=10, rank_visibility=20):
        """
         Overflowed crowd initialize new seed and self-assemble based on the top rank list, which is more visible to newcomers.
        :param gravitation:
        :param rank_visibility:
        :return:
        """

    def interest_directed_blossom(self, gravitation=10, interest_visibility=10):
        """
        Overflowed crowd initialize new seed and self-assemble based on interest.
        :param gravitation:
        :param interest_visibility:
        :return:
        """

    def platform_intervention(self, rule_func=None):
        """
        According to a certain rule of intervention, platform might achieve a higher aggregated performance level.
        :param rule_func: platform intervention design
        :return:
        """













