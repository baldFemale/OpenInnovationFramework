# -*- coding: utf-8 -*-
# @Time     : 1/26/2022 17:11
# @Author   : Junyi
# @FileName: Socialized Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
import numpy as np
from Landscape import Landscape
from Socialized_Agent import Agent


class Network:

    def __init__(self, crowd_size=200, crowd=None):
        self.crowd_size = crowd_size  # e.g., 10 million individuals
        self.crowd = crowd  # a crowd of agent
        self.crowd_states = {}  # state: rank

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








