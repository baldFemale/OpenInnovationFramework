# -*- coding: utf-8 -*-
# @Time     : 3/9/2022 14:21
# @Author   : Junyi
# @FileName: ParentLandscape.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from collections import defaultdict
import numpy as np


class ParentLandscape:

    def __init__(self, N, state_num=4):
        """
        Here we only need to indicate the N and state number; no K
        :param N: the length of decision space
        :param state_num: the depth of each decision element
        """
        self.N = N
        self.state_num = state_num
        self.IM, self.dependency_map = np.ones((self.N, self.N), dtype=int), [[1]*(self.N - 1)]*self.N  # full rank IM
        self.FC = None  # fitness configuration -> the key cache to generate the child landscape
        # print("IM: \n", self.IM)
        self.create_fitness_config()

    def create_fitness_config(self,):
        """
        Create the key fitness configuration of chile landscape generation process
        :return: the parent landscape Fitness Configuration
        """
        FC = defaultdict(dict)
        for row in range(len(self.IM)):
            k = int(sum(self.IM[row]))
            for column in range(pow(self.state_num, k)):
                FC[row][column] = np.random.uniform(0, 1)
        self.FC = FC

if __name__ == '__main__':
    # Test Example
    parent = ParentLandscape(N=8, state_num=4)
    FC = parent.FC
    print(len(FC))
