# -*- coding: utf-8 -*-
# @Time     : 5/20/2023 17:11
# @Author   : Junyi
# @FileName: Test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import sys
sys.path.append("../Main/")
from Landscape import Landscape
from CogLandscape import CogLandscape
from BinaryLandscape import BinaryLandscape
import numpy as np

N = 9
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
state_num = 4
expertise_domain_list = 18
np.random.seed(None)
landscape = Landscape(N=N, K=4, state_num=state_num, norm="MaxMin")
landscape.describe()