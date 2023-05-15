# -*- coding: utf-8 -*-
# @Time     : 5/15/2023 20:54
# @Author   : Junyi
# @FileName: landscape_run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Landscape import Landscape
from CogLandscape import CogLandscape
import numpy as np
import pickle


N = 9
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
state_num = 4
expertise_domain_list = [18, 12, 6]  # 9, 6, 3 domains

for K in K_list:
    for expertise_domain in expertise_domain_list:
        np.random.seed(None)
        landscape = Landscape(N=N, K=K, state_num=state_num)
        cog_landscape = CogLandscape(landscape=landscape, expertise_domain=list(range(N)),
                                     expertise_representation=["A", "B"], norm=False)
        data_1 = list(landscape.cache.values())
        data_2 = list(cog_landscape.cache.values())
        result = [data_1, data_2]

        with open("landscape_K_{0}_E_{1}".format(K, expertise_domain), 'wb') as out_file:
            pickle.dump(result, out_file)