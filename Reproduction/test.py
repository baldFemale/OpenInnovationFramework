# -*- coding: utf-8 -*-
# @Time     : 12/13/2021 16:19
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import time
from itertools import product
import numpy as np

# list is better than the dict
# long_dict = {}
# long_list = list(range(1,4**10))
# for index, value in enumerate(long_list):
#     long_dict[str(index)] = value
#
# start_time_1 = time.time()
# for i in range(10000):
#     print(long_list[i])
# end_time_1 = time.time()
#
# start_time_2 = time.time()
# for i in range(10000):
#     print(long_dict[str(i)])
# end_time_2 = time.time()
#
# print(end_time_1-start_time_1, end_time_2-start_time_2)


# full permutation given the state number and string length
# x = product(range(4), repeat=10)
# for each in x:
#     print(each)


# random type
FC_np = []
IM = np.array([[1,0,0,1],[0,1,1,0],[0,0,1,1],[1,1,1,1]])
# for row in range(len(IM)):
#     print(IM[row])
#     print('xxx')
for row in range(4):
    k = int(sum(IM[row]))
    FC_np.append(np.random.uniform(0, 1, pow(2, k)))
FC_np = np.array(FC_np, dtype=object)
print(FC_np)


