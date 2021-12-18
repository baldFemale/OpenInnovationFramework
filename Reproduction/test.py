# -*- coding: utf-8 -*-
# @Time     : 12/13/2021 16:19
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import time
from itertools import product
import numpy as np
import random

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



# alternative = [[1,2],[3,4]]
# alternative = list(product(*alternative))
# print(alternative)

# decision_space = np.random.choice(10, 4, replace=False).tolist()
# print(decision_space)

# decision_space = [str(i) + str(j) for i in range(10) for j in range(10)]
# print(decision_space)
# decision_space = []
# generalist_knowledge_domain = [0,1,3]
# specialist_knowledge_domain = [4,5]
# N = 10
# decision_space += [str(i) + str(j) for i in specialist_knowledge_domain for j in range(4)]
# for cur in generalist_knowledge_domain:
#     full_states = list(range(4))
#     random.shuffle(full_states)
#     random_half_depth = full_states[:int(4*0.5)]
#     decision_space += [str(cur) + str(j) for j in random_half_depth]
# print(decision_space)
#
#
# state = "1122331122"
# state = list(state)
# state_occupation = [str(i) + str(j) for i, j in enumerate(state)]
# print(state_occupation)
# freedom_space = [each for each in decision_space if each not in state_occupation]
# print(freedom_space)
# for _ in range(10):
#     next_step = random.choice(freedom_space)
#     print(next_step)
#     i = int(next_step)//10
#     j = int(next_step)%10
#     print(i,j)

# state = [3, 3, 2, 1, 0, 0, 2, 3, 3, 2]
# print(state)
# generalist_knowledge_space = [1, 5, 0, 6]
# def change_state_to_cog_state(state):
#     temp_state = []
#     for cur in range(len(state)):
#         if cur in generalist_knowledge_space:
#             temp_state.append(state[cur] // 2)
#         else:
#             temp_state.append(state[cur])
#     return temp_state
#
# result = change_state_to_cog_state(state=state)
# print(result)
# IM = np.array([[0,2,0],[2,0,0],[1,1,0]])
# zero_positions = np.argwhere(IM == 0).tolist()
# fill_with_one_positions = np.random.choice(len(zero_positions), 2, replace=False)
# fill_with_one_positions = [zero_positions[i] for i in fill_with_one_positions]
# # fill_with_one_positions = zero_positions[fill_with_one_positions]
# print(zero_positions)
# print(fill_with_one_positions)

from collections import defaultdict
# decision_space_dict = {}
decision_space_dict = {8: [2, 1], 9: [0, 3], 6: [3, 2], 2: [0, 3]}
for i in range(10):
    print(decision_space_dict[i])