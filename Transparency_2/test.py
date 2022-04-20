# -*- coding: utf-8 -*-
# @Time     : 12/13/2021 16:19
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import itertools
import time
from itertools import product
import numpy as np
import random
import matplotlib.pylab as plt
import os
import sys

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
# for i in range(10):
#     print(decision_space_dict[i])

# landscape_dict = {"Random Directed": {N:}}

# def test(N, state_num, k, name):
#     print(N, state_num, k, name)
#
#
# param = {"N": 10, "state_num": 4, "k": 22, "name": "Random Directed"}
#
# test(**param)
# from collections import defaultdict
#
# rewards = ["A", "B", "C", "D"]
# p = [0.0075, 0.076, 0.3375, 0.579]
# N = 1000
#
# count2 = {}
# for each in rewards:
#     if each not in count2.keys():
#         count2[each] = 0
#
# A_count = 0
# for i in range(1, N+1):
#     temp = np.random.choice(rewards, 1, p=p, replace=True).tolist()[0]
#     if temp == "A":
#         A_count = 0
#     else:
#         A_count += 1
#     if A_count == 40:
#         A_count = 0
#         temp = "A"
#     count2[temp] += 1
#
# for key, value in count2.items():
#     temp = value / N
#     count2[key] = temp
# print("Count2: ", count2)
#
#
# plt.figure(figsize=(9, 3))
# plt.subplot(131)
# plt.bar(count2.keys(), count2.values())
# plt.subplot(132)
# plt.scatter(count2.keys(), count2.values())
# plt.subplot(133)
# plt.plot(count2.keys(), count2.values())
# plt.suptitle('Categorical Plotting')
# plt.show()
# import re
# re_x_rule = r'N10(.*?)HeteroTeamSerial'
# x = r'C:\Python_Workplace\OpenInnovationFramework\Reproduction\N10K10HeteroTeamSerial'
# y = r'C:\Python_Workplace\OpenInnovationFramework\Reproduction\output7'
# match = re.search(pattern=re_x_rule, string=x).group(1)
# print("Match1", match)
# # match2 = re.search(pattern=re_x_rule,string=y)
# # print("Match2", match2)
#
# dir = ['K_10', 'K_2', 'K_4', 'K_6', 'K_8', 'K_0']
# dir.sort()
# print(dir)
# def cog_state_alternatives(cog_state=None):
#     alternative_pool = []
#     for bit in cog_state:
#         if bit in ["0", "1", "2", "3"]:
#             alternative_pool.append(bit)
#         elif bit == "A":
#             alternative_pool.append(["0", "1"])
#         elif bit == "B":
#             alternative_pool.append(["2", "3"])
#         elif bit == "C":
#             alternative_pool.append(["4", "5"])
#         elif bit == "*":
#             alternative_pool.append(["0", "1", "2", "3"])
#         else:
#             raise ValueError("Unsupported bit value: ", bit)
#     return [i for i in product(*alternative_pool)]
#
# cog_state = ["A", "B", "1", "0", "*"]
# res = cog_state_alternatives(cog_state=cog_state)
# # print(res)
# A = "A_" + "ssfas"
# x = A.replace("A_", '')
# print(x)

# def loop():
#     x = 1
#     y = [each for each in range(19)]
#     return x, y
#
# a = loop()[0]
# print(a)
# set 会改变重复元素数组的排序吗
import pickle
temp = None



# file = r'C:\Python_Workplace\hpc-0126\nk\Factor\5IM_Generalist_Factor Directed_N10_K0_k44_E20_G10_S0'
# with open(file, 'rb') as in_file:
#     IM = pickle.load(in_file)
# # for each in np.array(IM):
# #     print(each)
# file = r'C:\Python_Workplace\hpc-0126\nk\Factor\6Knowledge_Generalist_Factor Directed_N10_K0_k44_E20_G10_S0'
# with open(file, 'rb') as in_file:
#     knowledge_domain = pickle.load(in_file)
#
# C_row_match_temp = 0
# for l in range(500):
#     each_IM = IM[l]
#     print(np.array(each_IM))
#     for a in range(500):
#         each_agent_knowledge = knowledge_domain[l][a]
#         print(each_agent_knowledge)
#         for column in range(10):
#             if column in each_agent_knowledge[1]:
#                 # print(sum(each_IM[:][column]))
#                 print(each_IM[:][column])
#                 # C_row_match_temp += sum(each_IM[:][column]) * 2
#                 # if column in each_agent:
#                 #     C_row_match_temp += sum(IM[:][column]) *4
#         break
#     break
# print(C_row_match_temp)
# from itertools import combinations
# import math
# N = 10
# K = 6
# # selections = list(itertools.permutations(range(N), K)) # 排列permutations
# selections_2 = list(combinations(range(N), K)) # 组合combination
# print(selections_2)  # 151200
# print(len(selections_2))  # 210
# def combinations_num(n, k):
#     return math.factorial(n)/math.factorial(k)/math.factorial((n-k))
# x = combinations_num(10, 6)
# print(x)
# k = 64
# K = 0
# absolute_k = K if K else k // 10
# print("absolute_k:", absolute_k)

# test_x = list(range(10))
# test_y = test_x.copy()
# print(test_y)

# state = ['2', '2', '3', '0']
# binary_index = "".join(state)
# index = int(binary_index, 4)
# print(index)

# test = np.eye(10, dtype=int)
# print(test)

# import pandas as pd
#
# pd.set_option('display.max_colwidth', -1)
# print(pd.options.display.max_colwidth)
# a = [[1,2,3], [0,2,3]]
# a = np.array(a)
# b = [[1,2,3], [1,2,3]]
# b = np.array(b)
# print((a==b).all())
# index = np.random.choice(len(a))
# print(a[index])

# x = {"111": 2, "222": 3}
# x_ = max(x, key=x.get)
# print(x_)
# print(list(x_))

# list_1 = [[0,0,1],
#           [1,1,0],
#           [0,0,0]]
# list_2 = [[1,0,1],
#           [0,1,0],
#           [0,0,0]]
# list_3 = [[0,0,1],
#           [1,1,0],
#           [0,0,0]]
# list_1 = np.array(list_1)
# list_2 = np.array(list_2)
# list_3 = np.array(list_3)
# if (list_1 == list_3).all():
#     print("Yes")
# else:
#     print("No")

# p = [0.16666666666666666, 0.3333333333333333, 0.5]
# state_pool = [['1', '1'], ['1', '2'], ['1', '3']]
# pool_state_index = np.random.choice(len(state_pool), p=p)
# pool_state = state_pool[pool_state_index]
# print(pool_state)
# G_exposed_to_G_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# S_exposed_to_S_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# alternative_pool = [G_exposed_to_G_list, S_exposed_to_S_list]
# combination = [i for i in product(*alternative_pool)]
# print(combination)
# print(len(combination))

# inconsistent_len = 10 - int(0.5 * 10)
# inconsistent_index = np.random.choice(range(10), inconsistent_len, replace=False)
# print(inconsistent_index)
# test = np.random.choice(range(4))
# print(test)
# print(1 == 1.0)
# print(0.5 < 1)
# print(1.0 < 1.00)

# G_exposed_to_G_list = [0, 0.4, 0.8, 1.0]
# S_exposed_to_S_list = [0, 0.4, 0.8, 1.0]
# alternative_pool = [G_exposed_to_G_list, S_exposed_to_S_list]
# test = [i for i in product(*alternative_pool)]
# print(len(test))

# 3D figure
# def f(x, y):
#     return np.cos(np.sqrt(x ** 2 + y ** 2))
#
# x = np.linspace(-6, 6, 5)
# y = np.linspace(-6, 6, 5)
#
# X, Y = np.meshgrid(x, y)
#
# Z = f(X, Y)
# print(X)
# print("Z.shape: ", Z.shape)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none', color="grey")
# ax.set_title('surface')
# plt.show()

# Creating dataset
# x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
# y = x.copy().T  # transpose
# z = (np.sin(x ** 2) + np.cos(y ** 2))
#
# # Creating figure
# fig2 = plt.figure(figsize=(14, 9))
# ax2 = plt.axes(projection='3d')
# # Creating plot
# ax2.plot_surface(x, y, z)
#
# # show plot
# plt.show()

# G_exposed_to_G_list = [0, 0.25, 0.5, 0.75, 1.0]
# # Quality
# S_exposed_to_S_list = [0, 0.25, 0.5, 0.75, 1.0]
# alternative_pool = [G_exposed_to_G_list, S_exposed_to_S_list]
# alternative_pool = [each for each in alternative_pool if sum(each) != 0]
# print(alternative_pooEl)
# data = [[1,2], [3,4]]
# data = np.array(data,dtype=object)
# data = data.reshape((4,1))
# print(data)

#
# # Simulation Configuration
# landscape_iteration = 100
# agent_num = 400
# search_iteration = 50
# # Parameter
# N = 6
# state_num = 4
# knowledge_num = 8
# K_list = [1, 3, 5]
# frequency_list = [1]
# openness_list = [1.0]
# quality_list = [0, 0.25, 0.5, 0.75, 1.0]
# G_exposed_to_G_list = [0.5]
# S_exposed_to_S_list = [0.5]
# gs_proportion_list = [0.5]
# exposure_type_list = ["Self-interested"]
#
#
# if __name__ == '__main__':
#     k = 0
#     for K in K_list:
#         for socialization_freq in frequency_list:
#             for openness in openness_list:
#                 for quality in quality_list:
#                     for G_exposed_to_G in G_exposed_to_G_list:
#                         for S_exposed_to_S in S_exposed_to_S_list:
#                             for gs_proportion in gs_proportion_list:
#                                 for exposure_type in exposure_type_list:
#                                     p = mp.Process(target=loop,
#                                                    args=(k, K, exposure_type, socialization_freq, quality, openness,
#                                                          S_exposed_to_S, G_exposed_to_G, gs_proportion))
#                                     p.start()

# data = np.array([[1,1,2], [3,3,4],[1,5,3], [1,6,8]])
# data = data.reshape((1,-1))
# print(data)
# data = np.unique(data)
# print(data)
# print(type(data))