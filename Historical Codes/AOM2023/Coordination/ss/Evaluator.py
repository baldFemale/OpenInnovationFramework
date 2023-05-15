# -*- coding: utf-8 -*-
# @Time     : 10/3/2022 22:31
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import container

data_folder = r"F:\data\gst-1112\Coordination_2\ss"
s1_performance_0_file = data_folder + r"\s1_performance_across_K_0"
s1_performance_1_file = data_folder + r"\s1_performance_across_K_1"
s1_performance_2_file = data_folder + r"\s1_performance_across_K_2"
s1_performance_3_file = data_folder + r"\s1_performance_across_K_3"
with open(s1_performance_0_file, 'rb') as infile:
    s1_performance_0 = pickle.load(infile)
with open(s1_performance_1_file, 'rb') as infile:
    s1_performance_1 = pickle.load(infile)
with open(s1_performance_2_file, 'rb') as infile:
    s1_performance_2 = pickle.load(infile)
with open(s1_performance_3_file, 'rb') as infile:
    s1_performance_3 = pickle.load(infile)

s2_performance_0_file = data_folder + r"\s2_performance_across_K_0"
s2_performance_1_file = data_folder + r"\s2_performance_across_K_1"
s2_performance_2_file = data_folder + r"\s2_performance_across_K_2"
s2_performance_3_file = data_folder + r"\s2_performance_across_K_3"
with open(s2_performance_0_file, 'rb') as infile:
    s2_performance_0 = pickle.load(infile)
with open(s2_performance_1_file, 'rb') as infile:
    s2_performance_1 = pickle.load(infile)
with open(s2_performance_2_file, 'rb') as infile:
    s2_performance_2 = pickle.load(infile)
with open(s2_performance_3_file, 'rb') as infile:
    s2_performance_3 = pickle.load(infile)

# Performance
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# S1
plt.plot(x, s1_performance_0, "g-", label="Zero")
plt.plot(x, s1_performance_1, "k:", label="Low")
plt.plot(x, s1_performance_2, "k--", label="Middle")
plt.plot(x, s1_performance_3, "k-", label="High")
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.title("Coordination_SS_S1")
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\Coordination_SS_S1.png", transparent=False, dpi=200)
plt.show()
plt.clf()


# S2
plt.plot(x, s2_performance_0, "g-", label="Zero")
plt.plot(x, s2_performance_1, "k:", label="Low")
plt.plot(x, s2_performance_2, "k--", label="Middle")
plt.plot(x, s2_performance_3, "k-", label="High")
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.title("Coordination_SS_S2")
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\Coordination_SS_S2.png", transparent=False, dpi=200)
plt.show()
plt.clf()
print("END")

# Testing the iteration boundary
# data_folder = r"E:\data\gst-1018\outbound_without_map_jump_3"
# g_original_performance_file = data_folder + r"\g_original_performance_data_across_K"
# s_original_performance_file = data_folder + r"\s_original_performance_data_across_K"
# t_original_performance_file = data_folder + r"\t_original_performance_data_across_K"
# with open(g_original_performance_file, 'rb') as infile:
#     g_original_performance = pickle.load(infile)
# with open(s_original_performance_file, 'rb') as infile:
#     s_original_performance = pickle.load(infile)
# with open(t_original_performance_file, 'rb') as infile:
#     t_original_performance = pickle.load(infile)
#
#
# average_across_iteration_across_k = []
# for row in range(len(s_original_performance)):
#     average_across_iteration = []
#     one_k_performance = s_original_performance[row]
#     for index in range(len(one_k_performance)):
#         if (index+1) % 100 == 0:
#             temp_average = sum(one_k_performance[:index]) / (index + 1)
#             average_across_iteration.append(temp_average)
#     average_across_iteration_across_k.append(average_across_iteration)
#
# x = np.arange(100, 5001, 100)
# color_list = ["r-", "b-", "y-", "g-", "k-", "k--", "k:", "r--", "b--", "g--"]
# for index, data in enumerate(average_across_iteration_across_k):
#     plt.plot(x, data, color_list[index], label="K={0}".format(index))
# # plt.title('Diversity Decrease')
# plt.xlabel('Repetition', fontweight='bold', fontsize=10)
# plt.ylabel('Performance', fontweight='bold', fontsize=10)
# # plt.xticks(x)
# plt.legend(frameon=False, ncol=3, fontsize=10)
# plt.savefig(data_folder + r"\S_performance_across_repetition.png", transparent=False, dpi=200)
# plt.clf()
# plt.show()