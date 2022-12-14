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

data_folder = r"F:\data\gst-1112\Priority\sg"
g_performance_0_file = data_folder + r"\g_performance_across_K_0"
g_performance_1_file = data_folder + r"\g_performance_across_K_1"
g_performance_2_file = data_folder + r"\g_performance_across_K_2"
g_performance_3_file = data_folder + r"\g_performance_across_K_3"
with open(g_performance_0_file, 'rb') as infile:
    g_performance_0 = pickle.load(infile)
with open(g_performance_1_file, 'rb') as infile:
    g_performance_1 = pickle.load(infile)
with open(g_performance_2_file, 'rb') as infile:
    g_performance_2 = pickle.load(infile)
with open(g_performance_3_file, 'rb') as infile:
    g_performance_3 = pickle.load(infile)

s_performance_0_file = data_folder + r"\s_performance_across_K_0"
s_performance_1_file = data_folder + r"\s_performance_across_K_1"
s_performance_2_file = data_folder + r"\s_performance_across_K_2"
s_performance_3_file = data_folder + r"\s_performance_across_K_3"
with open(s_performance_0_file, 'rb') as infile:
    s_performance_0 = pickle.load(infile)
with open(s_performance_1_file, 'rb') as infile:
    s_performance_1 = pickle.load(infile)
with open(s_performance_2_file, 'rb') as infile:
    s_performance_2 = pickle.load(infile)
with open(s_performance_3_file, 'rb') as infile:
    s_performance_3 = pickle.load(infile)

# Performance
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# G
plt.plot(x, g_performance_1, "g-", label="Zero")
plt.plot(x, g_performance_1, "k--", label="Low")
plt.plot(x, g_performance_2, "k:", label="Middle")
plt.plot(x, g_performance_3, "k-", label="High")
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.title("GS Priority across Overlap")
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\G_Performance_K.png", transparent=False, dpi=200)
plt.show()
plt.clf()


# G2
plt.plot(x, s_performance_0, "g-", label="Zero")
plt.plot(x, s_performance_1, "k--", label="Low")
plt.plot(x, s_performance_2, "k:", label="Middle")
plt.plot(x, s_performance_3, "k-", label="High")
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.title("GS Priority across Overlap")
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\S_Performance_K.png", transparent=False, dpi=200)
plt.show()
plt.clf()
print("END")


# two sample t-test
# from scipy import stats
# t_result = stats.ttest_ind(s_performance, t_performance, equal_var=False)
# print(t_result)
# print("END")



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