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

data_folder = r"E:\data\gst-1102\max_normalized"
g_performance_file = data_folder + r"\g_performance_across_K"
# s_performance_file = data_folder + r"\s_performance_across_K"
# t_performance_file = data_folder + r"\t_performance_across_K"
with open(g_performance_file, 'rb') as infile:
    g_performance = pickle.load(infile)
# with open(s_performance_file, 'rb') as infile:
#     s_performance = pickle.load(infile)
# with open(t_performance_file, 'rb') as infile:
#     t_performance = pickle.load(infile)


g_deviation_file = data_folder + r"\g_deviation_across_K"
# s_deviation_file = data_folder + r"\s_deviation_across_K"
# t_deviation_file = data_folder + r"\t_deviation_across_K"
with open(g_deviation_file, 'rb') as infile:
    g_deviation = pickle.load(infile)
# with open(s_deviation_file, 'rb') as infile:
#     s_deviation = pickle.load(infile)
# with open(t_deviation_file, 'rb') as infile:
#     t_deviation = pickle.load(infile)

# Performance
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Error bar figure

fig, (ax1) = plt.subplots(1, 1)
ax1.errorbar(range(len(g_performance)), g_performance, yerr=g_deviation, color="g", fmt="-", capsize=5, capthick=0.8, ecolor="g", label="G")
# ax1.errorbar(x, s_performance, yerr=s_deviation, color="b", fmt="-", capsize=5, capthick=0.8, ecolor="b", label="S")
# ax1.errorbar(x, t_performance, yerr=t_deviation, color="r", fmt="-", capsize=5, capthick=0.8, ecolor="r", label="T")
plt.xlabel('Complexity', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
# plt.xticks(x)
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
plt.legend(handles, labels, numpoints=1, frameon=False)
plt.savefig(data_folder + r"\GST_performance_K.png", transparent=False, dpi=200)
plt.show()


# two sample t-test
# from scipy import stats
# t_result = stats.ttest_ind(s_performance, t_performance, equal_var=False)
# print(t_result)
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