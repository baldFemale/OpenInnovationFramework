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


# data_folder = r"E:\data\gst-1102\max_min_2"
# g_performance_file = data_folder + r"\g_performance_across_K"
# g_cog_performance_file = data_folder + r"\g_cog_performance_across_K"
# g_potential_performance_file = data_folder + r"\g_cog_performance_across_K"
# g_original_cog_performance_file = data_folder + r"\g_original_cog_performance_data_across_K"
# g_original_performance_file = data_folder + r"\g_original_performance_data_across_K"
# with open(g_performance_file, 'rb') as infile:
#     g_performance = pickle.load(infile)
# with open(g_cog_performance_file, 'rb') as infile:
#     g_cog_performance = pickle.load(infile)
# with open(g_potential_performance_file, 'rb') as infile:
#     g_potential_performance = pickle.load(infile)
# with open(g_original_performance_file, 'rb') as infile:
#     g_original_performance = pickle.load(infile)
# with open(g_original_cog_performance_file, 'rb') as infile:
#     g_original_cog_performance = pickle.load(infile)
#
# s_performance_file = data_folder + r"\s_performance_across_K"
# s_cog_performance_file = data_folder + r"\s_cog_performance_across_K"
# s_potential_performance_file = data_folder + r"\s_cog_performance_across_K"
# s_original_cog_performance_file = data_folder + r"\s_original_cog_performance_data_across_K"
# s_original_performance_file = data_folder + r"\s_original_performance_data_across_K"
# with open(s_performance_file, 'rb') as infile:
#     s_performance = pickle.load(infile)
# with open(s_cog_performance_file, 'rb') as infile:
#     s_cog_performance = pickle.load(infile)
# with open(s_potential_performance_file, 'rb') as infile:
#     s_potential_performance = pickle.load(infile)
# with open(s_original_performance_file, 'rb') as infile:
#     s_original_performance = pickle.load(infile)
# with open(s_original_cog_performance_file, 'rb') as infile:
#     s_original_cog_performance = pickle.load(infile)
#
# t_performance_file = data_folder + r"\t_performance_across_K"
# t_cog_performance_file = data_folder + r"\t_cog_performance_across_K"
# t_potential_performance_file = data_folder + r"\t_cog_performance_across_K"
# t_original_cog_performance_file = data_folder + r"\t_original_cog_performance_data_across_K"
# t_original_performance_file = data_folder + r"\t_original_performance_data_across_K"
# with open(t_performance_file, 'rb') as infile:
#     t_performance = pickle.load(infile)
# with open(t_cog_performance_file, 'rb') as infile:
#     t_cog_performance = pickle.load(infile)
# with open(t_potential_performance_file, 'rb') as infile:
#     t_potential_performance = pickle.load(infile)
# with open(t_original_performance_file, 'rb') as infile:
#     t_original_performance = pickle.load(infile)
# with open(t_original_cog_performance_file, 'rb') as infile:
#     t_original_cog_performance = pickle.load(infile)
#
# g_deviation_file = data_folder + r"\g_deviation_across_K"
# s_deviation_file = data_folder + r"\s_deviation_across_K"
# t_deviation_file = data_folder + r"\t_deviation_across_K"
# with open(g_deviation_file, 'rb') as infile:
#     g_deviation = pickle.load(infile)
# with open(s_deviation_file, 'rb') as infile:
#     s_deviation = pickle.load(infile)
# with open(t_deviation_file, 'rb') as infile:
#     t_deviation = pickle.load(infile)
#
# # Performance
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#
# # Error bar figure
# fig, (ax1) = plt.subplots(1, 1)
# ax1.errorbar(x, g_performance, yerr=g_deviation, color="g", fmt="-", capsize=5, capthick=0.8, ecolor="g", label="G")
# ax1.errorbar(x, s_performance, yerr=s_deviation, color="b", fmt="-", capsize=5, capthick=0.8, ecolor="b", label="S")
# ax1.errorbar(x, t_performance, yerr=t_deviation, color="r", fmt="-", capsize=5, capthick=0.8, ecolor="r", label="T")
# plt.xlabel('Complexity', fontweight='bold', fontsize=10)
# plt.ylabel('Performance', fontweight='bold', fontsize=10)
# # plt.xticks(x)
# handles, labels = ax1.get_legend_handles_labels()
# handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
# plt.legend(handles, labels, numpoints=1, frameon=False)
# plt.savefig(data_folder + r"\GST_performance_K.png", transparent=False, dpi=200)
# plt.show()
# plt.clf()
#
# plt.plot(x, g_performance, "r-", label="G")
# plt.plot(x, g_cog_performance, "r--", label="Gc")
# plt.plot(x, s_performance, "g-", label="S")
# plt.plot(x, s_cog_performance, "g--", label="Sc")
# plt.plot(x, t_performance, "b-", label="T")
# plt.plot(x, t_cog_performance, "b--", label="Tc")
# plt.xlabel('K', fontweight='bold', fontsize=10)
# plt.ylabel('Performance', fontweight='bold', fontsize=10)
# plt.xticks(x)
# plt.legend()
# plt.savefig(data_folder + r"\GST_Cognition_K.png", transparent=False, dpi=200)
# plt.clf()


# Testing the iteration boundary
data_folder = r"E:\data\gst-1102\max_min_2"
g_original_performance_file = data_folder + r"\g_original_performance_data_across_K"
s_original_performance_file = data_folder + r"\s_original_performance_data_across_K"
t_original_performance_file = data_folder + r"\t_original_performance_data_across_K"
with open(g_original_performance_file, 'rb') as infile:
    g_original_performance = pickle.load(infile)
with open(s_original_performance_file, 'rb') as infile:
    s_original_performance = pickle.load(infile)
with open(t_original_performance_file, 'rb') as infile:
    t_original_performance = pickle.load(infile)


g_0 = g_original_performance[0]
s_0 = s_original_performance[0]
t_0 = t_original_performance[0]
# print(g_0)
print(g_0[0], g_0[399], g_0[799], g_0[1199])
print(s_0[0], s_0[399], s_0[799], s_0[1199])
print(s_0[1], s_0[400], s_0[800], s_0[1200])
print(t_0[0], t_0[399], t_0[799], t_0[1199])
# x = np.arange(100, 1001, 100)
# for index, data in enumerate(average_across_iteration_across_k):
#     plt.plot(x, data, label="K={0}".format(index))
# # plt.title('Diversity Decrease')
# plt.xlabel('Repetition', fontweight='bold', fontsize=10)
# plt.ylabel('Performance', fontweight='bold', fontsize=10)
# # plt.xticks(x)
# plt.legend(frameon=False, ncol=3, fontsize=10)
# plt.savefig(data_folder + r"\S_performance_across_repetition.png", transparent=False, dpi=200)
# plt.clf()
# plt.show()
