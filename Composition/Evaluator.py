# -*- coding: utf-8 -*-
# @Time     : 10/3/2022 22:31
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle

# data_folder = r"E:\data\gst-1003\Composition"
# g_performance_file = data_folder + r"\g_performance_across_K"
# s_performance_file = data_folder + r"\s_performance_across_K"
# t_performance_file = data_folder + r"\t_performance_across_K"
# with open(g_performance_file, 'rb') as infile:
#     g_performance = pickle.load(infile)
# with open(s_performance_file, 'rb') as infile:
#     s_performance = pickle.load(infile)
# with open(t_performance_file, 'rb') as infile:
#     t_performance = pickle.load(infile)
#
# # print(g_performance)
# x = range(len(g_performance))
# plt.plot(x, g_performance, "r-", label="G")
# plt.plot(x, s_performance, "b-", label="S")
# plt.plot(x, t_performance, "g-", label="T")
# # plt.title('Diversity Decrease')
# plt.xlabel('K', fontweight='bold', fontsize=10)
# plt.ylabel('Performance', fontweight='bold', fontsize=10)
# plt.legend(frameon=False, ncol=3, fontsize=10)
# plt.savefig("GST_performance_K.png", transparent=True, dpi=1200)
# plt.show()


data_folder = r"E:\data\gst-1003\Composition"
g_performance_file = data_folder + r"\g_jump_across_K"
s_performance_file = data_folder + r"\s_jump_across_K"
t_performance_file = data_folder + r"\t_jump_across_K"
with open(g_performance_file, 'rb') as infile:
    g_performance = pickle.load(infile)
with open(s_performance_file, 'rb') as infile:
    s_performance = pickle.load(infile)
with open(t_performance_file, 'rb') as infile:
    t_performance = pickle.load(infile)

# print(g_performance)
x = range(len(g_performance))
plt.plot(x, g_performance, "r-", label="G")
plt.plot(x, s_performance, "b-", label="S")
plt.plot(x, t_performance, "g-", label="T")
# plt.title('Diversity Decrease')
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Jump', fontweight='bold', fontsize=10)
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig("GST_performance_K.png", transparent=True, dpi=1200)
plt.show()