# -*- coding: utf-8 -*-
# @Time     : 10/3/2022 22:31
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle

data_folder = r"E:\data\gst-1003\inbound_divergence"
g_performance_file = data_folder + r"\g_performance_across_divergence"
s_performance_file = data_folder + r"\s_performance_across_divergence"
t_performance_file = data_folder + r"\t_performance_across_divergence"
with open(g_performance_file, 'rb') as infile:
    g_performance = pickle.load(infile)
with open(s_performance_file, 'rb') as infile:
    s_performance = pickle.load(infile)
with open(t_performance_file, 'rb') as infile:
    t_performance = pickle.load(infile)

g_learn_file = data_folder + r"\g_learn_across_divergence"
s_learn_file = data_folder + r"\s_learn_across_divergence"
t_learn_file = data_folder + r"\t_learn_across_divergence"
with open(g_learn_file, 'rb') as infile:
    g_jump = pickle.load(infile)
with open(s_learn_file, 'rb') as infile:
    s_jump = pickle.load(infile)
with open(t_learn_file, 'rb') as infile:
    t_jump = pickle.load(infile)

g_deviation_file = data_folder + r"\g_deviation_across_divergence"
s_deviation_file = data_folder + r"\s_deviation_across_divergence"
t_deviation_file = data_folder + r"\t_deviation_across_divergence"
with open(g_deviation_file, 'rb') as infile:
    g_deviation = pickle.load(infile)
with open(s_deviation_file, 'rb') as infile:
    s_deviation = pickle.load(infile)
with open(t_deviation_file, 'rb') as infile:
    t_deviation = pickle.load(infile)

# Performance
x = [1, 2, 3, 4, 5, 6]
plt.plot(x, g_performance, "r-", label="G")
plt.plot(x, s_performance, "b-", label="S")
plt.plot(x, t_performance, "g-", label="T")
# plt.title('Diversity Decrease')
plt.xlabel('Divergence', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\GST_performance_divergence.png", transparent=True, dpi=1200)
plt.clf()

# Learn
plt.plot(x, g_jump, "r-", label="G")
plt.plot(x, s_jump, "b-", label="S")
plt.plot(x, t_jump, "g-", label="T")
# plt.title('Diversity Decrease')
plt.xlabel('Divergence', fontweight='bold', fontsize=10)
plt.ylabel('Learn', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\GST_jump_divergence.png", transparent=True, dpi=1200)
plt.clf()

# Deviation
plt.plot(x, g_deviation, "r-", label="G")
plt.plot(x, s_deviation, "b-", label="S")
plt.plot(x, t_deviation, "g-", label="T")
# plt.title('Diversity Decrease')
plt.xlabel('Divergence', fontweight='bold', fontsize=10)
plt.ylabel('Deviation', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\GST_deviation_divergence.png", transparent=True, dpi=1200)
plt.clf()
# plt.show()
print("END")