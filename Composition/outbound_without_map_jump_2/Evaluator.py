# -*- coding: utf-8 -*-
# @Time     : 10/3/2022 22:31
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle
from matplotlib import container

data_folder = r"E:\data\gst-1018\outbound_without_map_jump_2"
g_performance_file = data_folder + r"\g_performance_across_K"
s_performance_file = data_folder + r"\s_performance_across_K"
t_performance_file = data_folder + r"\t_performance_across_K"
with open(g_performance_file, 'rb') as infile:
    g_performance = pickle.load(infile)
with open(s_performance_file, 'rb') as infile:
    s_performance = pickle.load(infile)
with open(t_performance_file, 'rb') as infile:
    t_performance = pickle.load(infile)


g_deviation_file = data_folder + r"\g_deviation_across_K"
s_deviation_file = data_folder + r"\s_deviation_across_K"
t_deviation_file = data_folder + r"\t_deviation_across_K"
with open(g_deviation_file, 'rb') as infile:
    g_deviation = pickle.load(infile)
with open(s_deviation_file, 'rb') as infile:
    s_deviation = pickle.load(infile)
with open(t_deviation_file, 'rb') as infile:
    t_deviation = pickle.load(infile)

# Performance
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Deviation
plt.plot(x, g_deviation, "r-", label="G")
plt.plot(x, s_deviation, "b-", label="S")
plt.plot(x, t_deviation, "g-", label="T")
# plt.title('Diversity Decrease')
plt.xlabel('K', fontweight='bold', fontsize=10)
plt.ylabel('Deviation', fontweight='bold', fontsize=10)
plt.xticks(x)
plt.legend(frameon=False, ncol=3, fontsize=10)
plt.savefig(data_folder + r"\GST_deviation_K.png", transparent=False, dpi=200)
plt.clf()
# plt.show()

# Error bar figure

fig, (ax1) = plt.subplots(1, 1)
ax1.errorbar(x, g_performance, yerr=g_deviation, color="g", fmt="-", capsize=5, capthick=0.8, ecolor="g", label="G")
ax1.errorbar(x, s_performance, yerr=s_deviation, color="b", fmt="-", capsize=5, capthick=0.8, ecolor="b", label="S")
ax1.errorbar(x, t_performance, yerr=t_deviation, color="r", fmt="-", capsize=5, capthick=0.8, ecolor="r", label="T")
plt.xlabel('Complexity', fontweight='bold', fontsize=10)
plt.ylabel('Performance', fontweight='bold', fontsize=10)
plt.xticks(x)
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
plt.legend(handles, labels, numpoints=1, frameon=False)
plt.savefig(data_folder + r"\GST_performance_K.png", transparent=False, dpi=200)
plt.show()


# two sample t-test
from scipy import stats
t_result = stats.ttest_ind(s_performance, t_performance, equal_var=False)
print(t_result)
print("END")