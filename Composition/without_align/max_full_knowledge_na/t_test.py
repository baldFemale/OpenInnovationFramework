# -*- coding: utf-8 -*-
# @Time     : 11/5/2022 17:02
# @Author   : Junyi
# @FileName: t_test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
# two sample t-test
from scipy import stats
import pickle

data_folder = r"E:\data\gst-1102\max_normalized_3"
g_original_cog_performance_file = data_folder + r"\g_original_cog_performance_data_across_K"
g_original_performance_file = data_folder + r"\g_original_performance_data_across_K"
s_original_cog_performance_file = data_folder + r"\s_original_cog_performance_data_across_K"
s_original_performance_file = data_folder + r"\s_original_performance_data_across_K"
t_original_cog_performance_file = data_folder + r"\t_original_cog_performance_data_across_K"
t_original_performance_file = data_folder + r"\t_original_performance_data_across_K"

with open(g_original_performance_file, 'rb') as infile:
    g_original_performance = pickle.load(infile)
with open(g_original_cog_performance_file, 'rb') as infile:
    g_original_cog_performance = pickle.load(infile)
with open(s_original_performance_file, 'rb') as infile:
    s_original_performance = pickle.load(infile)
with open(s_original_cog_performance_file, 'rb') as infile:
    s_original_cog_performance = pickle.load(infile)
with open(t_original_performance_file, 'rb') as infile:
    t_original_performance = pickle.load(infile)
with open(t_original_cog_performance_file, 'rb') as infile:
    t_original_cog_performance = pickle.load(infile)


K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
print("========G vs. S=========")
for index, K in enumerate(K_list):
    list_a = g_original_performance[index]
    list_b = s_original_performance[index]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))

print("========G vs. T=========")
for index, K in enumerate(K_list):
    list_a = g_original_performance[index]
    list_b = t_original_performance[index]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))

print("========S vs. T=========")
for index, K in enumerate(K_list):
    list_a = s_original_performance[index]
    list_b = t_original_performance[index]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))


print("========G across K=========")
for index, K in enumerate(K_list):
    if index == len(K_list) - 1:
        break
    list_a = g_original_performance[index]
    list_b = g_original_performance[index+1]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))

print("========S across K=========")
for index, K in enumerate(K_list):
    if index == len(K_list) - 1:
        break
    list_a = s_original_performance[index]
    list_b = s_original_performance[index+1]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))

print("========T across K=========")
for index, K in enumerate(K_list):
    if index == len(K_list) - 1:
        break
    list_a = t_original_performance[index]
    list_b = t_original_performance[index+1]
    t_value, p_value = stats.ttest_ind(list_a, list_b)
    # print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    # print('p-value for two tailed test is %f' % p_value)
    alpha = 0.05
    if p_value <= alpha:
        print("K={0}: {1} Sig. p-value={2}".format(K, sum(list_a) / len(list_a) - sum(list_b) / len(list_b), p_value))
    else:
        print("K={0}: {1} Not Sig.".format(K, sum(list_a) / len(list_a) - sum(list_b / len(list_b)), p_value))