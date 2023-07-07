# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle
import statistics


# mp version
def func(N=None, K=None, state_num=None, generalist_expertise=None, specialist_expertise=None, agent_num=None,
         search_iteration=None, roll_back=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num)
    performance_across_agent_time = []
    cog_performance_across_agent_time = []
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0)
    for _ in range(agent_num):
        specialist = Agent(N=N, landscape=landscape, state_num=state_num, crowd=crowd,
                           generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
        for _ in range(search_iteration):
            specialist.feedback_search(roll_back_ratio=0, roll_forward_ratio=roll_back)
        performance_across_agent_time.append(specialist.fitness_across_time)
        cog_performance_across_agent_time.append(specialist.cog_fitness_across_time)

    performance_across_time = []
    cog_performance_across_time = []
    variance_across_time = []
    first_quantile_across_time = []
    last_quantile_across_time = []
    for period in range(search_iteration):
        temp_1 = [performance_list[period] for performance_list in performance_across_agent_time]
        temp_2 = [performance_list[period] for performance_list in cog_performance_across_agent_time]
        performance_across_time.append(sum(temp_1) / len(temp_1))

        # Measure the quantiles
        quantiles = statistics.quantiles(temp_1, n=4)
        first_quantile = quantiles[0]
        last_quantile = quantiles[-1]
        above_first_quantile = [num for num in temp_1 if num >= first_quantile]
        first_quantile_across_time.append(sum(above_first_quantile) / len(above_first_quantile))
        below_last_quantile = [num for num in temp_1 if num <= last_quantile]
        last_quantile_across_time.append(sum(below_last_quantile) / len(below_last_quantile))

        cog_performance_across_time.append(sum(temp_2) / len(temp_2))
        variance_across_time.append(np.std(temp_1))
    return_dict[loop] = [performance_across_time, cog_performance_across_time,
                         variance_across_time, first_quantile_across_time, last_quantile_across_time]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 50
    search_iteration = 400
    N = 9
    state_num = 4
    generalist_expertise = 0
    specialist_expertise = 12
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    roll_back_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    concurrency = 50

    for roll_back in roll_back_list:
        # DVs
        performance_across_K = []
        variance_across_K = []
        first_quantile_across_K = []
        lats_quantile_across_K = []

        performance_across_K_time = []
        cog_performance_across_K_time = []
        variance_across_K_time = []
        first_quantile_across_K_time = []
        last_quantile_across_K_time = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, generalist_expertise, specialist_expertise,
                                                  agent_num, search_iteration, roll_back, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_fitness_time, temp_cog_time, temp_var_time, \
                temp_first_time, temp_last_time = [], [], [], [], []
            temp_fitness, temp_cog, temp_var, \
                temp_first, temp_last = [], [], [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_fitness_time.append(result[0])
                temp_cog_time.append(result[1])
                temp_var_time.append(result[2])
                temp_first_time.append(result[3])
                temp_last_time.append(result[4])

                temp_fitness.append(result[0][-1])
                temp_cog.append(result[1][-1])
                temp_var.append(result[2][-1])
                temp_first.append(result[3][-1])
                temp_last.append(result[4][-1])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            variance_across_K.append(sum(temp_var) / len(temp_var))
            first_quantile_across_K.append(sum(temp_first) / len(temp_first))
            lats_quantile_across_K.append(sum(temp_last) / len(temp_last))

            performance_across_time = []
            cog_performance_across_time = []
            variance_across_time = []
            first_quantile_across_time = []
            last_quantile_across_time = []
            for period in range(search_iteration):
                temp_1 = [performance_list[period] for performance_list in temp_fitness_time]
                performance_across_time.append(sum(temp_1) / len(temp_1))
                temp_2 = [performance_list[period] for performance_list in temp_cog_time]
                cog_performance_across_time.append(sum(temp_2) / len(temp_2))
                temp_3 = [performance_list[period] for performance_list in temp_var_time]
                variance_across_time.append(sum(temp_3) / len(temp_3))
                temp_4 = [performance_list[period] for performance_list in temp_first_time]
                first_quantile_across_time.append((sum(temp_4) / len(temp_4)))
                temp_5 = [performance_list[period] for performance_list in temp_last_time]
                last_quantile_across_time.append((sum(temp_5) / len(temp_5)))
            performance_across_K_time.append(performance_across_time)
            cog_performance_across_K_time.append(cog_performance_across_time)
            variance_across_K_time.append(variance_across_time)
            first_quantile_across_K_time.append(first_quantile_across_time)
            last_quantile_across_K_time.append(last_quantile_across_time)
        # remove time dimension
        with open("gs_performance_across_K_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("gs_variance_across_K_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("gs_first_quantile_across_K_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(first_quantile_across_K, out_file)
        with open("gs_last_quantile_across_K_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(lats_quantile_across_K, out_file)
        # retain time dimension
        with open("gs_performance_across_K_time_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(performance_across_K_time, out_file)
        with open("gs_cog_performance_across_K_time_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(cog_performance_across_K_time, out_file)
        with open("gs_variance_across_K_time_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(variance_across_K_time, out_file)
        with open("gs_first_quantile_across_K_time_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(first_quantile_across_K_time, out_file)
        with open("gs_last_quantile_across_K_time_beta_{0}".format(roll_back), 'wb') as out_file:
            pickle.dump(last_quantile_across_K_time, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


