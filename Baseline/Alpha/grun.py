# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None, alpha=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=alpha)
    performance_across_agent_time = []
    cog_performance_across_agent_time = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num,
                           generalist_expertise=expertise_amount)
        for _ in range(search_iteration):
            generalist.search()
        performance_across_agent_time.append(generalist.fitness_across_time)
        cog_performance_across_agent_time.append(generalist.cog_fitness_across_time)

    performance_across_time = []
    cog_performance_across_time = []
    variance_across_time = []
    for period in range(search_iteration):
        temp_1 = [performance_list[period] for performance_list in performance_across_agent_time]
        temp_2 = [performance_list[period] for performance_list in cog_performance_across_agent_time]
        performance_across_time.append(sum(temp_1) / len(temp_1))
        cog_performance_across_time.append(sum(temp_2) / len(temp_2))
        variance_across_time.append(np.std(temp_1))
    return_dict[loop] = [performance_across_time, cog_performance_across_time, variance_across_time]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 300
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    expertise_amount = 18
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    alpha_list = [0.1, 0.2, 0.3, 0.4]
    concurrency = 50
    for alpha in alpha_list:
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
                p = mp.Process(target=func, args=(N, K, state_num, expertise_amount, agent_num, alpha,
                                                  search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_fitness_time, temp_cog_time, temp_var_time = [], [], []
            temp_fitness, temp_cog, temp_var = [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_fitness_time.append(result[0])
                temp_cog_time.append(result[1])
                temp_var_time.append(result[2])

                temp_fitness.append(result[0][-1])
                temp_cog.append(result[1][-1])
                temp_var.append(result[2][-1])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            variance_across_K.append(sum(temp_var) / len(temp_var))

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
            performance_across_K_time.append(performance_across_time)
            cog_performance_across_K_time.append(cog_performance_across_time)
            variance_across_K_time.append(variance_across_time)
        # remove time dimension
        with open("g_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("g_variance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        # retain time dimension
        with open("g_performance_across_K_time_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(performance_across_K_time, out_file)
        with open("g_cog_performance_across_K_time_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(cog_performance_across_K_time, out_file)
        with open("g_variance_across_K_time_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(variance_across_K_time, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


