# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Tshape import Tshape
from Landscape import Landscape
from CogLandscape import CogLandscape
import multiprocessing as mp
import time
from multiprocessing import Pool
from multiprocessing import Semaphore
import pickle
import math


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num)
    ave_performance_across_agent_time = []
    max_performance_across_agent_time = []
    min_performance_across_agent_time = []
    cog_performance_across_agent_time = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        cog_landscape = CogLandscape(landscape=landscape, expertise_domain=specialist.expertise_domain,
                                     expertise_representation=specialist.expertise_representation)
        specialist.cog_landscape = cog_landscape
        specialist.update_cog_fitness()

        ave_performance_one_run = []
        max_performance_one_run = []
        min_performance_one_run = []
        cog_performance_one_run = []
        for _ in range(search_iteration):
            specialist.search()
            ave_performance_one_run.append(specialist.ave_fitness)
            max_performance_one_run.append(specialist.max_fitness)
            min_performance_one_run.append(specialist.min_fitness)
            cog_performance_one_run.append(specialist.cog_fitness)
        ave_performance_across_agent_time.append(ave_performance_one_run)
        max_performance_across_agent_time.append(max_performance_one_run)
        min_performance_across_agent_time.append(min_performance_one_run)
        cog_performance_across_agent_time.append(cog_performance_one_run)

    ave_performance_across_time = []
    max_performance_across_time = []
    min_performance_across_time = []
    cog_performance_across_time = []
    performance_variance_across_time = []
    first_quantile_across_time = []
    last_quantile_across_time = []
    cog_variance_across_time = []
    for period in range(search_iteration):
        temp_1 = [performance_list[period] for performance_list in ave_performance_across_agent_time]
        ave_performance_across_time.append(sum(temp_1) / len(temp_1))
        performance_variance_across_time.append(np.std(temp_1))
        first_quantile_across_time.append(np.percentile(temp_1, 25))
        last_quantile_across_time.append(np.percentile(temp_1, 75))

        temp_2 = [performance_list[period] for performance_list in max_performance_across_agent_time]
        max_performance_across_time.append(sum(temp_2) / len(temp_2))

        temp_3 = [performance_list[period] for performance_list in min_performance_across_agent_time]
        min_performance_across_time.append(sum(temp_3) / len(temp_3))

        temp_4 = [performance_list[period] for performance_list in cog_performance_across_agent_time]
        cog_performance_across_time.append(sum(temp_4) / len(temp_4))
        cog_variance_across_time.append(np.std(temp_4))

    return_dict[loop] = [ave_performance_across_time, max_performance_across_time, min_performance_across_time,
                         cog_performance_across_time, performance_variance_across_time, cog_variance_across_time,
                         first_quantile_across_time, last_quantile_across_time]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 50
    agent_num = 100
    search_iteration = 200  # In pre-test, 200 is quite enough for convergence
    N = 9
    state_num = 4
    expertise_amount = 36
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    ave_performance_across_K = []
    max_performance_across_K = []
    min_performance_across_K = []
    cog_performance_across_K = []
    first_quantile_across_K = []
    last_quantile_across_K = []
    variance_across_K = []
    cog_variance_across_K = []

    ave_performance_across_K_time = []
    max_performance_across_K_time = []
    min_performance_across_K_time = []
    cog_performance_across_K_time = []
    first_quantile_across_K_time = []
    last_quantile_across_K_time = []
    variance_across_K_time = []
    cog_variance_across_K_time = []

    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        returns = return_dict.values()  # Don't need dict index, since it is repetition.

        temp_ave, temp_max, temp_min, temp_cog = [], [], [], []
        temp_var, temp_cog_var = [], []
        temp_first, temp_last = [], []
        for result in returns:
            temp_ave.append(result[0])   # ave among the alternatives (full, Landscape)
            temp_max.append(result[1])  # max in the alternatives (full, Landscape)
            temp_min.append(result[2])  # min in the alternatives (full, Landscape)
            temp_cog.append(result[3])  # cog_performance (partial, CogLandscape)
            temp_var.append(result[4])  # variance among the alternatives (full, Landscape)
            temp_cog_var.append(result[5])  # cog_variance (partial, CogLandscape)
            temp_first.append(result[6])  # first quantile in the alternatives (full, Landscape)
            temp_last.append(result[7])  # third quantile in the alternatives (full, Landscape)

        result_ave, result_max, result_min, result_cog = [], [], [], []
        result_var, result_cog_var = [], []
        result_first, result_last = [], []
        for period in range(search_iteration):
            temp_1 = [indicator_list[period] for indicator_list in temp_ave]
            temp_2 = [indicator_list[period] for indicator_list in temp_max]
            temp_3 = [indicator_list[period] for indicator_list in temp_min]
            temp_4 = [indicator_list[period] for indicator_list in temp_cog]
            temp_5 = [indicator_list[period] for indicator_list in temp_var]
            temp_6 = [indicator_list[period] for indicator_list in temp_cog_var]
            temp_7 = [indicator_list[period] for indicator_list in temp_first]
            temp_8 = [indicator_list[period] for indicator_list in temp_last]

            result_ave.append(sum(temp_1) / len(temp_1))
            result_max.append(sum(temp_2) / len(temp_2))
            result_min.append(sum(temp_3) / len(temp_3))
            result_cog.append(sum(temp_4) / len(temp_4))
            result_var.append(sum(temp_5) / len(temp_5))
            result_cog_var.append(sum(temp_6) / len(temp_6))
            result_first.append(sum(temp_7) / len(temp_7))
            result_last.append(sum(temp_8) / len(temp_8))

        ave_performance_across_K.append(result_ave[-1])
        max_performance_across_K.append(result_max[-1])
        min_performance_across_K.append(result_min[-1])
        cog_performance_across_K.append(result_cog[-1])
        variance_across_K.append(result_var[-1])
        cog_variance_across_K.append(result_cog_var[-1])
        first_quantile_across_K.append(result_first[-1])
        last_quantile_across_K.append(result_last[-1])

        ave_performance_across_K_time.append(result_ave)
        max_performance_across_K_time.append(result_max)
        min_performance_across_K_time.append(result_min)
        cog_performance_across_K_time.append(result_cog)
        variance_across_K_time.append(result_var)
        cog_variance_across_K_time.append(result_cog_var)
        first_quantile_across_K_time.append(result_first)
        last_quantile_across_K_time.append(result_last)

    with open("s_ave_performance_across_K", 'wb') as out_file:
        pickle.dump(ave_performance_across_K, out_file)
    with open("s_max_performance_across_K", 'wb') as out_file:
        pickle.dump(max_performance_across_K, out_file)
    with open("s_min_performance_across_K", 'wb') as out_file:
        pickle.dump(min_performance_across_K, out_file)
    with open("s_cog_performance_across_K", 'wb') as out_file:
        pickle.dump(cog_performance_across_K, out_file)
    with open("s_variance_across_K", 'wb') as out_file:
        pickle.dump(variance_across_K, out_file)
    with open("s_cog_variance_across_K", 'wb') as out_file:
        pickle.dump(cog_variance_across_K, out_file)
    with open("s_first_quantile_across_K", 'wb') as out_file:
        pickle.dump(first_quantile_across_K, out_file)
    with open("s_last_quantile_across_K", 'wb') as out_file:
        pickle.dump(last_quantile_across_K, out_file)

    with open("s_ave_performance_across_K_time", 'wb') as out_file:
        pickle.dump(ave_performance_across_K_time, out_file)
    with open("s_max_performance_across_K_time", 'wb') as out_file:
        pickle.dump(max_performance_across_K_time, out_file)
    with open("s_min_performance_across_K_time", 'wb') as out_file:
        pickle.dump(min_performance_across_K_time, out_file)
    with open("s_cog_performance_across_K_time", 'wb') as out_file:
        pickle.dump(cog_performance_across_K_time, out_file)
    with open("s_variance_across_K_time", 'wb') as out_file:
        pickle.dump(variance_across_K_time, out_file)
    with open("s_cog_variance_across_K_time", 'wb') as out_file:
        pickle.dump(cog_variance_across_K_time, out_file)
    with open("s_first_quantile_across_K_time", 'wb') as out_file:
        pickle.dump(first_quantile_across_K_time, out_file)
    with open("s_last_quantile_across_K_time", 'wb') as out_file:
        pickle.dump(last_quantile_across_K_time, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


