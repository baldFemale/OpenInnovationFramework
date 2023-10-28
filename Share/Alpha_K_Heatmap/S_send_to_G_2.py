#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, alpha=None, agent_num=None, search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=alpha)
    # Transparent Crowd
    crowd_s = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    crowd_g = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    crowd_s.share_prob = 1
    crowd_s.lr = 1
    crowd_g.share_prob = 1
    crowd_g.lr = 1
    for _ in range(search_iteration):
        crowd_s.search()
        crowd_g.search()
        # S share a pool to G
        crowd_s.get_shared_pool()
        s_pool = crowd_s.solution_pool.copy()
        crowd_g.solution_pool = s_pool
        crowd_g.learn_from_shared_pool()
        # G share a pool to S
        # crowd_g.get_shared_pool()
        # g_pool = crowd_g.solution_pool.copy()
        # crowd_s.solution_pool = g_pool
        # crowd_s.learn_from_shared_pool()

    performance_list = [agent.fitness for agent in crowd_g.agents]  # !!!!!
    average_performance = sum(performance_list) / len(performance_list)
    best_performance = max(performance_list)
    variance = np.std(performance_list)
    return_dict[loop] = [average_performance, best_performance, variance]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    state_num = 4
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    alpha_list = [0.20, 0.25, 0.30]
    concurrency = 50
    for alpha in alpha_list:
        ave_performance_across_K = []
        best_performance_across_K = []
        variance_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, alpha, agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_ave_performance, temp_best_performance, temp_variance = [], [], []
            for result in returns:  # 200 landscape repetitions
                temp_ave_performance.append(result[0])
                temp_best_performance.append(result[1])
                temp_variance.append(result[2])

            ave_performance_across_K.append(sum(temp_ave_performance) / len(temp_ave_performance))
            best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))

        with open("sg_ave_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(ave_performance_across_K, out_file)
        with open("sg_best_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("sg_variance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)

    t1 = time.time()
    print("SG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
