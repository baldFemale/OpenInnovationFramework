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
def func(N=None, K=None, agent_num=None, search_iteration=None, uniform_prob=None,
         visibility_start=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.25)
    # Transparent Crowd
    crowd_s = Crowd(N=N, agent_num=agent_num // 2, landscape=landscape, state_num=4,
                    generalist_expertise=0, specialist_expertise=12, label="S")
    crowd_g = Crowd(N=N, agent_num=agent_num // 2, landscape=landscape, state_num=4,
                    generalist_expertise=12, specialist_expertise=0, label="G")
    crowd_s.share_prob_list = [uniform_prob] * (agent_num // 2)
    crowd_g.share_prob_list = [uniform_prob] * (agent_num // 2)
    for period in range(search_iteration):
        crowd_s.search()
        crowd_g.search()
        if period >= visibility_start:
            # G share a pool to S
            crowd_g.get_shared_pool()
            g_pool = crowd_g.solution_pool.copy()
            crowd_s.solution_pool = g_pool
            crowd_s.learn_from_shared_pool()

    performance_list = [agent.fitness for agent in crowd_s.agents]
    fitness_rank_list = [landscape.query_second_fitness_rank(state=agent.state) for agent in crowd_s.agents]
    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    return_dict[loop] = [breakthrough_fitness, breakthrough_rank]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    landscape_iteration = 400
    search_iteration = 200
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    uniform_prob = 0.2
    visibility_start_list = [0, 50, 100, 150, 200]
    agent_num = 200
    concurrency = 100

    for visibility_start in visibility_start_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration, uniform_prob,
                                                  visibility_start, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            arr = np.asarray(list(returns))  # shape: (n_runs, 2)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])

        # remove time dimension
        with open("gs_visibility_start_{0}_breakthrough_fitness_across_K_size_{1}".format(visibility_start, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)
        with open("gs_visibility_start_{0}_breakthrough_rank_across_K_size_{1}".format(visibility_start, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GS Visibility Timing: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
