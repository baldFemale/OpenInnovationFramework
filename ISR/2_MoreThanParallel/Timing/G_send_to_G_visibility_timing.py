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
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                  generalist_expertise=12, specialist_expertise=0, label="G")
    crowd.share_prob_list = [uniform_prob] * agent_num
    for period in range(search_iteration):
        crowd.search()
        if period >= visibility_start:
            crowd.get_shared_pool()
            crowd.learn_from_shared_pool()

    performance_list = [agent.fitness for agent in crowd.agents]
    fitness_rank_list = [landscape.query_second_fitness_rank(state=agent.state) for agent in crowd.agents]
    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    domain_solution_dict = {}
    for agent in crowd.agents:
        domains = agent.generalist_domain.copy()
        domains.sort()
        domain_str = "".join([str(i) for i in domains])
        solution_str = [agent.state[index] for index in domains]
        solution_str = "".join(solution_str)
        if domain_str not in domain_solution_dict.keys():
            domain_solution_dict[domain_str] = [solution_str]
        else:
            if solution_str not in domain_solution_dict[domain_str]:
                domain_solution_dict[domain_str].append(solution_str)
    diversity = 0
    for key, value in domain_solution_dict.items():
        diversity += len(value)

    return_dict[loop] = [breakthrough_fitness, breakthrough_rank, diversity]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    landscape_iteration = 500
    search_iteration = 300
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    uniform_prob = 0.2
    visibility_start_list = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    agent_num = 200
    concurrency = 100

    for visibility_start in visibility_start_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        diversity_across_K = []
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

            arr = np.asarray(list(returns))  # shape: (n_runs, 3)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])

        # remove time dimension
        with open("gg_visibility_start_{0}_breakthrough_fitness_across_K_size_{1}".format(visibility_start, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)
        with open("gg_visibility_start_{0}_breakthrough_rank_across_K_size_{1}".format(visibility_start, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)
        with open("gg_visibility_start_{0}_diversity_across_K_size_{1}".format(visibility_start, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GG Visibility Timing: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
