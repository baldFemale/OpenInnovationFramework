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
def func(N=None, K=None, state_num=None, agent_num=None, search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.35)
    # Transparent Crowd
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    crowd.share_prob = 1
    crowd.lr = 1
    for _ in range(search_iteration):
        crowd.search()
        crowd.get_shared_pool()
        crowd.learn_from_shared_pool()
    performance_list = [agent.fitness for agent in crowd.agents]
    average_performance = sum(performance_list) / len(performance_list)
    best_performance = max(performance_list)
    variance = np.std(performance_list)
    domain_solution_dict = {}
    for agent in crowd.agents:
        domains = agent.generalist_domain.copy()  # !!!!!
        domains.sort()
        domain_str = "".join([str(i) for i in domains])
        solution_str = [agent.cog_state[index] for index in domains]
        solution_str = "".join(solution_str)
        if domain_str not in domain_solution_dict.keys():
            domain_solution_dict[domain_str] = [solution_str]
        else:
            if solution_str not in domain_solution_dict[domain_str]:
                domain_solution_dict[domain_str].append(solution_str)
    diversity = 0
    for index, value in domain_solution_dict.items():
        diversity += len(value)
    diversity /= agent_num
    return_dict[loop] = [average_performance, best_performance, variance, diversity]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    state_num = 4
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 40
    # DVs
    ave_performance_across_K = []
    best_performance_across_K = []
    variance_across_K = []
    diversity_across_K = []
    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, state_num, agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        returns = return_dict.values()  # Don't need dict index, since it is repetition.

        temp_ave_performance, temp_best_performance, temp_variance, temp_diversity = [], [], [], []
        for result in returns:  # 200 landscape repetitions
            temp_ave_performance.append(result[0])
            temp_best_performance.append(result[1])
            temp_variance.append(result[2])
            temp_diversity.append(result[3])

        ave_performance_across_K.append(sum(temp_ave_performance) / len(temp_ave_performance))
        best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
        variance_across_K.append(sum(temp_variance) / len(temp_variance))
        diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))

    with open("gg_ave_performance_across_K", 'wb') as out_file:
        pickle.dump(ave_performance_across_K, out_file)
    with open("gg_best_performance_across_K", 'wb') as out_file:
        pickle.dump(best_performance_across_K, out_file)
    with open("gg_variance_across_K", 'wb') as out_file:
        pickle.dump(variance_across_K, out_file)
    with open("gg_diversity_across_K", 'wb') as out_file:
        pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    print("GG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
