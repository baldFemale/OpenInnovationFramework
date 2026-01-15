#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
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
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    crowd.share_prob = 1
    crowd.lr = 1
    for _ in range(search_iteration):
        crowd.search()
        crowd.get_shared_pool()
        crowd.learn_from_shared_pool()
    performance_list = [agent.fitness for agent in crowd.agents]
    average_performance = sum(performance_list) / len(performance_list)
    breakthrough_likelihood = max(performance_list)
    variance = np.std(performance_list)

    domain_solution_dict = {}
    for agent in crowd.agents:
        domains = agent.generalist_domain.copy()  # !!!!
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

    return_dict[loop] = [breakthrough_likelihood, average_performance, variance, diversity]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 500
    search_iteration = 200
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    # alpha_list = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425]
    alpha = 0.25
    concurrency = 100
    # DVs
    breakthrough_likelihood_across_K = []
    average_performance_across_K = []
    variance_across_K = []
    solution_diversity_across_K = []
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
        arr = np.asarray(list(returns))  # shape: (n_runs, 4)
        means = arr.mean(axis=0)

        breakthrough_likelihood_across_K.append(means[0])
        average_performance_across_K.append(means[1])
        variance_across_K.append(means[2])
        solution_diversity_across_K.append(means[3])

        with open("gg_breakthrough_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(breakthrough_likelihood_across_K, out_file)
        with open("gg_ave_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(average_performance_across_K, out_file)
        with open("gg_variance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("gg_diversity_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(solution_diversity_across_K, out_file)

    t1 = time.time()
    print("GG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
