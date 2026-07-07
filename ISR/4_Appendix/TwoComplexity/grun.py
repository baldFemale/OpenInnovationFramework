# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle

from SearchTrajectory import generalist


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None, alpha=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=alpha)
    convergence_list = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num,
                           generalist_expertise=12)
        for _ in range(search_iteration):
            generalist.search()
        convergence_list.append(generalist.fitness)
    convergence = sum(convergence_list) / len(convergence_list)
    return_dict[loop] = [convergence]
    sema.release()

if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 200
    search_iteration = 300
    N = 9
    state_num = 4
    expertise_amount = 12   # Equal Expertise
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]
    alpha_list = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
    concurrency = 100

    for alpha in alpha_list:
        # DVs across K for each alpha
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        diversity_across_K = []
        pairwise_diversity_across_K = []
        average_fitness_across_K = []

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

            arr = np.asarray(list(returns))  # shape: (n_runs, 5)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])
            pairwise_diversity_across_K.append(means[3])
            average_fitness_across_K.append(means[4])

        # Save results across K for each alpha condition.
        with open("g_breakthrough_fitness_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("g_breakthrough_rank_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("g_diversity_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("g_pairwise_diversity_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("g_average_fitness_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(average_fitness_across_K, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
