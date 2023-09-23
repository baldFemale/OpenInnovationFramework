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
    convergence_list = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, generalist_expertise=expertise_amount)
        for _ in range(search_iteration):
            generalist.search()
        convergence_list.append(generalist.fitness)
    convergence = sum(convergence_list) / len(convergence_list)
    return_dict[loop] = [convergence]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    expertise_amount = 18  # Full Domains
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25]
    concurrency = 40
    for alpha in alpha_list:
        # DVs
        performance_across_K = []
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

            temp_fitness = []
            for result in returns:  # 50 landscape repetitions
                temp_fitness.append(result[0])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
        # remove time dimension
        with open("g_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


