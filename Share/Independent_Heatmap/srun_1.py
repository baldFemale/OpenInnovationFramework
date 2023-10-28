# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Specialist import Specialist
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, alpha=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=alpha)
    performance_list = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=4, specialist_expertise=expertise_amount)
        for _ in range(search_iteration):
            specialist.search()
        performance_list.append(specialist.fitness)
    ave_performance = sum(performance_list) / len(performance_list)
    best_performance = max(performance_list)
    variance = np.std(performance_list)
    return_dict[loop] = [ave_performance, best_performance, variance]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    expertise_amount = 12   # Equal Expertise
    # alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    alpha_list = [0.05, 0.10, 0.15, 0.20]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    for alpha in alpha_list:
        # DVs
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
                p = mp.Process(target=func, args=(N, K, alpha, expertise_amount, agent_num,
                                                  search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_average, temp_best, temp_variance = [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_average.append(result[0])
                temp_best.append(result[1])
                temp_variance.append(result[2])

            ave_performance_across_K.append(sum(temp_average) / len(temp_average))
            best_performance_across_K.append(sum(temp_best) / len(temp_best))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))

        with open("s_ave_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(ave_performance_across_K, out_file)
        with open("s_best_performance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("s_variance_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)

    t1 = time.time()
    print("S: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


