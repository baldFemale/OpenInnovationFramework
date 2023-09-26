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
def func(N=None, K=None, state_num=None, generalist_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    converged_performance_list = []
    converged_solution_list = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num,
                           generalist_expertise=generalist_expertise)
        for _ in range(search_iteration):
            generalist.search()
        converged_performance_list.append(generalist.fitness)
        converged_solution_list.append(generalist.state)
    average_performance = sum(converged_performance_list) / len(converged_performance_list)
    best_performance = max(converged_performance_list)
    worst_performance = min(converged_performance_list)
    variance = np.std(converged_performance_list)
    diversity = get_diversity(belief_pool=converged_solution_list)
    return_dict[loop] = [average_performance, variance, diversity, best_performance, worst_performance]
    sema.release()

def get_diversity(belief_pool: list):
    diversity = 0
    for index in range(len(belief_pool)):
        selected_pool = belief_pool[index + 1::]
        one_pair_diversity = [get_distance(belief_pool[index], belief) for belief in selected_pool]
        diversity += sum(one_pair_diversity)
    return diversity / len(belief_pool[0]) / (len(belief_pool) - 1) / len(belief_pool) * 2

def get_distance(a=None, b=None):
    acc = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            acc += 1
    return acc

if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    # agent_num = 100
    agent_num_list = np.arange(110, 160, step=10, dtype=int).tolist()
    search_iteration = 200
    N = 9
    state_num = 4
    generalist_expertise = 12
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 40
    for agent_num in agent_num_list:
        # DVs
        performance_across_K = []
        variance_across_K = []
        diversity_across_K = []
        best_performance_across_K = []
        worst_performance_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, generalist_expertise,
                                                  agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_fitness, temp_variance, temp_diversity = [], [], []
            temp_best_performance, temp_worst_performance = [], []
            for result in returns:  # 50 landscape repetitions
                temp_fitness.append(result[0])
                temp_variance.append(result[1])
                temp_diversity.append(result[2])
                temp_best_performance.append(result[3])
                temp_worst_performance.append(result[4])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))
            best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
            worst_performance_across_K.append(sum(temp_worst_performance) / len(temp_worst_performance))
        # remove time dimension
        with open("g_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("g_variance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("g_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)
        with open("g_best_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("g_worst_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(worst_performance_across_K, out_file)

    t1 = time.time()
    print("G12: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


