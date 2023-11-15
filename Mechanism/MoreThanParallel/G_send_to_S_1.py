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
def func(N=None, K=None, agent_num=None, search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.25)
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
        # crowd_s.get_shared_pool()
        # s_pool = crowd_s.solution_pool.copy()
        # crowd_g.solution_pool = s_pool
        # crowd_g.learn_from_shared_pool()
        # G share a pool to S
        crowd_g.get_shared_pool()
        g_pool = crowd_g.solution_pool.copy()
        crowd_s.solution_pool = g_pool
        crowd_s.learn_from_shared_pool()

    performance_list = [agent.fitness for agent in crowd_s.agents]  # !!!!!
    average_performance = sum(performance_list) / len(performance_list)
    best_performance = max(performance_list)
    variance = np.std(performance_list)
    # Calculate the diversity indicator
    domain_list = []
    for agent in crowd_s.agents:  # !!!!
        domains = agent.specialist_domain.copy()  # !!!!
        domains.sort()
        if domains not in domain_list:
            domain_list.append(domains)
    solution_dict = {}
    for agent in crowd_s.agents:
        for domains in domain_list:
            domain_str = "".join([str(i) for i in domains])
            # Using state as to solution diversity
            solution_str = [agent.state[index] for index in domains]
            solution_str = "".join(solution_str)
            if domain_str not in solution_dict.keys():
                solution_dict[domain_str] = [solution_str]
            else:
                if solution_str not in solution_dict[domain_str]:
                    solution_dict[domain_str].append(solution_str)
    partitioned_diversity = 0
    for value in solution_dict.values():
        partitioned_diversity += len(value)
    return_dict[loop] = [average_performance, best_performance, variance, partitioned_diversity]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    landscape_iteration = 200
    search_iteration = 200
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    agent_num_list = np.arange(50, 400, step=50, dtype=int).tolist()
    concurrency = 100
    for agent_num in agent_num_list:
        # DVs
        performance_across_K = []
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
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_fitness, temp_best_performance, temp_variance, temp_diversity = [], [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_fitness.append(result[0])
                temp_best_performance.append(result[1])
                temp_variance.append(result[2])
                temp_diversity.append(result[3])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))

        # remove time dimension
        with open("gs_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("gs_variance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("gs_best_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("gs_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GS: ", time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))