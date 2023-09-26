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
def func(N=None, K=None, state_num=None, specialist_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    # Sharing Crowd
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    for agent in crowd.agents:
        for _ in range(search_iteration):
            agent.search()
    solution_list = [agent.state.copy() for agent in crowd.agents]
    converged_performance_list = []
    converged_solution_list = []
    for agent_index in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, specialist_expertise=specialist_expertise)
        specialist.state = solution_list[agent_index]
        specialist.cog_fitness = specialist.get_cog_fitness(state=specialist.state, cog_state=specialist.cog_state)
        specialist.fitness = specialist.landscape.query_second_fitness(state=specialist.state)
        for _ in range(search_iteration):
            specialist.search()
        converged_performance_list.append(specialist.fitness)
        converged_solution_list.append(specialist.state)
    average_performance = sum(converged_performance_list) / len(converged_performance_list)
    best_performance = max(converged_performance_list)
    variance = np.std(converged_performance_list)
    diversity = get_diversity(belief_pool=converged_solution_list) / agent_num
    return_dict[loop] = [average_performance, variance, best_performance, diversity]
    sema.release()

def get_diversity(belief_pool: list):
    unique_solutions = []
    for belief in belief_pool:
        string_belief = "".join(belief)
        unique_solutions.append(string_belief)
    unique_solutions = set(unique_solutions)
    return len(unique_solutions)


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    search_iteration = 200
    N = 9
    state_num = 4
    specialist_expertise = 12
    # K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    agent_num_list = np.arange(10, 60, step=10, dtype=int).tolist()
    concurrency = 40
    for agent_num in agent_num_list:
        # DVs
        performance_across_K = []
        variance_across_K = []
        best_performance_across_K = []
        diversity_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, specialist_expertise,
                                                  agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_fitness, temp_variance, temp_best_performance, temp_diversity = [], [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_fitness.append(result[0])
                temp_variance.append(result[1])
                temp_best_performance.append(result[2])
                temp_diversity.append(result[3])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
            diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))

        # remove time dimension
        with open("ss_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("ss_variance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("ss_best_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("ss_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    print("SS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
