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
    domain_solution_dict = {}
    aligned_domain = np.random.choice(range(N), generalist_expertise // 2, replace=False).tolist()
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num,
                           generalist_expertise=generalist_expertise)
        generalist.generalist_domain = aligned_domain
        generalist.cog_state = generalist.state_2_cog_state(state=generalist.state)
        generalist.cog_fitness = generalist.get_cog_fitness(cog_state=generalist.cog_state, state=generalist.state)
        for _ in range(search_iteration):
            generalist.search()
        converged_performance_list.append(generalist.fitness)
        converged_solution_list.append(generalist.state)
        domains = generalist.generalist_domain.copy()
        domains.sort()
        domain_str = "".join([str(i) for i in domains])
        solution_str = [generalist.state[index] for index in domains]
        solution_str = "".join(solution_str)
        if domain_str not in domain_solution_dict.keys():
            domain_solution_dict[domain_str] = [solution_str]
        else:
            if solution_str not in domain_solution_dict[domain_str]:
                domain_solution_dict[domain_str].append(solution_str)
    partial_unique_diversity = 0
    for key, value in domain_solution_dict.items():
        partial_unique_diversity += len(value)
    average_performance = sum(converged_performance_list) / len(converged_performance_list)
    best_performance = max(converged_performance_list)
    worst_performance = min(converged_performance_list)
    variance = np.std(converged_performance_list)
    unique_diversity = get_unique_diversity(belief_pool=converged_solution_list)
    pair_wise_diversity = get_pair_wise_diversity(belief_pool=converged_solution_list)
    return_dict[loop] = [average_performance, variance, unique_diversity, pair_wise_diversity,
                         best_performance, worst_performance, partial_unique_diversity]
    sema.release()

def get_unique_diversity(belief_pool: list):
    unique_solutions = []
    for belief in belief_pool:
        string_belief = "".join(belief)
        unique_solutions.append(string_belief)
    unique_solutions = set(unique_solutions)
    return len(unique_solutions)

def get_pair_wise_diversity(belief_pool: list):
    diversity = 0
    for index, focal_belief in enumerate(belief_pool):
        selected_pool = belief_pool[index + 1::]
        one_pair_diversity = [get_distance(focal_belief, belief) for belief in selected_pool]
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
    agent_num_list = np.arange(600, 850, step=50, dtype=int).tolist()
    search_iteration = 100
    N = 9
    state_num = 4
    generalist_expertise = 12
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    for agent_num in agent_num_list:
        # DVs
        performance_across_K = []
        variance_across_K = []
        unique_diversity_across_K = []
        pair_wise_diversity_across_K = []
        best_performance_across_K = []
        worst_performance_across_K = []
        partial_unique_diversity_across_K = []
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

            temp_fitness, temp_variance, temp_unique_diversity, temp_pair_wise_diversity = [], [], [], []
            temp_best_performance, temp_worst_performance = [], []
            temp_partial_unique = []
            for result in returns:  # 50 landscape repetitions
                temp_fitness.append(result[0])
                temp_variance.append(result[1])
                temp_unique_diversity.append(result[2])
                temp_pair_wise_diversity.append(result[3])
                temp_best_performance.append(result[4])
                temp_worst_performance.append(result[5])
                temp_partial_unique.append(result[6])

            performance_across_K.append(sum(temp_fitness) / len(temp_fitness))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            unique_diversity_across_K.append(sum(temp_unique_diversity) / len(temp_unique_diversity))
            pair_wise_diversity_across_K.append(sum(temp_pair_wise_diversity) / len(temp_pair_wise_diversity))
            best_performance_across_K.append(sum(temp_best_performance) / len(temp_best_performance))
            worst_performance_across_K.append(sum(temp_worst_performance) / len(temp_worst_performance))
            partial_unique_diversity_across_K.append(sum(temp_partial_unique) / len(temp_partial_unique))
        # remove time dimension
        with open("g_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("g_variance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("g_unique_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(unique_diversity_across_K, out_file)
        with open("g_pair_wise_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(pair_wise_diversity_across_K, out_file)
        with open("g_best_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("g_worst_performance_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(worst_performance_across_K, out_file)
        with open("g_partial_unique_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(partial_unique_diversity_across_K, out_file)

    t1 = time.time()
    print("G12: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


