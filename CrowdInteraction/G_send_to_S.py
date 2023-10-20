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
def func(N=None, K=None, state_num=None, share_prob=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    # Sharing Crowd
    g_crowd_1 = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0, label="G", share_prob=share_prob)
    s_crowd_2 = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12, label="S", share_prob=share_prob)
    for _ in range(50):
        # Crowd 1 search
        for agent_1 in g_crowd_1.agents:
            for _ in range(search_iteration):
                agent_1.search()
        # Crowd 1 share
        shared_states = g_crowd_1.get_shared_state_pool()
        if len(shared_states) == 0:
            pass
        else:
            # Crowd 2 imitate
            for agent_2 in s_crowd_2.agents:
                selected_index = np.random.choice(range(len(shared_states)))
                agent_2.state = shared_states[selected_index].copy()
                agent_2.cog_state = agent_2.state_2_cog_state(state=agent_2.state)
                agent_2.cog_fitness = agent_2.get_cog_fitness(cog_state=agent_2.cog_state, state=agent_2.state)
                agent_2.fitness = landscape.query_second_fitness(state=agent_2.state)
        for agent_2 in s_crowd_2.agents:
            for _ in range(search_iteration):
                agent_2.search()
        # Crowd 2 also share
        shared_states = s_crowd_2.get_shared_state_pool()
        if len(shared_states) == 0:
            pass
        else:
            # Crowd 1 also imitate
            for agent_1 in g_crowd_1.agents:
                selected_index = np.random.choice(range(len(shared_states)))
                agent_1.state = shared_states[selected_index].copy()
                agent_1.cog_state = agent_1.state_2_cog_state(state=agent_1.state)
                agent_1.cog_fitness = agent_1.get_cog_fitness(cog_state=agent_1.cog_state, state=agent_1.state)
                agent_1.fitness = landscape.query_second_fitness(state=agent_1.state)
    # Only measure the second one -> being consistent with prior comparison
    performance_list = []
    for agent_2 in s_crowd_2.agents:
        performance_list.append(agent_2.fitness_across_time[-1])
    # Average
    performance = sum(performance_list) / len(performance_list)
    variance = np.std(performance_list)
    domain_solution_dict = {}
    for agent_2 in s_crowd_2.agents:
        # Only measure one half; Agent 2
        domains = agent_2.specialist_domain.copy()
        domains.sort()
        domain_str = "".join([str(i) for i in domains])
        solution_str = [agent_2.cog_state[index] for index in domains]
        solution_str = "".join(solution_str)
        if domain_str not in domain_solution_dict.keys():
            domain_solution_dict[domain_str] = [solution_str]
        else:
            if solution_str not in domain_solution_dict[domain_str]:
                domain_solution_dict[domain_str].append(solution_str)
    partial_unique_diversity = 0
    for key, value in domain_solution_dict.items():
        partial_unique_diversity += len(value)
    return_dict[loop] = [performance, variance, partial_unique_diversity]
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
    landscape_iteration = 200
    agent_num = 100
    search_iteration = 100
    N = 9
    state_num = 4
    # K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    share_prob_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    concurrency = 50
    # DVs
    for share_prob in share_prob_list:
        performance_across_K = []
        variance_across_K = []
        diversity_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, share_prob,
                                                  agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_performance, temp_variance, temp_diversity = [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_performance.append(result[0])
                temp_variance.append(result[1])
                temp_diversity.append(result[2])

            performance_across_K.append(sum(temp_performance) / len(temp_performance))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))

        # remove time dimension
        with open("gs_performance_across_K_share_prob_{}".format(share_prob), 'wb') as out_file:
            pickle.dump(performance_across_K, out_file)
        with open("gs_variance_across_K_share_prob_{}".format(share_prob), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("gs_diversity_across_K_share_prob_{}".format(share_prob), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    print("GS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
