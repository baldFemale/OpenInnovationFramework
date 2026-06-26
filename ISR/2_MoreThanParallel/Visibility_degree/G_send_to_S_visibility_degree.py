#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: G_send_to_S_visibility_degree.py
# @Software : PyCharm
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
def func(N=None, K=None, agent_num=None, search_iteration=None,
         visibility_prob=None, loop=None, return_dict=None, sema=None):
    """
    Visibility-degree experiment.

    Difference from maturity-based visibility experiment:
    - No maturity threshold is used.
    - Visibility is available in every period.
    - A G solution enters the shared pool with probability visibility_prob.

    Sharing condition:
        share if random_draw < visibility_prob
    """
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.25)

    # Transparent Crowd: Generalist-to-Specialist visibility
    crowd_s = Crowd(N=N, agent_num=agent_num // 2, landscape=landscape, state_num=4,
                    generalist_expertise=0, specialist_expertise=12, label="S")
    crowd_g = Crowd(N=N, agent_num=agent_num // 2, landscape=landscape, state_num=4,
                    generalist_expertise=12, specialist_expertise=0, label="G")
    crowd_s.share_prob_list = [visibility_prob] * (agent_num // 2)
    crowd_g.share_prob_list = [visibility_prob] * (agent_num // 2)

    for period in range(search_iteration):
        crowd_s.search()
        crowd_g.search()

        # Visibility-degree shared pool: visibility is always available,
        # but each G solution is disclosed to S with probability visibility_prob.
        crowd_g.solution_pool = []
        for agent, share_prob in zip(crowd_g.agents, crowd_g.share_prob_list):
            if np.random.uniform(0, 1) < share_prob:
                domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
                partial_solution = [agent.state[index] for index in domains]
                crowd_g.solution_pool.append([domains, partial_solution])
        np.random.shuffle(crowd_g.solution_pool)

        # G share a pool to S
        g_pool = crowd_g.solution_pool.copy()
        crowd_s.solution_pool = g_pool
        crowd_s.learn_from_shared_pool()

    performance_list = [agent.fitness for agent in crowd_s.agents]
    fitness_rank_list = [landscape.query_second_fitness_rank(state=agent.state) for agent in crowd_s.agents]
    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    # Calculate the diversity indicator
    domain_solution_dict = {}
    for agent in crowd_s.agents:
        domains = agent.specialist_domain.copy()
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

    return_dict[loop] = [breakthrough_fitness, breakthrough_rank, diversity]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 200
    search_iteration = 300
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Visibility degree p_v: probability that a G solution becomes visible to S.
    # p_v = 0.0 means no G solutions are visible to S.
    # p_v = 1.0 means all G solutions are visible to S in every period.
    visibility_prob_list = [0.0, 0.1, 0.2, 0.3, 0.4,
                            0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    agent_num = 200
    concurrency = 100

    for visibility_prob in visibility_prob_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        diversity_across_K = []

        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []

            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration,
                                                  visibility_prob, loop, return_dict, sema))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            returns = return_dict.values()  # Don't need dict index, since it is repetition.
            arr = np.asarray(list(returns))  # shape: (n_runs, 3)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])

        # Save results across K for each visibility probability.
        with open("gs_visibility_prob_{0}_breakthrough_fitness_across_K_size_{1}".format(
                visibility_prob, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("gs_visibility_prob_{0}_breakthrough_rank_across_K_size_{1}".format(
                visibility_prob, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("gs_visibility_prob_{0}_diversity_across_K_size_{1}".format(
                visibility_prob, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GS Visibility Degree: ", time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
