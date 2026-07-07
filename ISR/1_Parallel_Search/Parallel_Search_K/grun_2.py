#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: grun_1.py
# @Software : PyCharm
# Observing PEP 8 coding style

"""
Independent-search baseline for G crowds.

This script is the no-visibility counterpart to the visibility timing experiment.
It measures the same focal DVs used in the visibility scripts, but here agents
only conduct independent search. There is no sender crowd, receiver crowd,
visibility probability, visibility timing, or visibility interval.

DVs measured after search:
    - breakthrough_fitness: highest realized fitness in the crowd
    - breakthrough_rank: best realized global rank in the crowd
        smaller rank means better solution; rank 1 is the global best
    - diversity: number of unique complete solutions in the crowd
    - pairwise_diversity: average pairwise normalized Hamming distance

The outer loop varies crowd size. For each crowd size, results are averaged
across landscape repetitions for every K.
"""

import datetime
import multiprocessing as mp
import pickle
import time
from multiprocessing import Semaphore

import numpy as np

from Crowd import Crowd
from Landscape import Landscape


# mp version
def func(N=None, K=None, agent_num=None, search_iteration=None,
         loop=None, return_dict=None, sema=None):
    """
    Run one independent-search repetition on one landscape.

    This is intentionally simpler than the visibility experiment:
    - One G crowd is created on the landscape.
    - Agents search independently for search_iteration periods.
    - No solutions are disclosed, pooled, or learned through visibility.
    - DVs are measured on the final states of this independent crowd.
    """
    np.random.seed(None)

    try:
        landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

        crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                      generalist_expertise=12, specialist_expertise=0, label="G")

        for _ in range(search_iteration):
            crowd.search()

        performance_list = [agent.fitness for agent in crowd.agents]
        fitness_rank_list = [
            landscape.query_second_fitness_rank(state=agent.state)
            for agent in crowd.agents
        ]

        breakthrough_fitness = max(performance_list)
        breakthrough_rank = min(fitness_rank_list)

        # Full-solution diversity, aligned with the visibility scripts' DVs.
        full_solution_set = set()
        for agent in crowd.agents:
            solution_str = "".join([str(bit) for bit in agent.state])
            full_solution_set.add(solution_str)

        diversity = len(full_solution_set)
        pairwise_diversity = crowd.calculate_pairwise_solution_distance()

        return_dict[loop] = [
            breakthrough_fitness, breakthrough_rank, diversity, pairwise_diversity
        ]

    finally:
        sema.release()


if __name__ == '__main__':
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 400
    search_iteration = 300
    N = 9
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Independent-search baseline across crowd size.
    # Keep this list aligned with the evaluator for the crowd-size analysis.
    agent_num_list = np.arange(550, 850, step=50, dtype=int).tolist()

    concurrency = 100

    for agent_num in agent_num_list:
        # DVs across K for a given crowd size.
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        diversity_across_K = []
        pairwise_diversity_across_K = []

        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []

            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(
                    target=func,
                    args=(
                        N, K, agent_num, search_iteration,
                        loop, return_dict, sema
                    )
                )
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            returns = return_dict.values()  # Don't need dict index, since it is repetition.
            arr = np.asarray(list(returns))  # shape: (n_runs, 4)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])
            pairwise_diversity_across_K.append(means[3])

        # Save results across K for each independent-search crowd-size condition.
        with open("g_independent_breakthrough_fitness_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("g_independent_breakthrough_rank_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("g_independent_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("g_independent_pairwise_diversity_across_K_size_{0}".format(agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("G Independent Search: ", time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
