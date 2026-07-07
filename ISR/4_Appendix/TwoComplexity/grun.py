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


def calculate_pairwise_solution_distance(states, N):
    """Average pairwise normalized Hamming distance across complete solutions."""
    if len(states) <= 1:
        return 0

    distance_list = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distance = (
                sum(
                    1
                    for bit_i, bit_j in zip(states[i], states[j])
                    if bit_i != bit_j
                )
                / N
            )
            distance_list.append(distance)

    return np.mean(distance_list)


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None, alpha=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=alpha)

    fitness_list = []
    fitness_rank_list = []
    full_solution_set = set()
    state_list = []

    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num,
                                generalist_expertise=expertise_amount)
        for _ in range(search_iteration):
            generalist.search()

        fitness_list.append(generalist.fitness)
        fitness_rank_list.append(
            landscape.query_second_fitness_rank(state=generalist.state)
        )

        solution_str = "".join([str(bit) for bit in generalist.state])
        full_solution_set.add(solution_str)
        state_list.append(generalist.state)

    breakthrough_fitness = max(fitness_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best
    diversity = len(full_solution_set)
    pairwise_diversity = calculate_pairwise_solution_distance(states=state_list, N=N)

    return_dict[loop] = [
        breakthrough_fitness, breakthrough_rank, diversity, pairwise_diversity
    ]
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

            arr = np.asarray(list(returns))  # shape: (n_runs, 4)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])
            pairwise_diversity_across_K.append(means[3])

        # Save results across K for each alpha condition.
        with open("g_breakthrough_fitness_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("g_breakthrough_rank_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("g_diversity_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("g_pairwise_diversity_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
