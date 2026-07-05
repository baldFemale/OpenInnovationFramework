#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: S_send_to_S_visibility_degree.py
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
def func(N=None, K=None, agent_num=None, search_iteration=None, visibility_prob=None,
         visibility_interval=1, loop=None, return_dict=None, sema=None):
    """
    Visibility-degree experiment with separated sender and receiver crowds.

    # visibility_prob determines the proportion of sender agents whose
    # visibility_status is fixed as True for the whole run.
    # Every visibility interval, those visible senders disclose their current full solution.

    Visibility condition:
        visibility is activated every visibility_interval periods;
        when activated, sender visibility is determined by visibility_prob.

    Interpretation:
        visibility_prob = visibility intensity
        visibility_interval = visibility frequency
            visibility_interval = 1 means visible every period, same as the original design.
            visibility_interval = 5 means visible at periods 5, 10, 15, ...
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 1
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Specialists who only search and make solutions visible
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=0, specialist_expertise=32, label="S")

    # Receiver crowd: Specialists who search and learn from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=32, label="S")

    crowd_sender.set_visibility_status(visibility_prob=visibility_prob)

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        if (period + 1) % visibility_interval == 0:
            # Full visibility design:
            # sender agents disclose their complete solution states, rather than
            # only the fragments located in their own knowledge domains.
            crowd_sender.get_visible_pool(visible_mode="full")

            crowd_receiver.solution_pool = [
                [domains.copy(), solution.copy()]
                for domains, solution in crowd_sender.solution_pool
            ]
            crowd_receiver.learn_from_visible_pool()

    # DVs are measured only on the receiver crowd.
    performance_list = [agent.fitness for agent in crowd_receiver.agents]
    fitness_rank_list = [
        landscape.query_second_fitness_rank(state=agent.state)
        for agent in crowd_receiver.agents
    ]

    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    # Calculate diversity among receiver agents.
    # Under full visibility, diversity should be measured at the complete-solution
    # level, not only within each agent's knowledge domains.
    full_solution_set = set()
    for agent in crowd_receiver.agents:
        solution_str = "".join([str(bit) for bit in agent.state])
        full_solution_set.add(solution_str)

    diversity = len(full_solution_set)
    pairwise_diversity = crowd_receiver.calculate_pairwise_solution_distance()

    return_dict[loop] = [
        breakthrough_fitness, breakthrough_rank, diversity, pairwise_diversity
    ]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 400
    search_iteration = 300
    N = 9
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Visibility degree p_v: probability that a focal S solution is disclosed to S
    # during a visibility period.
    # p_v = 0.0 means no cross-agent visibility.
    # p_v = 1.0 means all S solutions enter the visible shared pool during each visibility period.
    visibility_prob_list = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08,
                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Visibility frequency: S solutions become visible to S every x periods.
    visibility_interval = 10

    agent_num = 200
    concurrency = 100

    for visibility_prob in visibility_prob_list:
        # DVs
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
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration,
                                                  visibility_prob, visibility_interval,
                                                  loop, return_dict, sema))
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

        # Save results across K for each visibility probability and visibility interval.
        with open("ss_visibility_prob_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("ss_visibility_prob_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("ss_visibility_prob_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("ss_visibility_prob_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("SS Visibility Degree with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))