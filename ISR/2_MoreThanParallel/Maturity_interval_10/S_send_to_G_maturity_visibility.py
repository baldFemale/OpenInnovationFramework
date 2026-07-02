#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: S_send_to_G_maturity_visibility.py
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
def func(N=None, K=None, agent_num=None, search_iteration=None, visibility_prob=1,
         maturity_threshold=None, visibility_interval=1, loop=None, return_dict=None, sema=None):
    """
    Maturity-based visibility experiment with separated sender and receiver crowds.

    - Two independent crowds are created on the same landscape.
    - The sender crowd is composed of specialists who only search and share visible full solutions.
    - The receiver crowd is composed of generalists who search and learn from sender's visible full solutions.
    - The receiver crowd's learned solutions do not feed back into the visible pool.

    Visibility condition:
        visibility is activated every visibility_interval periods;
        when activated, disclose a sender solution if:
            1. the sender is structurally visible, and
            2. the sender's cognitive fitness reaches maturity_threshold.

    Interpretation:
        maturity_threshold = maturity selectivity
            maturity_threshold = 0.0 means almost all visible sender solutions are eligible.
            maturity_threshold = 0.9 means only highly mature visible sender solutions are eligible.

        visibility_prob = visibility intensity
            visibility_prob = 1.0 means all sender solvers are structurally visible.
            visibility_prob = 0.5 means half of sender solvers are structurally visible.

        visibility_interval = visibility frequency
            visibility_interval = 1 means visible every period, same as the original design.
            visibility_interval = 5 means visible at periods 5, 10, 15, ...
            visibility_interval = 20 means visible at periods 20, 40, 60, ...

        Visibility object:
            visible_mode = "full" means the visible object is the sender's
            complete solution string, not a partial knowledge fragment.
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 1
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Specialists who only search and share
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=0, specialist_expertise=20, label="S")

    # Receiver crowd: Generalists who search and learn from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=18, specialist_expertise=0, label="G")

    crowd_sender.set_visibility_status(visibility_prob=visibility_prob)

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        if (period + 1) % visibility_interval == 0:
            # Full-solution maturity-based visibility:
            # The sender crowd discloses complete solution strings rather than
            # domain-specific partial fragments. A sender solution is disclosed
            # only if the sender is structurally visible and its cognitive
            # fitness reaches the maturity threshold.
            crowd_sender.solution_pool = []
            for agent in crowd_sender.agents:
                if agent.visibility_status and agent.cog_fitness >= maturity_threshold:
                    domains = list(range(N))
                    full_solution = agent.state.copy()
                    crowd_sender.solution_pool.append([domains, full_solution])

            np.random.shuffle(crowd_sender.solution_pool)

            # Receiver crowd learns only from sender's visible full solutions.
            # No receiver solution is added back to the sender pool.
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

    # Calculate full-solution diversity among receiver agents.
    # Since visibility now discloses complete solutions, diversity should also
    # be measured at the complete-solution level rather than only on each
    # agent's knowledge domains.
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

    landscape_iteration = 200
    search_iteration = 300
    N = 9
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Visibility degree is fixed in this maturity-threshold experiment.
    # visibility_prob = 1.0 means all sender solvers are structurally visible.
    visibility_prob = 1.0

    # Maturity threshold M_v: minimum cognitive fitness required for disclosure.
    # M_v = 0.0 means almost all visible sender solutions can be disclosed.
    # M_v = 1.0 means only nearly perfect subjectively evaluated solutions can be disclosed.
    maturity_threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4,
                               0.5, 0.6, 0.7, 0.8, 0.9]

    # Visibility frequency: sender solutions become visible every x periods.
    visibility_interval = 10

    agent_num = 200
    concurrency = 100

    for maturity_threshold in maturity_threshold_list:
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
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration, visibility_prob,
                                                  maturity_threshold, visibility_interval,
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

        # Save results across K for each maturity threshold and visibility interval.
        with open("sg_maturity_threshold_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("sg_maturity_threshold_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("sg_maturity_threshold_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("sg_maturity_threshold_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("SG Maturity-Based Visibility with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
