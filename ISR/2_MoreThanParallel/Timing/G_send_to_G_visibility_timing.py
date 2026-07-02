#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: G_send_to_G_visibility_timing.py
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
         visibility_start=None, visibility_interval=10, loop=None, return_dict=None, sema=None):
    """
    Timing-based visibility experiment with separated sender and receiver crowds.

    - Two independent G crowds are created on the same landscape.
    - The sender crowd only searches and discloses visible full solutions after
      visibility starts and at specified visibility intervals.
    - The receiver crowd searches throughout and learns from sender's visible
      full solutions only after visibility starts and at specified visibility intervals.
    - The receiver crowd's learned solutions do not feed back into the visible pool.

    Visibility condition:
        visibility is activated after visibility_start and then every
        visibility_interval periods.

    Interpretation:
        visibility_prob = visibility intensity
            visibility_prob = 0.0 means no sender solution is structurally visible.
            visibility_prob = 1.0 means all sender solutions are structurally visible.

        visibility_start = visibility timing
            the first period from which sender solutions become visible.

        visibility_interval = visibility frequency after visibility starts
            visibility_interval = 1 means visible every period after visibility starts.
            visibility_interval = 5 means visible at visibility_start,
            visibility_start + 5, visibility_start + 10, ...
            visibility_interval = 10 is the default setting.

        Visibility object:
            visible_mode = "full" means the visible object is the sender's
            complete solution string, not a partial knowledge fragment.
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 10
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Generalists who only search and disclose visible solutions
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=18, specialist_expertise=0, label="G")

    # Receiver crowd: Generalists who search and learn from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=18, specialist_expertise=0, label="G")

    crowd_sender.set_visibility_status(visibility_prob=visibility_prob)

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        # Sender crowd constructs the visible full-solution pool only after
        # visibility starts and only at the specified visibility interval.
        # Importantly, this pool is based only on sender agents, whose states
        # are not affected by receiver learning.
        if (period >= visibility_start) and ((period - visibility_start) % visibility_interval == 0):
            crowd_sender.get_visible_pool(visible_mode="full")

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

    # Visibility probability is fixed in the timing experiment.
    # Keep the focal running parameter as visibility_start.
    visibility_prob = 1.0
    visibility_start_list = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 280, 290, 300]

    # Visibility interval: how frequently visibility is activated after visibility starts.
    # visibility_interval = 1 means visible every period after visibility starts.
    # visibility_interval = 5 means visible at visibility_start, visibility_start + 5, ...
    # visibility_interval = 10 is the default setting.
    visibility_interval = 10

    agent_num = 200
    concurrency = 100

    for visibility_start in visibility_start_list:
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
                p = mp.Process(
                    target=func,
                    args=(
                        N, K, agent_num, search_iteration, visibility_prob,
                        visibility_start, visibility_interval,
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

        # Save results across K for each visibility-start and visibility-interval condition.
        with open("gg_visibility_start_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("gg_visibility_start_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("gg_visibility_start_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("gg_visibility_start_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GG Visibility Timing with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
