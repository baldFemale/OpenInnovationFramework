#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: S_send_to_S_rank_visibility.py
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
         visibility_interval=10, loop=None, return_dict=None, sema=None):
    """
    Visibility condition:
        visibility is activated every visibility_interval periods;
        when activated, all structurally visible sender solutions enter the
        visible pool. Receiver attention is then rank-weighted by sender
        objective fitness, so higher-ranked sender solutions are more likely
        to be imitated.

    Interpretation:
        visibility_prob = visibility intensity
            visibility_prob = 0.0 means no sender solution is structurally visible.
            visibility_prob = 1.0 means all sender solutions are structurally visible.

        visibility_interval = visibility frequency
            visibility_interval = 1 means visible every period.
            visibility_interval = 5 means visible at periods 5, 10, 15, ...
            visibility_interval = 10 is the default setting.

        Visibility object:
            visible_mode = "full" means the visible object is the sender's
            complete solution string, not a partial knowledge fragment.
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 150
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Specialists who only search and disclose visible solutions
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=0, specialist_expertise=20, label="S")

    # Receiver crowd: Specialists who search and imitate from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=20, label="S")

    crowd_sender.set_visibility_status(visibility_prob=visibility_prob)

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        # Sender crowd constructs the visible full-solution pool only at
        # disclosure intervals. Importantly, this pool is based only on sender
        # agents, whose states are not affected by receiver imitation.
        if (period + 1) % visibility_interval == 0:
            visible_pool = []
            full_domains = list(range(N))

            for agent in crowd_sender.agents:
                if agent.visibility_status:
                    # Full-solution visibility:
                    visible_pool.append([
                        full_domains.copy(), agent.state.copy(), agent.fitness
                    ])

            if visible_pool:
                # Greater objective fitness -> higher probability of being imitated.
                fitness_values = np.asarray(
                    [fitness for _, _, fitness in visible_pool], dtype=float
                )

                # Use fitness-proportional attention. The small constant avoids
                # division-by-zero if all visible solutions have zero fitness.
                imitation_weights = fitness_values + 1e-10
                imitation_probs = imitation_weights / imitation_weights.sum()

                for agent in crowd_receiver.agents:
                    selected_index = np.random.choice(
                        len(visible_pool), p=imitation_probs
                    )
                    domains, solution, _ = visible_pool[selected_index]

                    imitated_solution = agent.state.copy()
                    for domain, bit in zip(domains, solution):
                        imitated_solution[domain] = bit

                    # Direct imitation: no cognitive-fitness improvement check.
                    agent.state = imitated_solution
                    agent.cog_state = agent.state_2_cog_state(
                        state=imitated_solution
                    )
                    agent.cog_fitness = agent.get_cog_fitness(
                        cog_state=agent.cog_state, state=imitated_solution
                    )
                    agent.fitness = agent.landscape.query_second_fitness(
                        state=imitated_solution
                    )
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

    # Visibility degree is the focal running parameter.
    visibility_prob_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9, 1.0]

    # Visibility interval: how frequently visibility is activated.
    # visibility_interval = 1 means visible every period.
    # visibility_interval = 5 means visible at periods 5, 10, 15, ...
    # visibility_interval = 10 is the default setting.
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
                p = mp.Process(
                    target=func,
                    args=(
                        N, K, agent_num, search_iteration, visibility_prob,
                        visibility_interval, loop, return_dict, sema
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

        # Save results across K for each visibility degree and visibility interval.
        with open("ss_rank_based_visibility_prob_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("ss_rank_based_visibility_prob_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("ss_rank_based_visibility_prob_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

        with open("ss_rank_based_visibility_prob_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("SS Rank-Based Visibility with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
