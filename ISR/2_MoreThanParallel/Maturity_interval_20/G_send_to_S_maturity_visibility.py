#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: G_send_to_S_maturity_visibility.py
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
def func(N=None, K=None, agent_num=None, search_iteration=None, uniform_sharing_prob=None,
         maturity_threshold=None, visibility_interval=1, loop=None, return_dict=None, sema=None):
    """
    Maturity-based visibility experiment with separated sender and receiver crowds.

    Difference from within-crowd visibility experiment:
    - Two independent crowds are created on the same landscape.
    - The sender crowd is composed of generalists who only search and share visible solutions.
    - The receiver crowd is composed of specialists who search and learn from the sender crowd's visible solutions.
    - The receiver crowd's learned solutions do not feed back into the visible pool.

    Sharing condition:
        visibility is activated every visibility_interval periods;
        when activated, share if random_draw < uniform_sharing_prob
        and agent.cog_fitness >= maturity_threshold.

    Interpretation:
        maturity_threshold = maturity selectivity
            maturity_threshold = 0.0 means almost all solutions are eligible.
            maturity_threshold = 0.9 means only highly mature solutions are eligible.

        uniform_sharing_prob = visibility intensity
            uniform_sharing_prob = 1 means every eligible solution is disclosed.

        visibility_interval = visibility frequency
            visibility_interval = 1 means visible every period, same as the original design.
            visibility_interval = 5 means visible at periods 5, 10, 15, ...
            visibility_interval = 20 means visible at periods 20, 40, 60, ...
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 1
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Generalists who only search and share
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=12, specialist_expertise=0, label="G")

    # Receiver crowd: Specialists who search and learn from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=12, label="S")

    crowd_sender.share_prob_list = [uniform_sharing_prob] * agent_num

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        # Visibility is activated only at specified intervals.
        if (period + 1) % visibility_interval == 0:

            # Sender crowd constructs the visible solution pool.
            # Importantly, this pool is based only on sender agents,
            # whose states are not affected by receiver learning.
            crowd_sender.solution_pool = []
            for agent, share_prob in zip(crowd_sender.agents, crowd_sender.share_prob_list):
                if (np.random.uniform(0, 1) < share_prob) and (agent.cog_fitness >= maturity_threshold):
                    domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
                    partial_solution = [agent.state[index] for index in domains]
                    crowd_sender.solution_pool.append([domains, partial_solution])

            np.random.shuffle(crowd_sender.solution_pool)

            # Receiver crowd learns only from sender's visible solutions.
            # No receiver solution is added back to the sender pool.
            crowd_receiver.solution_pool = [
                [domains.copy(), partial_solution.copy()]
                for domains, partial_solution in crowd_sender.solution_pool
            ]
            crowd_receiver.learn_from_shared_pool()

    # DVs are measured only on the receiver crowd.
    performance_list = [agent.fitness for agent in crowd_receiver.agents]
    fitness_rank_list = [
        landscape.query_second_fitness_rank(state=agent.state)
        for agent in crowd_receiver.agents
    ]

    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    # Calculate diversity among receiver agents.
    domain_solution_dict = {}
    for agent in crowd_receiver.agents:
        domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
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
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]
    uniform_sharing_prob = 1

    # Maturity threshold M_v: minimum cognitive fitness required for disclosure.
    # M_v = 0.0 means almost all solutions can be shared.
    # M_v = 1.0 means only nearly perfect subjectively evaluated solutions can be shared.
    maturity_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5,
                               0.6, 0.7, 0.8, 0.9]

    # Visibility interval: how frequently visibility is activated.
    # visibility_interval = 1 means visible every period, same as the original design.
    # visibility_interval = 5 means visible at periods 5, 10, 15, ...
    # visibility_interval = 20 means visible at periods 20, 40, 60, ...
    visibility_interval_list = [20]
    # For complementary experiments, you can use:
    # visibility_interval_list = [1, 5, 10, 20, 50]

    agent_num = 200
    concurrency = 100

    for visibility_interval in visibility_interval_list:
        for maturity_threshold in maturity_threshold_list:
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
                    p = mp.Process(
                        target=func,
                        args=(
                            N, K, agent_num, search_iteration,
                            uniform_sharing_prob, maturity_threshold,
                            visibility_interval,
                            loop, return_dict, sema
                        )
                    )
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

            # Save results across K for each maturity threshold and visibility interval.
            with open("gs_maturity_threshold_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                    maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
                pickle.dump(breakthrough_fitness_across_K, out_file)

            with open("gs_maturity_threshold_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                    maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
                pickle.dump(breakthrough_rank_across_K, out_file)

            with open("gs_maturity_threshold_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                    maturity_threshold, visibility_interval, agent_num), 'wb') as out_file:
                pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GS Maturity-Based Visibility with Interval: ",
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))