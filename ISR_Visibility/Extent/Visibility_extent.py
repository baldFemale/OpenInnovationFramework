#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_extent.py
# @Software : PyCharm
# Observing PEP 8 coding style

import copy
import numpy as np
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, agent_num=None, knowledge_breadth=None,
         search_iteration=None, visibility_extent=None,
         visibility_interval=10, loop=None, return_dict=None, sema=None):
    """
    Visibility-extent experiment with separate sender and receiver crowds.

    Both crowds contain the same type of bounded-knowledge solvers and search
    independently on the same binary NK landscape.

    visibility_extent determines the proportion of sender solvers whose current
    solutions are structurally visible throughout the run.

    Every visibility_interval periods:
        1. the sender crowd constructs a pool of its visible current solutions;
        2. the receiver crowd observes that pool;
        3. receiver solvers self-selectively adopt visible solutions according
           to their own bounded evaluation.

    Receiver learning does not feed back into the sender crowd.

    All dependent variables are measured only on the receiver crowd.
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 10
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    # Conventional binary NK landscape
    landscape = Landscape(N=N, K=K)

    # Two separate crowds with the same solver architecture and knowledge breadth.
    crowd_sender = Crowd(
        N=N,
        agent_num=agent_num,
        knowledge_breadth=knowledge_breadth,
        landscape=landscape
    )

    crowd_receiver = Crowd(
        N=N,
        agent_num=agent_num,
        knowledge_breadth=knowledge_breadth,
        landscape=landscape
    )

    # Only sender solutions become structurally visible.
    crowd_sender.set_visibility_status(visibility_extent)

    # Search-process trajectory:
    # cumulative mean objective fitness of solutions adopted by receivers.
    adopted_solution_fitness_across_time = []

    for period in range(search_iteration):
        # Both crowds conduct their own independent local search.
        crowd_sender.search()
        crowd_receiver.search()

        if (period + 1) % visibility_interval == 0:
            # Sender crowd releases its currently visible complete solutions.
            crowd_sender.get_visible_pool()

            # Receiver crowd observes only solutions from the sender crowd.
            # Deep copy keeps receiver adoption from changing the sender pool.
            crowd_receiver.solution_pool = copy.deepcopy(crowd_sender.solution_pool)
            crowd_receiver.learn_from_visible_pool()

            adopted_solution_fitness_across_time.append(
                np.mean(crowd_receiver.adopted_solution_fitness_history)
                if crowd_receiver.adopted_solution_fitness_history else np.nan
            )

    # ------------------------------------------------------------
    # Dependent variables: receiver crowd only
    # ------------------------------------------------------------

    performance_list = [
        agent.fitness
        for agent in crowd_receiver.agents
    ]

    fitness_rank_list = [
        landscape.query_fitness_rank(state=agent.state)
        for agent in crowd_receiver.agents
    ]

    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)

    # Coverage / number of unique complete solutions.
    unique_solution_count = crowd_receiver.unique_solution_count()

    # Dispersion / average pairwise normalized Hamming distance.
    dispersion = crowd_receiver.calculate_dispersion()

    # Mean objective quality of solutions actually adopted from the sender crowd.
    adopted_solution_fitness = (
        np.mean(crowd_receiver.adopted_solution_fitness_history)
        if crowd_receiver.adopted_solution_fitness_history else np.nan
    )

    return_dict[loop] = [
        breakthrough_fitness,
        breakthrough_rank,
        unique_solution_count,
        dispersion,
        adopted_solution_fitness,
        adopted_solution_fitness_across_time,
    ]

    sema.release()


if __name__ == '__main__':
    import datetime

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    # ------------------------------------------------------------
    # Experiment parameters
    # ------------------------------------------------------------

    landscape_iteration = 400
    search_iteration = 300

    # Binary N=18 has 2^18 = 262,144 possible solutions,
    # matching the solution-space size of the previous four-state N=9 model.
    N = 18

    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Number of problem dimensions each solver can deliberately search/evaluate.
    knowledge_breadth = 6

    # Proportion of sender solvers whose current solutions are visible.
    visibility_extent_list = [
        0.0, 0.005, 0.01, 0.02, 0.04, 0.08,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1.0
    ]

    # Sender solutions are released every x periods.
    visibility_interval = 10

    agent_num = 200

    # N=18 requires substantially more memory than the old binary N=9 model.
    # Increase concurrency only after checking memory use on the cluster.
    concurrency = 4

    # ------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------

    for visibility_extent in visibility_extent_list:

        # DVs across K
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        unique_solution_count_across_K = []
        dispersion_across_K = []
        adopted_solution_fitness_across_K = []
        adopted_solution_fitness_across_time_across_K = []

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
                        N,
                        K,
                        agent_num,
                        knowledge_breadth,
                        search_iteration,
                        visibility_extent,
                        visibility_interval,
                        loop,
                        return_dict,
                        sema,
                    )
                )

                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            returns = list(return_dict.values())

            # First five outcomes are scalar measures.
            arr = np.asarray(
                [item[:5] for item in returns],
                dtype=float
            )

            # Sixth outcome is the adoption-quality trajectory.
            time_series_arr = np.asarray(
                [item[5] for item in returns],
                dtype=float
            )

            # First four DVs are always defined.
            means = arr[:, :4].mean(axis=0)

            # Adoption quality is undefined when no visible solution is adopted.
            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )

            # Average cumulative adoption-quality trajectory across repetitions.
            if np.all(np.isnan(time_series_arr)):
                adopted_solution_fitness_across_time_mean = time_series_arr[0].tolist()
            else:
                adopted_solution_fitness_across_time_mean = np.nanmean(
                    time_series_arr,
                    axis=0
                ).tolist()

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            unique_solution_count_across_K.append(means[2])
            dispersion_across_K.append(means[3])
            adopted_solution_fitness_across_K.append(
                adopted_solution_fitness_mean
            )
            adopted_solution_fitness_across_time_across_K.append(
                adopted_solution_fitness_across_time_mean
            )

        # ------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------

        with open(
            "visibility_extent_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open(
            "visibility_extent_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open(
            "visibility_extent_{0}_interval_{1}_unique_solution_count_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open(
            "visibility_extent_{0}_interval_{1}_dispersion_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(dispersion_across_K, out_file)

        with open(
            "visibility_extent_{0}_interval_{1}_adopted_solution_fitness_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open(
            "visibility_extent_{0}_interval_{1}_adopted_solution_fitness_across_time_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num
            ),
            'wb'
        ) as out_file:
            pickle.dump(
                adopted_solution_fitness_across_time_across_K,
                out_file
            )

    t1 = time.time()

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    print(
        "Visibility Extent with Interval {0}: ".format(visibility_interval),
        time.strftime("%H:%M:%S", time.gmtime(t1 - t0))
    )
