#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_rank.py
# @Software : PyCharm
# Observing PEP 8 coding style

import numpy as np
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, agent_num=None, knowledge_breadth=None, search_iteration=None,
         visibility_extent=None, rank_share_ratio=None, visibility_interval=10,
         loop=None, return_dict=None, sema=None):
    """
    Rank-based visibility experiment with separated sender and receiver crowds.

    visibility_extent fixes the proportion of eligible sender agents for the run.
    rank_share_ratio determines the proportion of eligible sender solutions
    released at each visibility event, selected from highest to lowest true
    fitness rank (rank 1 is the best solution in the complete NK landscape).

        rank_share_ratio = 0.0: no solutions released (independent-search baseline).
        rank_share_ratio = 0.005 with 200 senders: the highest-ranked solution.
        rank_share_ratio = 1.0: solutions of every available rank are released.

    Sender solutions are re-ranked at every visibility event. Both crowds
    search every period; disclosure occurs every visibility_interval periods.
    Each receiver samples one disclosed solution with probability proportional
    to its true fitness and imitates it directly, as in the reference design.

    Receiver solutions do not feed back into the sender pool. All dependent
    variables concern the receiver crowd. Adoption-quality trajectories contain
    one cumulative mean per visibility event, including np.nan before imitation.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")
        if rank_share_ratio is None or not 0 <= rank_share_ratio <= 1:
            raise ValueError("rank_share_ratio must be between 0 and 1.")

        landscape = Landscape(N=N, K=K)

        # Sender crowd: solvers who independently search and make solutions visible
        crowd_sender = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                             landscape=landscape)

        # Receiver crowd: solvers who independently search and learn from visible sender solutions
        crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                               landscape=landscape)

        crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

        # Search-process trajectory: cumulative mean true fitness of imitated solutions.
        adopted_solution_fitness_across_time = []

        for period in range(search_iteration):
            # Both crowds conduct their own independent search.
            crowd_sender.search()
            crowd_receiver.search()

            if (period + 1) % visibility_interval == 0:
                # Rank the eligible sender solutions by their TRUE landscape fitness rank.
                # Lower rank numbers indicate better solutions (rank 1 = global best).
                ranked_pool = []
                for agent in crowd_sender.agents:
                    if agent.visibility_status:
                        fitness_rank = landscape.query_fitness_rank(state=agent.state)
                        ranked_pool.append([agent.state.copy(), agent.fitness, fitness_rank])

                # Randomize ties, then select the top fraction of available senders.
                np.random.shuffle(ranked_pool)
                ranked_pool.sort(key=lambda item: item[2])
                visible_num = int(round(rank_share_ratio * len(ranked_pool)))
                visible_pool = ranked_pool[:visible_num]

                if visible_pool:
                    fitness_values = np.asarray(
                        [fitness for _, fitness, _ in visible_pool], dtype=float
                    )
                    # Preserve fitness-proportional imitation within the released pool.
                    imitation_weights = fitness_values + 1e-10
                    imitation_probs = imitation_weights / imitation_weights.sum()

                    for agent in crowd_receiver.agents:
                        selected_index = np.random.choice(
                            len(visible_pool), p=imitation_probs
                        )
                        solution, _, _ = visible_pool[selected_index]

                        # Direct imitation, followed by evaluation with the receiver's knowledge.
                        agent.state = solution.copy()
                        agent.cog_fitness = agent.landscape.query_scoped_fitness(
                            state=agent.state, knowledge_domain=agent.knowledge_domain
                        )
                        agent.fitness = agent.landscape.query_fitness(state=agent.state)
                        crowd_receiver.adopted_solution_fitness_history.append(agent.fitness)

                adopted_solution_fitness_across_time.append(
                    np.mean(crowd_receiver.adopted_solution_fitness_history)
                    if crowd_receiver.adopted_solution_fitness_history else np.nan
                )

        # DVs are measured only on the receiver crowd.
        performance_list = [agent.fitness for agent in crowd_receiver.agents]
        fitness_rank_list = [
            landscape.query_fitness_rank(state=agent.state)
            for agent in crowd_receiver.agents
        ]

        breakthrough_fitness = max(performance_list)
        breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

        # Calculate the number of unique complete solutions among receiver agents.
        full_solution_set = set()
        for agent in crowd_receiver.agents:
            solution_str = "".join([str(bit) for bit in agent.state])
            full_solution_set.add(solution_str)

        unique_solution_count = len(full_solution_set)

        # Average pairwise normalized Hamming distance among receiver agents.
        pairwise_diversity = crowd_receiver.calculate_pairwise_solution_distance()

        # Search-process measure for visibility.
        # Use true fitness rather than perceived fitness.
        adopted_solution_fitness = (
            np.mean(crowd_receiver.adopted_solution_fitness_history)
            if crowd_receiver.adopted_solution_fitness_history else np.nan
        )

        return_dict[loop] = [
            breakthrough_fitness,
            breakthrough_rank,
            unique_solution_count,
            pairwise_diversity,
            adopted_solution_fitness,
            adopted_solution_fitness_across_time,
        ]
    finally:
        sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 400
    search_iteration = 300

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; Upper-complexity half

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # All sender paths are eligible; fitness rank determines which are released.
    visibility_extent = 1.0

    # Fraction of sender solutions released, ordered by true fitness rank.
    # With 200 senders: 0.0 releases none, 0.005 releases the top 1, 1.0 releases all.
    rank_share_ratio_list = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08,
                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Sender solutions are disclosed every visibility_interval periods.
    visibility_interval = 10

    # Restore the original crowd size; rank_share_ratio=0.005 selects the top sender.
    agent_num = 200
    concurrency = 100

    for rank_share_ratio in rank_share_ratio_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        unique_solution_count_across_K = []
        pairwise_diversity_across_K = []
        adopted_solution_fitness_across_K = []
        adopted_solution_fitness_across_time_across_K = []

        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []

            try:
                for loop in range(landscape_iteration):
                    # Check failed workers while waiting, including workers that
                    # were killed before they could release their semaphore.
                    while True:
                        acquired = sema.acquire(timeout=0.5)
                        failed_jobs = [(index, proc.exitcode)
                                       for index, proc in enumerate(jobs)
                                       if proc.exitcode not in (None, 0)]
                        if failed_jobs:
                            raise RuntimeError("Failed repetitions: {0}".format(failed_jobs))
                        if acquired:
                            break

                    p = mp.Process(target=func, args=(N, K, agent_num, knowledge_breadth, search_iteration,
                                                      visibility_extent, rank_share_ratio, visibility_interval,
                                                      loop, return_dict, sema))
                    p.start()
                    jobs.append(p)

                for proc in jobs:
                    while True:
                        proc.join(timeout=0.5)
                        failed_jobs = [(index, job.exitcode)
                                       for index, job in enumerate(jobs)
                                       if job.exitcode not in (None, 0)]
                        if failed_jobs:
                            raise RuntimeError("Failed repetitions: {0}".format(failed_jobs))
                        if not proc.is_alive():
                            break

                returns = list(return_dict.values())  # Repetition order does not affect the averages.
            finally:
                for proc in jobs:
                    if proc.is_alive():
                        proc.terminate()
                for proc in jobs:
                    proc.join()
                    proc.close()
                manager.shutdown()

            arr = np.asarray([item[:5] for item in returns], dtype=float)  # shape: (n_runs, 5)
            time_series_arr = np.asarray([item[5] for item in returns], dtype=float)

            # The first four DVs are always defined. The visibility-process
            # measure can be np.nan when a run contains no adoption.
            means = arr[:, :4].mean(axis=0)

            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )

            # Average the cumulative adoption-quality trajectory across landscape repetitions.
            if np.all(np.isnan(time_series_arr)):
                adopted_solution_fitness_across_time_mean = time_series_arr[0].tolist()
            else:
                adopted_solution_fitness_across_time_mean = np.nanmean(time_series_arr, axis=0).tolist()

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            unique_solution_count_across_K.append(means[2])
            pairwise_diversity_across_K.append(means[3])
            adopted_solution_fitness_across_K.append(adopted_solution_fitness_mean)
            adopted_solution_fitness_across_time_across_K.append(
                adopted_solution_fitness_across_time_mean
            )

        # Save results across K for each rank-share ratio and visibility interval.
        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_unique_solution_count_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_adopted_solution_fitness_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("rank_based_visibility_share_ratio_{0}_interval_{1}_adopted_solution_fitness_across_time_across_K_size_{2}".format(
                rank_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Rank-Based Visibility with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
