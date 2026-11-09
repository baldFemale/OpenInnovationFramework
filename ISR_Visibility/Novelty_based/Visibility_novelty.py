#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_novelty.py
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
         visibility_extent=None, novelty_share_ratio=None, visibility_interval=10, solver_reset_num=1,
         loop=None, return_dict=None, sema=None):
    """
    Novelty-based visibility with separated sender and recipient crowds.

    visibility_extent fixes which senders are eligible throughout a run.
    novelty_share_ratio is the fraction of eligible sender solutions recommended
    separately to each recipient at every visibility event. Solutions are selected
    by their Hamming distance from that recipient's current complete configuration:
    larger distances first (more different configurations).

        novelty_share_ratio = 0.0: no exposure (search-apart benchmark).
        novelty_share_ratio = 0.05: 5 recommendations per recipient with 100 senders.
        novelty_share_ratio = 1.0: all eligible sender solutions are exposed.

    Both crowds conduct local search every period. Every visibility_interval
    periods, each recipient's recommendations are recomputed using the current
    states after local search. Ties in Hamming distance are broken randomly,
    and the selected recommendations are shuffled before evaluation. Recipients
    evaluate complete solutions with their own bounded knowledge and adopt at
    most the first perceived improvement, without feedback to senders.

    All DVs concern the recipient crowd. Adoption-fitness trajectories contain
    a cumulative mean at each visibility event, or np.nan before any adoption.
    solver_reset_num reruns independent crowds on each shared NK landscape.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")
        if novelty_share_ratio is None or not 0 <= novelty_share_ratio <= 1:
            raise ValueError("novelty_share_ratio must be between 0 and 1.")

        solver_reset_num = int(solver_reset_num)
        if solver_reset_num < 1:
            raise ValueError("solver_reset_num must be a positive integer.")

        # Generate the landscape once, then restart the crowds on the same landscape.
        landscape = Landscape(N=N, K=K)
        solver_reset_returns = []

        for solver_reset in range(solver_reset_num):
            # Sender crowd: solvers who independently search and make solutions visible
            crowd_sender = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                 landscape=landscape)

            # Receiver crowd: solvers who independently search and learn from visible sender solutions
            crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                   landscape=landscape)

            crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

            # Search-process trajectory: cumulative mean true fitness of adopted solutions.
            adopted_solution_fitness_across_time = []

            for period in range(search_iteration):
                # Both crowds conduct their own independent search.
                crowd_sender.search()
                crowd_receiver.search()

                if (period + 1) % visibility_interval == 0:
                    # All structurally visible senders are eligible for recommendation.
                    sender_pool = [agent.state.copy() for agent in crowd_sender.agents
                                   if agent.visibility_status]
                    visible_num = int(round(novelty_share_ratio * len(sender_pool)))

                    if visible_num > 0:
                        # When every sender is disclosed, ranking is unnecessary.
                        if visible_num < len(sender_pool):
                            # Hamming distance uses each recipient's current solution,
                            # measured before this exposure event.
                            sender_states = np.asarray(sender_pool, dtype=np.uint8)
                            recipient_states = np.asarray(
                                [agent.state for agent in crowd_receiver.agents], dtype=np.uint8
                            )
                            distances = np.count_nonzero(
                                recipient_states[:, None, :] != sender_states[None, :, :], axis=2
                            )

                        for recipient_index, recipient in enumerate(crowd_receiver.agents):
                            # Random permutation breaks ties in recommendation scores.
                            candidate_indices = np.random.permutation(len(sender_pool))
                            if visible_num < len(sender_pool):
                                candidate_indices = candidate_indices[np.argsort(
                                    -distances[recipient_index, candidate_indices], kind='stable'
                                )]
                            selected_indices = candidate_indices[:visible_num].copy()

                            # Ranking selects the candidates; evaluation order is random.
                            np.random.shuffle(selected_indices)
                            for index in selected_indices:
                                if recipient.search(solution=sender_pool[index]):
                                    crowd_receiver.adopted_solution_fitness_history.append(recipient.fitness)
                                    break

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

            solver_reset_returns.append([
                breakthrough_fitness,
                breakthrough_rank,
                unique_solution_count,
                pairwise_diversity,
                adopted_solution_fitness,
                adopted_solution_fitness_across_time,
            ])

        # One landscape repetition returns solver_reset_num sets of outcomes.
        return_dict[loop] = solver_reset_returns
    finally:
        sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 600
    solver_reset_num = 10  # Fresh sender/receiver crowds per landscape. Set to 1 for the original design.
    search_iteration = 300

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; Upper-complexity half

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # All sender paths are eligible; recommendations are personalized to each recipient.
    visibility_extent = 1.0

    # Share of eligible sender solutions recommended per recipient by novelty.
    # With 100 senders: 0.0 recommends none, 0.05 recommends 5, 1.0 recommends all.
    novelty_share_ratio_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Sender solutions are disclosed every visibility_interval periods.
    visibility_interval = 10

    # Recommendations are ranked for each recipient independently.
    agent_num = 100
    concurrency = 100

    for novelty_share_ratio in novelty_share_ratio_list:
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
                    # Close completed processes as the experiment runs.
                    # Otherwise, keeping all 600 Process objects open can
                    # exhaust the operating system's file-descriptor limit.
                    while True:
                        acquired = sema.acquire(timeout=0.5)

                        for index, proc in jobs[:]:
                            if proc.exitcode is not None:
                                proc.join()
                                exitcode = proc.exitcode
                                proc.close()
                                jobs.remove((index, proc))

                                if exitcode != 0:
                                    raise RuntimeError(
                                        "Repetition {0} failed with exit code {1}".format(index, exitcode)
                                    )

                        if acquired:
                            break

                    p = mp.Process(target=func, args=(N, K, agent_num, knowledge_breadth, search_iteration,
                                                      visibility_extent, novelty_share_ratio, visibility_interval, solver_reset_num,
                                                      loop, return_dict, sema))
                    p.start()
                    jobs.append((loop, p))

                # Collect and close the processes that are still running.
                while jobs:
                    for index, proc in jobs[:]:
                        if proc.exitcode is not None:
                            proc.join()
                            exitcode = proc.exitcode
                            proc.close()
                            jobs.remove((index, proc))

                            if exitcode != 0:
                                raise RuntimeError(
                                    "Repetition {0} failed with exit code {1}".format(index, exitcode)
                                )

                    if jobs:
                        time.sleep(0.5)

                if len(return_dict) != landscape_iteration:
                    raise RuntimeError("Missing results: expected {0}, received {1}".format(
                        landscape_iteration, len(return_dict)))

                landscape_returns = [return_dict[index] for index in range(landscape_iteration)]
            finally:
                for index, proc in jobs:
                    if proc.is_alive():
                        proc.terminate()
                for index, proc in jobs:
                    proc.join()
                    proc.close()
                manager.shutdown()

            # Flatten the two repetition layers into individual runs for averaging.
            # Each landscape contributes the same number of solver resets.
            returns = [item for landscape_results in landscape_returns for item in landscape_results]
            if len(returns) != landscape_iteration * solver_reset_num:
                raise RuntimeError("Missing solver resets: expected {0}, received {1}".format(
                    landscape_iteration * solver_reset_num, len(returns)))

            arr = np.asarray([item[:5] for item in returns], dtype=float)  # shape: (landscape_iteration * solver_reset_num, 5)
            time_series_arr = np.asarray([item[5] for item in returns], dtype=float)

            # The first four DVs are always defined. The visibility-process
            # measure can be np.nan when a run contains no adoption.
            means = arr[:, :4].mean(axis=0)

            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )

            # Average the cumulative adoption-quality trajectory across landscapes and solver resets.
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

        # Save results across K for each novelty-share ratio and visibility interval.
        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_unique_solution_count_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_adopted_solution_fitness_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("novelty_based_visibility_share_ratio_{0}_interval_{1}_adopted_solution_fitness_across_time_across_K_size_{2}".format(
                novelty_share_ratio, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Novelty-Based Visibility with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
