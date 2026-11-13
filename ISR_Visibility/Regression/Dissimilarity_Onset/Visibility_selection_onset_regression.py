#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName : Visibility_selection_onset_regression.py
# @Software : PyCharm
# Observing PEP 8 coding style

import csv
import datetime
import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import Semaphore

import numpy as np
from Landscape import Landscape
from Crowd import Crowd


# mp version
def func(N=None, K=None, agent_num=None, knowledge_breadth=None, search_iteration=None,
         visibility_extent=1.0, selection_rule=None, selectivity=None, recommendation_num=10,
         visibility_start=0, visibility_interval=10, solver_reset_num=1,
         loop=None, return_dict=None, sema=None):
    """
    Selection x Onset experiment with separated sender and receiver crowds.

    The selection rule is Performance, Similarity, or Dissimilarity.
    Selectivity determines how many of the 10 recommendations follow that
    rule; the remaining recommendations are selected randomly from other
    sender entries. Selectivity = 0 gives 10 random recommendations, not
    search-apart. Visibility extent is fixed at 1.0.

    Visibility begins at the zero-based visibility_start period, then recurs
    every visibility_interval periods, following each period's local search.
    Each receiver gets exactly recommendation_num sender-solution entries
    per visibility event and adopts at most one perceived improvement.

    Senders and receivers search separately on the same landscape; receiver
    solutions do not feed back into the sender pool. All outcomes are
    measured on receivers, with each solver reset returned separately.
    """
    try:
        np.random.seed(None)

        if selection_rule not in ("Performance", "Similarity", "Dissimilarity"):
            raise ValueError("Unknown selection rule: {0}".format(selection_rule))
        if selectivity is None or not 0 <= selectivity <= 1:
            raise ValueError("selectivity must be between 0 and 1.")
        if visibility_extent != 1.0:
            raise ValueError("visibility_extent must be 1.0.")
        if recommendation_num != 10:
            raise ValueError("recommendation_num must be 10.")
        if agent_num < recommendation_num:
            raise ValueError("Sender pool must contain at least 10 agents.")
        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")
        if not 0 <= visibility_start < search_iteration:
            raise ValueError("visibility_start must be a zero-based search period.")

        solver_reset_num = int(solver_reset_num)
        if solver_reset_num < 1:
            raise ValueError("solver_reset_num must be a positive integer.")

        # Keep exactly 10 recommendations regardless of selection strength.
        rule_selection_num = int(round(selectivity * recommendation_num))
        random_selection_num = recommendation_num - rule_selection_num

        # Generate one landscape; run independent crowds on the same landscape.
        landscape = Landscape(N=N, K=K)
        solver_reset_returns = []

        for solver_reset in range(solver_reset_num):
            crowd_sender = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                 landscape=landscape)
            crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                   landscape=landscape)
            crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

            # Search-process trajectory: cumulative mean true fitness of adopted solutions.
            adopted_solution_fitness_across_time = []

            for period in range(search_iteration):
                # Both crowds conduct their own independent search.
                crowd_sender.search()
                crowd_receiver.search()

                if (period >= visibility_start) and ((period - visibility_start) % visibility_interval == 0):
                    sender_pool = [agent.state.copy() for agent in crowd_sender.agents
                                   if agent.visibility_status]
                    if len(sender_pool) < recommendation_num:
                        raise ValueError("Eligible sender pool is smaller than 10.")

                    # Performance ranks are shared across receivers; Hamming-distance
                    # rankings for Similarity and Dissimilarity are receiver-specific.
                    if rule_selection_num > 0:
                        if selection_rule == "Performance":
                            fitness_ranks = np.asarray([
                                landscape.query_fitness_rank(state=state) for state in sender_pool
                            ])
                        else:
                            sender_states = np.asarray(sender_pool, dtype=np.uint8)
                            recipient_states = np.asarray(
                                [agent.state for agent in crowd_receiver.agents], dtype=np.uint8
                            )
                            distances = np.count_nonzero(
                                recipient_states[:, None, :] != sender_states[None, :, :], axis=2
                            )

                    for recipient_index, recipient in enumerate(crowd_receiver.agents):
                        # Random permutation breaks ties; stable sort keeps that order.
                        candidate_indices = np.random.permutation(len(sender_pool))
                        if rule_selection_num > 0:
                            if selection_rule == "Performance":
                                candidate_indices = candidate_indices[np.argsort(
                                    fitness_ranks[candidate_indices], kind='stable'
                                )]
                            elif selection_rule == "Similarity":
                                candidate_indices = candidate_indices[np.argsort(
                                    distances[recipient_index, candidate_indices], kind='stable'
                                )]
                            else:  # Dissimilarity: prefer more distant sender solutions.
                                candidate_indices = candidate_indices[np.argsort(
                                    -distances[recipient_index, candidate_indices], kind='stable'
                                )]

                        # Select the strongest matches first, then fill the remaining
                        # positions randomly WITHOUT replacement from other senders.
                        selected_indices = candidate_indices[:rule_selection_num].copy()
                        if random_selection_num > 0:
                            random_indices = np.random.choice(
                                candidate_indices[rule_selection_num:],
                                size=random_selection_num, replace=False
                            )
                            selected_indices = np.concatenate((selected_indices, random_indices))

                        if len(selected_indices) != recommendation_num:
                            raise RuntimeError("Recommendation count must be exactly 10.")

                        # Selection determines pool membership, not evaluation order.
                        np.random.shuffle(selected_indices)
                        for index in selected_indices:
                            if recipient.search(solution=sender_pool[index]):
                                crowd_receiver.adopted_solution_fitness_history.append(recipient.fitness)
                                break

                    adopted_solution_fitness_across_time.append(
                        np.mean(crowd_receiver.adopted_solution_fitness_history)
                        if crowd_receiver.adopted_solution_fitness_history else np.nan
                    )

            # All dependent variables are measured on the receiver crowd.
            performance_list = [agent.fitness for agent in crowd_receiver.agents]
            fitness_rank_list = [
                landscape.query_fitness_rank(state=agent.state)
                for agent in crowd_receiver.agents
            ]

            breakthrough_fitness = max(performance_list)
            breakthrough_rank = min(fitness_rank_list)  # Smaller rank is better; rank 1 is global best.

            # Number of unique complete solutions among receivers.
            full_solution_set = set()
            for agent in crowd_receiver.agents:
                solution_str = "".join([str(bit) for bit in agent.state])
                full_solution_set.add(solution_str)
            unique_solution_count = len(full_solution_set)

            # Average pairwise normalized Hamming distance among receivers.
            pairwise_diversity = crowd_receiver.calculate_pairwise_solution_distance()

            # Search-process measure, using true fitness rather than perceived fitness.
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

        # Preserve the two repetition levels: landscape repetition and solver reset.
        return_dict[loop] = solver_reset_returns
    finally:
        sema.release()


if __name__ == '__main__':
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 200
    solver_reset_num = 5  # Independent sender/receiver crowds per landscape.
    search_iteration = 300

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; upper-complexity half.

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # Factorial interaction design: 3 selection rules x 6 strengths x 14 onsets.
    selection_rule_list = ["Performance", "Similarity", "Dissimilarity"]
    selectivity_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    visibility_start_list = [0, 10, 20, 30, 40, 50, 75, 100, 200, 250, 260, 270, 280, 290]

    # Fix extent and recommendation quantity for all selection conditions.
    visibility_extent = 1.0
    recommendation_num = 10
    visibility_interval = 10

    # Use 100 agents in each sender and receiver crowd.
    agent_num = 100
    concurrency = 100

    # Fresh output folder for each invocation avoids mixing separate runs.
    # The main CSV has one row per solver reset and is ready for regression.
    # The separate pickle files retain each run's original visibility-event trajectory.
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "visibility_selection_onset_results_{0}".format(now.strftime("%Y%m%d_%H%M%S"))
    )
    trajectory_dir = os.path.join(output_dir, "adoption_trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "visibility_selection_onset_runs.csv")

    fieldnames = [
        "SelectionRule", "Selectivity", "Onset", "K", "landscape_id",
        "landscape_repetition", "solver_reset",
        "breakthrough_fitness", "breakthrough_rank", "unique_solution_count",
        "pairwise_diversity", "adopted_solution_fitness",
    ]
    landscape_id_counter = 0
    row_count = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for selection_rule in selection_rule_list:
            for selectivity in selectivity_list:
                for visibility_start in visibility_start_list:
                    for K in K_list:
                        manager = mp.Manager()
                        return_dict = manager.dict()
                        sema = Semaphore(concurrency)
                        jobs = []

                        try:
                            for loop in range(landscape_iteration):
                                # Close completed processes as the experiment runs to avoid
                                # exhausting the operating system's file-descriptor limit.
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

                                p = mp.Process(
                                    target=func,
                                    args=(N, K, agent_num, knowledge_breadth, search_iteration,
                                          visibility_extent, selection_rule, selectivity, recommendation_num,
                                          visibility_start, visibility_interval, solver_reset_num,
                                          loop, return_dict, sema)
                                )
                                p.start()
                                jobs.append((loop, p))

                            # Collect and close the remaining processes.
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
                                raise RuntimeError("Missing landscapes: expected {0}, received {1}".format(
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

                        # Save every outcome from every solver reset; do NOT average across
                        # landscapes, solver resets, or levels of K.
                        trajectories = []
                        expected_events = len(range(visibility_start, search_iteration, visibility_interval))
                        for landscape_repetition, solver_results in enumerate(landscape_returns):
                            if len(solver_results) != solver_reset_num:
                                raise RuntimeError(
                                    "Missing solver resets at landscape {0}: expected {1}, received {2}".format(
                                        landscape_repetition, solver_reset_num, len(solver_results))
                                )

                            # Each independent landscape is assigned one globally unique ID;
                            # its solver resets share that ID (useful for clustered SEs).
                            landscape_id = landscape_id_counter
                            landscape_id_counter += 1

                            for solver_reset, result in enumerate(solver_results):
                                if len(result) != 6 or len(result[5]) != expected_events:
                                    raise RuntimeError(
                                        "Invalid outcomes/trajectory at landscape {0}, reset {1}".format(
                                            landscape_repetition, solver_reset)
                                    )

                                writer.writerow({
                                    "SelectionRule": selection_rule,
                                    "Selectivity": selectivity,
                                    "Onset": visibility_start,
                                    "K": K,
                                    "landscape_id": landscape_id,
                                    "landscape_repetition": landscape_repetition,
                                    "solver_reset": solver_reset,
                                    "breakthrough_fitness": result[0],
                                    "breakthrough_rank": result[1],
                                    "unique_solution_count": result[2],
                                    "pairwise_diversity": result[3],
                                    "adopted_solution_fitness": result[4],
                                })
                                row_count += 1
                                trajectories.append({
                                    "landscape_id": landscape_id,
                                    "landscape_repetition": landscape_repetition,
                                    "solver_reset": solver_reset,
                                    "adopted_solution_fitness_across_time": result[5],
                                })

                        # Keep the trajectory (a variable-length array) outside the CSV
                        # so the primary data remain compact and rectangular for regression.
                        trajectory_path = os.path.join(
                            trajectory_dir,
                            "selection_{0}_strength_{1:.1f}_onset_{2}_K_{3}.pkl".format(
                                selection_rule.lower(), selectivity, visibility_start, K)
                        )
                        with open(trajectory_path, 'wb') as out_file:
                            pickle.dump({
                                "SelectionRule": selection_rule,
                                "Selectivity": selectivity,
                                "Onset": visibility_start,
                                "K": K,
                                "visibility_interval": visibility_interval,
                                "trajectories": trajectories,
                            }, out_file, protocol=pickle.HIGHEST_PROTOCOL)

                        # Flush each completed configuration, preserving progress if interrupted.
                        csv_file.flush()
                        print("Saved SelectionRule={0}, Selectivity={1:.1f}, Onset={2}, K={3}: {4} runs".format(
                            selection_rule, selectivity, visibility_start, K,
                            landscape_iteration * solver_reset_num), flush=True)

    expected_count = (len(selection_rule_list) * len(selectivity_list) *
                      len(visibility_start_list) * len(K_list) *
                      landscape_iteration * solver_reset_num)
    if row_count != expected_count:
        raise RuntimeError("Missing observations: expected {0}, saved {1}".format(
            expected_count, row_count))

    t1 = time.time()
    print("Regression data: {0}".format(csv_path))
    print("Adoption trajectories: {0}".format(trajectory_dir))
    print("Total run-level observations: {0}".format(row_count))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Visibility Selection x Onset with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
