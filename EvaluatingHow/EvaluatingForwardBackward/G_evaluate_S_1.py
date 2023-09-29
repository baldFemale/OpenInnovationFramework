# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
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
def func(N=None, K=None, state_num=None, individual_expertise=None, agent_num=None,
         search_iteration=None, roll_forward=None, roll_back=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    converged_performance_list = []
    converged_solution_list = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, crowd=crowd, specialist_expertise=individual_expertise)
        for _ in range(search_iteration):
            specialist.feedback_search(roll_back_ratio=roll_back, roll_forward_ratio=roll_forward)
        converged_performance_list.append(specialist.fitness)
        converged_solution_list.append(specialist.state)
    average_performance = sum(converged_performance_list) / len(converged_performance_list)
    best_performance = max(converged_performance_list)
    variance = np.std(converged_performance_list)
    unique_diversity = get_unique_diversity(belief_pool=converged_solution_list)
    pair_wise_diversity = get_pair_wise_diversity(belief_pool=converged_solution_list)
    return_dict[loop] = [average_performance, best_performance, variance, unique_diversity, pair_wise_diversity]
    sema.release()

def get_unique_diversity(belief_pool: list):
    unique_solutions = []
    for belief in belief_pool:
        string_belief = "".join(belief)
        unique_solutions.append(string_belief)
    unique_solutions = set(unique_solutions)
    return len(unique_solutions)

def get_pair_wise_diversity(belief_pool: list):
    diversity = 0
    for index, focal_belief in enumerate(belief_pool):
        selected_pool = belief_pool[index + 1::]
        one_pair_diversity = [get_distance(focal_belief, belief) for belief in selected_pool]
        diversity += sum(one_pair_diversity)
    return diversity / len(belief_pool[0]) / (len(belief_pool) - 1) / len(belief_pool) * 2

def get_distance(self, a=None, b=None):
    acc = 0
    for i in range(self.m):
        if a[i] != b[i]:
            acc += 1
    return acc

# def get_within_domain_diversity(belief_pool: list, domain_list: list):
#     # shift into string
#     str_belief_list = []
#     for belief in belief_pool:
#         str_belief = "".join(belief)
#         str_belief_list.append(str_belief)
#     str_domain_list = []
#     for domain in domain_list:
#         str_domain = "".join(domain)
#         str_domain_list.append(str_domain)
#     unique_str_domain_list = set(str_domain_list)
#
#     cutted_belief_list = []
#     for belief, domain in zip(belief_pool, domain_list):
#         cutted_belief = [belief[i] for i in domain]


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    individual_expertise = 12
    K_list = [0, 4, 8]
    # roll_forward_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    roll_forward_list = [0, 0.1]
    roll_back_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    concurrency = 50
    for roll_forward in roll_forward_list:
        for roll_back in roll_back_list:
            # DVs
            performance_across_K = []
            best_performance_across_K = []
            variance_across_K = []
            unique_diversity_across_K = []
            pair_wise_diversity_across_K = []
            for K in K_list:
                manager = mp.Manager()
                return_dict = manager.dict()
                sema = Semaphore(concurrency)
                jobs = []
                for loop in range(landscape_iteration):
                    sema.acquire()
                    p = mp.Process(target=func, args=(N, K, state_num, individual_expertise,
                                                      agent_num, search_iteration, roll_forward, roll_back, loop, return_dict, sema))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()
                returns = return_dict.values()  # Don't need dict index, since it is repetition.

                temp_average, temp_best, temp_variance, temp_unique, temp_pair_wise = [], [], [], [], []
                for result in returns:  # 50 landscape repetitions
                    temp_average.append(result[0])
                    temp_best.append(result[1])
                    temp_variance.append(result[2])
                    temp_unique.append(result[3])
                    temp_pair_wise.append(result[4])

                performance_across_K.append(sum(temp_average) / len(temp_average))
                best_performance_across_K.append(sum(temp_best) / len(temp_best))
                variance_across_K.append(sum(temp_variance) / len(temp_variance))
                unique_diversity_across_K.append(sum(temp_unique) / len(temp_unique))
                pair_wise_diversity_across_K.append(sum(temp_pair_wise) / len(temp_pair_wise))

            # remove time dimension
            with open("gs_performance_across_K_forward_{0}_backward_{1}".format(roll_forward, roll_back), 'wb') as out_file:
                pickle.dump(performance_across_K, out_file)
            with open("gs_best_performance_across_K_forward_{0}_backward_{1}".format(roll_forward, roll_back), 'wb') as out_file:
                pickle.dump(best_performance_across_K, out_file)
            with open("gs_variance_across_K_forward_{0}_backward_{1}".format(roll_forward, roll_back), 'wb') as out_file:
                pickle.dump(variance_across_K, out_file)
            with open("gs_unique_diversity_across_K_forward_{0}_backward_{1}".format(roll_forward, roll_back), 'wb') as out_file:
                pickle.dump(unique_diversity_across_K, out_file)
            with open("gs_pair_wise_diversity_across_K_forward_{0}_backward_{1}".format(roll_forward, roll_back), 'wb') as out_file:
                pickle.dump(pair_wise_diversity_across_K, out_file)

    t1 = time.time()
    print("GS_1: ", time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
