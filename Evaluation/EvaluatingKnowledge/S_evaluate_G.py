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
def func(N=None, K=None, state_num=None, individual_expertise=None, crowd_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=crowd_expertise, label="S")
    converged_performance_list = []
    converged_solution_list = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, crowd=crowd, generalist_expertise=individual_expertise)
        for _ in range(search_iteration):
            generalist.feedback_search(roll_back_ratio=0.5, roll_forward_ratio=0.5)
        converged_performance_list.append(generalist.fitness)
        converged_solution_list.append(generalist.state)
    average_performance = sum(converged_performance_list) / len(converged_performance_list)
    best_performance = max(converged_performance_list)
    variance = np.std(converged_performance_list)
    diversity = get_diversity(belief_pool=converged_solution_list)
    return_dict[loop] = [average_performance, best_performance, variance, diversity]
    sema.release()

def get_diversity(belief_pool: list):
    unique_solutions = []
    for belief in belief_pool:
        string_belief = "".join(belief)
        unique_solutions.append(string_belief)
    unique_solutions = set(unique_solutions)
    return len(unique_solutions)


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    individual_expertise = 12
    crowd_knowledge_list = [12, 16, 20, 24, 32, 36]
    # K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    for crowd_knowledge in crowd_knowledge_list:
        # DVs
        average_performance_across_K = []
        best_performance_across_K = []
        variance_across_K = []
        diversity_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, individual_expertise, crowd_knowledge,
                                                  agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_average, temp_best, temp_variance, temp_diversity = [], [], [], []
            for result in returns:  # 50 landscape repetitions
                temp_average.append(result[0])
                temp_best.append(result[1])
                temp_variance.append(result[2])
                temp_diversity.append(result[3])

            average_performance_across_K.append(sum(temp_average) / len(temp_average))
            best_performance_across_K.append(sum(temp_best) / len(temp_best))
            variance_across_K.append(sum(temp_variance) / len(temp_variance))
            diversity_across_K.append(sum(temp_diversity) / len(temp_diversity))

        # remove time dimension
        with open("sg_performance_across_K_{0}".format(crowd_knowledge), 'wb') as out_file:
            pickle.dump(average_performance_across_K, out_file)
        with open("sg_best_performance_across_K_{0}".format(crowd_knowledge), 'wb') as out_file:
            pickle.dump(best_performance_across_K, out_file)
        with open("sg_variance_across_K_{0}".format(crowd_knowledge), 'wb') as out_file:
            pickle.dump(variance_across_K, out_file)
        with open("sg_diversity_across_K_{0}".format(crowd_knowledge), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    print("Evaluating SG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


