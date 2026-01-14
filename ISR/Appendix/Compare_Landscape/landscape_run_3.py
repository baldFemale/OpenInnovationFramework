# -*- coding: utf-8 -*-
# @Time     : 8/21/2023 19:35
# @Author   : Junyi
# @FileName: landscape_run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, state_num=None, alpha=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin", alpha=alpha)
    first_local_peak = landscape.count_first_local_optima()
    second_local_peak = landscape.count_second_local_optima()
    first_distance, second_distance = landscape.calculate_avg_fitness_distance()
    return_dict[loop] = [first_local_peak, second_local_peak, first_distance, second_distance]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 100
    N = 9
    state_num = 4
    alpha_list = [0.35, 0.40, 0.45, 0.50]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 100
    for alpha in alpha_list:
        # DVs
        first_local_peak_across_K = []
        second_local_peak_across_K = []
        first_distance_across_K = []
        second_distance_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, alpha, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_first_local_peak, temp_second_local_peak = [], []
            temp_first_distance, temp_second_distance = [], []
            for result in returns:  # 50 landscape repetitions
                temp_first_local_peak.append(result[0])
                temp_second_local_peak.append(result[1])
                temp_first_distance.append(result[2])
                temp_second_distance.append(result[3])

            first_local_peak_across_K.append(sum(temp_first_local_peak) / len(temp_first_local_peak))
            second_local_peak_across_K.append(sum(temp_second_local_peak) / len(temp_second_local_peak))
            first_distance_across_K.append(sum(temp_first_distance) / len(temp_first_distance))
            second_distance_across_K.append(sum(temp_second_distance) / len(temp_second_distance))

        with open("first_local_peak_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(first_local_peak_across_K, out_file)
        with open("second_local_peak_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(second_local_peak_across_K, out_file)
        with open("first_distance_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(first_distance_across_K, out_file)
        with open("second_distance_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(second_distance_across_K, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


