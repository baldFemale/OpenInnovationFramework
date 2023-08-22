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
    first_cache = landscape.first_cache.values()
    second_cache = landscape.second_cache.values()
    return_dict[loop] = [first_cache, second_cache]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 50
    N = 10
    state_num = 4
    alpha_list = [0.35, 0.40, 0.45, 0.50]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    concurrency = 50
    # DVs
    first_cache_across_K = []
    second_cache_across_K = []
    for alpha in alpha_list:
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

            temp_first_cache, temp_second_cache = [], []
            for result in returns:  # 50 landscape repetitions
                temp_first_cache.extend(result[0])
            first_cache_across_K.append(temp_first_cache)
            second_cache_across_K.append(returns[-1][1])
        with open("first_cache_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(first_cache_across_K, out_file)
        with open("second_local_peak_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(second_cache_across_K, out_file)

        import matplotlib.pyplot as plt
        for K, first_cache, second_cache in zip(K_list, first_cache_across_K, second_cache_across_K):
            plt.hist(first_cache, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.title("First Cache N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.xlabel("Range")
            plt.ylabel("Count")
            plt.savefig("First_N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.clf()

            plt.hist(second_cache, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.title("Second Cache N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.xlabel("Range")
            plt.ylabel("Count")
            plt.savefig("Second_N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.clf()

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


