# -*- coding: utf-8 -*-
# @Time     : 5/15/2023 20:54
# @Author   : Junyi
# @FileName: landscape_run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Landscape import Landscape
from CogLandscape import CogLandscape
from BinaryLandscape import BinaryLandscape
import numpy as np
import pickle
import time
from multiprocessing import Semaphore
import multiprocessing as mp


def func(N=None, K=None, state_num=None, expertise_amount=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="Max")
    cog_landscape = CogLandscape(landscape=landscape, expertise_domain=np.random.choice(range(N), expertise_amount//2),
                                 expertise_representation=["A", "B"], norm="Max", collaborator="None")
    bin_landscape = BinaryLandscape(N=N, K=K, K_within=None, K_between=None, norm="Max")
    data = list(landscape.cache.values())
    cog_data = list(cog_landscape.cache.values())
    bin_data = list(bin_landscape.cache.values())
    return_dict[loop] = [data, cog_data, bin_data]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    state_num = 4
    expertise_domain_list = [18]
    repeat = 50
    concurrency = 50
    for K in K_list:
        for expertise_domain in expertise_domain_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(repeat):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, expertise_domain, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            results = return_dict.values()  # Don't need dict index, since it is repetition.
            data, cog_data, bin_data = [], [], []
            for result in results:
                data += result[0]
                cog_data += result[1]
                bin_data += result[2]
            with open("cache_K_{0}_E_{1}".format(K, expertise_domain), 'wb') as out_file:
                pickle.dump(data, out_file)
            with open("cog_cache_K_{0}_E_{1}".format(K, expertise_domain), 'wb') as out_file:
                pickle.dump(cog_data, out_file)
            with open("bin_cache_K_{0}_E_{1}".format(K, expertise_domain), 'wb') as out_file:
                pickle.dump(bin_data, out_file)

        t1 = time.time()
        print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))