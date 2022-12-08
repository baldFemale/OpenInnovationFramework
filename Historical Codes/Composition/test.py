# -*- coding: utf-8 -*-
# @Time     : 9/24/2022 21:10
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy
import numpy as np
from multiprocessing import Semaphore
import multiprocessing as mp

# Test whether the random number across repetitions are the same
def func():
    random_number = np.random.uniform(0, 1)
    print(random_number)

if __name__ == '__main__':
    for index in range(10):
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(100)
        jobs = []
        for loop in range(30):
            sema.acquire()
            p = mp.Process(target=func)
            jobs.append(p)
            p.start()
        print("*" * 10, index)
# def func(divergence=None):
#     state_pool = []
#     seed_state = np.random.choice(range(4), 6).tolist()
#     seed_state = [str(i) for i in seed_state]  # state format: string
#     print(seed_state)
#     if divergence == 1:
#         for index in range(6):
#             alternative_state = seed_state.copy()
#             freedom_space = ["0", "1", "2", "3"]
#             freedom_space.remove(seed_state[index])
#             # print(freedom_space)
#             for bit in freedom_space:
#                 alternative_state[index] = bit
#                 state_pool.append(alternative_state.copy())
#         return state_pool
#
# def get_distance(state=None, state_list=None):
#     count = 0
#     if state_list == []:
#         return 0
#     for each in state_list:
#         for i in range(len(state)):
#             if state[i] == each[i]:
#                 pass
#             else:
#                 count += 1
#     return count / len(state) / len(state_list)
#
#
# x = func(divergence=1)
# distance = 0
# for index, each in enumerate(x):
#     distance += get_distance(state=each, state_list=x[index+1::])
# print(distance / len(x))