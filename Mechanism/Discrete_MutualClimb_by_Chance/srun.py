# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Specialist import Specialist
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    count = 0
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, specialist_expertise=expertise_amount)
        for _ in range(search_iteration):
            specialist.search()
        state = specialist.state.copy()
        neighbor_states = []
        for index in range(N):
            for bit in ["0", "1", "2", "3"]:
                new_state = state.copy()
                if bit != state[index]:
                    new_state[index] = bit
                    neighbor_states.append(new_state)

        for each_neighbor in neighbor_states:
            climbed_list = specialist.suggest_better_state_from_expertise(state=each_neighbor)
            if state in climbed_list:
                climbed_list.remove(state)
            if len(climbed_list) > 1:
                count += 1
                break
    mutual_climb_rate = count / 50
    return_dict[loop] = [mutual_climb_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    expertise_amount = 12
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    # DVs
    mutual_climb_across_K = []
    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, state_num, expertise_amount,
                                              agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        returns = return_dict.values()  # Don't need dict index, since it is repetition.

        temp_climb = []
        for result in returns:  # 50 landscape repetitions
            temp_climb.append(result[0])

        mutual_climb_across_K.append(sum(temp_climb) / len(temp_climb))
    # remove time dimension
    with open("s_mutual_climb_across_K", 'wb') as out_file:
        pickle.dump(mutual_climb_across_K, out_file)
    t1 = time.time()
    print("S Climb: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


