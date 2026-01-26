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
def func(N=None, K=None, agent_num=None, search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.05)
    sender_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    receiver_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    for sender in sender_crowd.agents:
        for _ in range(search_iteration):
            sender.search()
    # Test whether the solutions configured by others are more attractive to the focal solver
    for receiver in receiver_crowd.agents:
        for _ in range(search_iteration):
            receiver.search()
    joint_confirmation_rate_list = []
    mutual_deviation_rate_list = []
    for sender in sender_crowd.agents:
        sender_solution = sender.state.copy()
        sender_domain = sender.generalist_domain.copy()  # !!!
        confirmation_count = 0
        deviation_count = 0
        for receiver in receiver_crowd.agents:
            learnt_solution = receiver.state.copy()
            for index in sender_domain:
                learnt_solution[index] = sender_solution[index]
            # Joint Confirmation
            # if the learnt_solution is better
            cog_learnt_solution = receiver.state_2_cog_state(state=learnt_solution)
            cog_learnt_fitness = receiver.get_cog_fitness(cog_state=cog_learnt_solution, state=learnt_solution)
            if cog_learnt_fitness > receiver.cog_fitness:
                confirmation_count += 1

            # Mutual Deviation
            # if the learnt_solution will be better
            suggestions = receiver.suggest_better_state_from_expertise(state=learnt_solution)
            if len(suggestions) != 0:
                deviation_count += 1
        joint_confirmation_rate = confirmation_count / agent_num
        joint_confirmation_rate_list.append(joint_confirmation_rate)
        mutual_deviation_rate = deviation_count / agent_num
        mutual_deviation_rate_list.append(mutual_deviation_rate)
    joint_confirmation = sum(joint_confirmation_rate_list) / len(joint_confirmation_rate_list)
    mutual_deviation = sum(mutual_deviation_rate_list) / len(mutual_deviation_rate_list)
    return_dict[loop] = [joint_confirmation, mutual_deviation]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 500
    search_iteration = 200
    N = 9
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 100
    # DVs
    joint_confirmation_across_K = []
    mutual_deviation_across_K = []
    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        results = np.array(list(return_dict.values()), dtype=float)  # shape: (reps, 2)
        joint_confirmation_across_K.append(results[:, 0].mean())
        mutual_deviation_across_K.append(results[:, 1].mean())
    with open("gs_joint_confirmation.pkl", 'wb') as out_file:
        pickle.dump(joint_confirmation_across_K, out_file)
    with open("gs_mutual_deviation.pkl", 'wb') as out_file:
        pickle.dump(mutual_deviation_across_K, out_file)
    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Interpretation Dynamics of GS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
