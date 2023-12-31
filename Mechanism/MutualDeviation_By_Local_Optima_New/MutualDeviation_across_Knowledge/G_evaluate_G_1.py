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
def func(N=None, K=None, agent_num=None, expertise=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.25)
    sender_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    receiver_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=expertise, specialist_expertise=0, label="G")
    for sender in sender_crowd.agents:
        for _ in range(search_iteration):
            sender.search()
    for receiver in receiver_crowd.agents:
        for _ in range(search_iteration):
            receiver.search()
    # Mutual Deviation
    mutual_deviation_rate_list = []
    for sender in sender_crowd.agents:
        sender_solution = sender.state.copy()
        sender_domain = sender.generalist_domain.copy()  # !!!
        count = 0
        for receiver in receiver_crowd.agents:
            learnt_solution = receiver.state.copy()  # Keep the receiver's mindset
            for index in sender_domain:
                learnt_solution[index] = sender_solution[index]
            suggestions = receiver.suggest_better_state_from_expertise(state=learnt_solution)
            for each_suggestion in suggestions:
                learnt_suggestion = sender.state.copy()  # Keep the sender's mindset
                for index in receiver.generalist_domain:  # !!!
                    learnt_suggestion[index] = each_suggestion[index]
                deviations = sender.suggest_better_state_from_expertise(state=learnt_suggestion)
                if sender_solution in deviations:
                    deviations.remove(sender_solution)
                if len(deviations) != 0:
                    count += 1
                    break
        mutual_deviation_rate = count / agent_num
        mutual_deviation_rate_list.append(mutual_deviation_rate)
    final_mutual_deviation_rate = sum(mutual_deviation_rate_list) / len(mutual_deviation_rate_list)
    return_dict[loop] = [final_mutual_deviation_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    g_knowledge_list = [12]
    concurrency = 100
    for g_knowledge in g_knowledge_list:
        joint_confusion_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, g_knowledge, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_joint_confusion = []
            for result in returns:
                temp_joint_confusion.append(result[0])

            joint_confusion_across_K.append(sum(temp_joint_confusion) / len(temp_joint_confusion))
        with open("gg_mutual_deviation_across_K_knowledge_{0}".format(g_knowledge), 'wb') as out_file:
            pickle.dump(joint_confusion_across_K, out_file)

    t1 = time.time()
    print("GG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


