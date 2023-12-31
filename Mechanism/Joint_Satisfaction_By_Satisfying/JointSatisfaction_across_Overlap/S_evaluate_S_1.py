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
def func(N=None, K=None, agent_num=None, overlap=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.25)
    sender_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    receiver_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    for sender in sender_crowd.agents:
        for _ in range(search_iteration):
            sender.search()
    # Joint Satisfaction
    joint_satisfaction_rate_list = []
    for sender in sender_crowd.agents:
        sender_solution = sender.state.copy()
        sender_domain = sender.specialist_domain.copy()  # !!!
        count = 0
        # Adjust the receiver crowd according to overlap
        other_domain_list = [i for i in range(N) if i not in sender_domain]
        for receiver in receiver_crowd.agents:
            if overlap == 0:
                receiver.specialist_domain = np.random.choice(other_domain_list,3).tolist()  # !!!
            else:
                receiver.specialist_domain = (np.random.choice(other_domain_list, 3 - overlap).tolist() +
                                              np.random.choice(sender_domain, overlap).tolist())  # !!!
            # receiver.state = np.random.choice(range(4), N).tolist()
            # receiver.state = [str(i) for i in receiver.state]  # state format: a list of string
            # receiver.cog_state = receiver.state_2_cog_state(state=receiver.state)
            # receiver.cog_fitness = receiver.get_cog_fitness(cog_state=receiver.cog_state, state=receiver.state)
            # receiver.fitness = landscape.query_second_fitness(state=receiver.state)
            #
            # for _ in range(search_iteration):
            #     receiver.search()

            learnt_solution = receiver.state.copy()   # Keep the receiver's mindset
            for index in sender_domain:
                learnt_solution[index] = sender_solution[index]
            cog_learnt_solution = receiver.state_2_cog_state(state=learnt_solution)
            cog_learnt_fitness = receiver.get_cog_fitness(cog_state=cog_learnt_solution, state=learnt_solution)
            if cog_learnt_fitness > receiver.cog_fitness:
                count += 1
        joint_satisfaction_rate = count / agent_num
        joint_satisfaction_rate_list.append(joint_satisfaction_rate)
    final_joint_confusion_rate = sum(joint_satisfaction_rate_list) / len(joint_satisfaction_rate_list)
    return_dict[loop] = [final_joint_confusion_rate]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    landscape_iteration = 100
    agent_num = 400
    search_iteration = 100
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    overlap_list = [1]
    concurrency = 100
    # DVs
    for overlap in overlap_list:
        joint_confusion_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, overlap, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_joint_confusion = []
            for result in returns:  # 50 landscape repetitions
                temp_joint_confusion.append(result[0])
            joint_confusion_across_K.append(sum(temp_joint_confusion) / len(temp_joint_confusion))
        with open("ss_joint_satisfaction_across_K_overlap_{0}".format(overlap), 'wb') as out_file:
            pickle.dump(joint_confusion_across_K, out_file)
    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Joint Satisfaction SS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
