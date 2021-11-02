from generalSpecialist2 import *
import multiprocessing
import pickle

N = 10
land_num = 400
period = 30
agentNum = 100
state_num = 4

teamup = False
learn_probability = 0.1

# total elements 3 (8, 16, 24, 32)
knowledge_num = [4, 2, 3, 8, 4, 6, 10, 6, 9, 10, 8, 9]
specialist_num = [0, 2, 1, 0, 4, 2, 2, 6, 3, 6, 8, 7]

# to_file
file_path = "./output_multistate_generalistSpecialist_individual_{lr}".format(
    lr=learn_probability,
)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(return_dic, idx, N, k, land_num, period, agentNum, teamup, teamup_timing, knowledge_num, specialist_num,
               lr=0.1, state_num=2):
        """
    )

    jobs = []

    index = 0

    for knowledge_index in range(5, 6):
        for k in range(0, 1, 2):

            p = multiprocessing.Process(
                target=simulation, args=(
                    return_dic, index, 10, k, land_num, period, agentNum, teamup, 0, knowledge_num[knowledge_index], specialist_num[knowledge_index],
                    learn_probability, state_num,
                )
            )

            jobs.append(p)
            p.start()

            index += 1

    for proc in jobs:
        proc.join()

    res_dic = {}

    for k in return_dic.keys():
        res_dic[k] = return_dic[k]

    f = open(file_path, "wb")
    pickle.dump(res_dic, f)
    f.close()
