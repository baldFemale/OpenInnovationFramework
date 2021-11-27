from generalSpecialist5 import *
import multiprocessing
import pickle

N = 8
land_num = 400
period = 50
agentNum = 100
state_num = 4

teamup = False
learn_probability = None

IM_type = "influential"

# GST
knowledge_list = [
    [4, 4, ],
    [2, 2, 2, 2],
    [3, 3, 2,],
    [4, 4, 4],
    [2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 2, 2],

]

# to_file
file_path = "./output_multistate_generalistSpecialist_individual_{lr}_{IM_type}".format(
    lr=learn_probability,
    IM_type=IM_type,
)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(return_dic, idx, N, k, IM_type, land_num, period, agentNum, teamup, teamup_timing,
               knowledge_list, lr=0.1, state_num=4):
        """
    )

    jobs = []

    index = 0

    for knowledge_index in range(9):
        for k in [0, 14, 28, 42, 56]:

            p = multiprocessing.Process(
                target=simulation, args=(
                    return_dic, index, N, k, IM_type, land_num, period, agentNum, teamup, 0,
                    [knowledge_list[knowledge_index]], learn_probability, state_num,
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
