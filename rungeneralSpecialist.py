from generalSpecialist import *
import multiprocessing
import pickle

N = 10
land_num = 10
period = 20
agentNum = 100

teamup = True
learn_probability = 0.1
knowledge_num = 6

# when time=33 -> work as individuals

# to_file
file_path = "./output_generalistSpecialist_{teamup}_{knowledge_num}_{lr}".format(
    teamup=teamup,
    lr=learn_probability,
    knowledge_num=knowledge_num,
)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        simulation(return_dic, idx, N, k, land_num, period, agentNum, teamup, teamup_timing, knowledge_num, lr=0.1
               ):
        """
    )

    jobs = []

    index = 0

    for timing in range(0, 20, 4):
        for k in range(0, 8, 2):

            p = multiprocessing.Process(
                target=simulation, args=(
                    return_dic, index, 12, k, land_num, period, agentNum, teamup, timing, knowledge_num,
                    learn_probability
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
