from generalSpecialist2 import *
import multiprocessing
import pickle

N = 10
land_num = 400
period = 30
agentNum = 300
state_num = 4

teamup = True
learn_probability = 0.1

knowledge_type = "high"
knowledge_num = []
specialist_num = []

if knowledge_type == "high":
    knowledge_num = [10, 6, 9]
    specialist_num = [2, 6, 3]
elif knowledge_type == "medium":
    knowledge_num = [8, 4, 6]
    specialist_num = [0, 4, 2]
elif knowledge_type == "low":
    knowledge_num = [4, 2, 3]
    specialist_num = [0, 2, 1]

# to_file
file_path = "./output_multistate_generalistSpecialist_{teamup}_{knowledge_type}_{lr}".format(
    teamup=teamup,
    lr=learn_probability,
    knowledge_type=knowledge_type,
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

    for timing in range(0, 30, 3):
        for k in range(0, 10, 2):

            p = multiprocessing.Process(
                target=simulation, args=(
                    return_dic, index, 10, k, land_num, period, agentNum, teamup, timing, knowledge_num, specialist_num,
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
