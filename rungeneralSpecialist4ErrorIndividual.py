from generalSpecialist4Error import *
import multiprocessing
import pickle

N = 12
land_num = 400
period = 30
agentNum = 200
state_num = 2

IM_type = "random"
variance = 1

# total elements (16, 24, 32)
knowledge_list = [8, 8, 8, 12, 12, 12]
state_list = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2],
    [2, 2, 1, 1, 1, 1],
    [4, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 1, 1, 1, 1],
    [4, 1, 1, 1, 1, 1, 1, 1, 1],
]

# to_file
file_path = "./output_twoStatewithError_generalistSpecialist_individual_{IM_type}_{variance}".format(
    IM_type=IM_type,
    variance=variance,
)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(
            return_dic, idx, N, k, IM_type, land_num, period, agentNum, knowledge_num, state_list, state_num=2, 
            fixed_variance=True, variance=1
        ):
        """
    )

    jobs = []

    index = 0

    for knowledge_index in range(8):
        for k in range(0, 91, 18):
            for fixed_variance in [True, False]:
                p = multiprocessing.Process(
                    target=simulation, args=(
                        return_dic, index, N, k, IM_type, land_num, period, agentNum, knowledge_list[knowledge_index],
                        state_list[knowledge_index], state_num, fixed_variance, variance
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
