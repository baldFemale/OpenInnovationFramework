from generalSpecialist5 import *
import multiprocessing
import pickle

N = 8
land_num = 400
period = 50
agentNum = 900
state_num = 4

teamup = True
learn_probability = 1

IM_type = "influential"

# SGT
# Has to be this order !
knowledge_list = [
    [[4, 4, ], [2, 2, 2, 2], [3, 3, 2,]],
    [[4, 4, 4],  [2, 2, 2, 2, 2, 2], [3, 3, 3, 3]],
    [[4, 4, 4, 4], [2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 2, 2]]
]

negotiation_param_list = [
    [2, None],
    [0, None],
    [1, ["s", "g", "t"]],
    [1, ["s", "t", "g"]],
    [1, ["g", "s", "t"]],
    [1, ["g", "t", "s"]],
    [1, ["t", "g", "s"]],
    [1, ["t", "s", "g"]]
][:4]


# to_file
file_path = "./output_multistate_generalistSpecialist_team_{lr}_{IM_type}_nego_firsthalf".format(
    lr=learn_probability,
    IM_type=IM_type,
)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(return_dic, idx, N, k, IM_type, land_num, period, agentNum, teamup, teamup_timing,
               knowledge_list, lr=0.1, state_num=4, negotiation_round=2, negoitation_priority=["s", "g", "t"]):
        """
    )

    jobs = []

    index = 0

    for knowledge_index in range(3):
        for k in [0, 14, 28, 42, 56]:
            for negotiation_index in range(len(negotiation_param_list)):

                p = multiprocessing.Process(
                    target=simulation, args=(
                        return_dic, index, N, k, IM_type, land_num, period, agentNum, teamup, 0,
                        knowledge_list[knowledge_index], learn_probability, state_num,
                        negotiation_param_list[negotiation_index][0],
                        negotiation_param_list[negotiation_index][1],
                    )
                )

                jobs.append(p)
                p.start()

                index += 1
    print("start %d processors"%index)

    for proc in jobs:
        proc.join()

    res_dic = {}

    for k in return_dic.keys():
        res_dic[k] = return_dic[k]

    f = open(file_path, "wb")
    pickle.dump(res_dic, f)
    f.close()
