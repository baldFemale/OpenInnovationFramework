from generalFramework import *
import multiprocessing
import pickle

N = 10
land_num = 500
period = 50

agentNum = 10

# to_file
file_path = "./output_generalFramework_{agentNum}".format(
    agentNum=agentNum,
)


def parallel_simulation(return_dic, idx, k, interval):
    ress = []

    for firm_knowledge in [4, 6, 8][:1]:
        for agent_knowledge in [4, 6, 8][1:2]:
            print(firm_knowledge, agent_knowledge)
            ress.append(simulation(N, k, land_num, agentNum, period, firm_knowledge, agent_knowledge,
                                   interval))

    return_dic[idx] = ress


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(N, k, land_num, agent_num, period, firm_knowledge, agent_knowledge, time_interval
               )
        """
    )

    jobs = []

    time_interval_list = [
        [0, 50], # crowd-sourcing
        [0, 40, 50], # crowd then internal R&D
        [0, 30, 50], # crowd then internal R&D
        [0, 20, 50], # crowd then internal R&D
        [0, 10, 50], # crowd then internal R&D
        [50], # internal R & D
        [40, 50], # internal R&D then crowd
        [30, 50], # internal R&D then crowd
        [20, 50], # internal R&D then crowd
        [10, 50], # internal R&D then crowd
        [10, 20, 50], # internal R&D, crowd, then internal R&D
        [10, 30, 50], # internal R&D, crowd, then internal R&D
        [10, 40, 50], # internal R&D, crowd, then internal R&D
        [20, 30, 50], # internal R&D, crowd, then internal R&D
        [20, 40, 50], # internal R&D, crowd, then internal R&D
        [30, 40, 50], # internal R&D, crowd, then internal R&D
        [0, 10, 20, 50], # crowd, firm, crowd
        [0, 10, 30, 50], # crowd, firm, crowd
        [0, 10, 40, 50], # crowd, firm, crowd
        [0, 20, 30, 50], # crowd, firm, crowd
        [0, 20, 40, 50], # crowd, firm, crowd
        [0, 30, 40, 50], # crowd, firm, crowd
    ]

    index = 0

    for interval_index in range(5, len(time_interval_list), 100):
        for k in range(8, 10, 2):

            p = multiprocessing.Process(
                target=parallel_simulation, args=(
                    return_dic, index, k, time_interval_list[interval_index]
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
