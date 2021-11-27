from generalSpecialistPopulation import *
import multiprocessing
import pickle

N = 10
land_num = 500
period = 50
crowd_num = 100
aspiration_ratio = 0.5
state_num = 4
IM_type = "random"

time_interval_index = 2

# to_file
file_path = "./output_generalFramework_specialistGeneralist_{time_interval_index}".format(
    time_interval_index=time_interval_index,
)

time_interval_list = [
    [0, 50],  # crowd-sourcing
    [50],  # internal R & D
    [0, 30, 50],  # idea community
    [20, 30, 50],  # idea polishment
]

crowd_proportion_list = [
    [0, 1],  # all specialist
    [0.25, 1],
    [0.5, 1],
    [0.75, 1],
    [1, 1]
]

# total elements (16, 24, 32)
knowledge_num = [10, 6, 8]
specialist_num = [2, 6, 4]


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(
                return_dic, idx, N, k, land_num, crowd_num, period, crowd_proportion, crowd_knowledge, crowd_special_knowledge,
                 aspiration_ratio, time_interval, state_num=4, IM_type="random", in_progress_feedback=False, open_env=False
               )
        """
    )

    jobs = []

    index = 0

    for proportion_index in range(0, len(crowd_proportion_list)):
        for k in range(0, 91, 30):

            for in_progress_feedback in [False, True]:
                for open_env in [False, True]:

                    p = multiprocessing.Process(
                        target=simulation, args=(
                            return_dic, index, N, k, land_num, crowd_num, period, crowd_proportion_list[proportion_index],
                            knowledge_num, specialist_num, aspiration_ratio, time_interval_list[time_interval_index],
                            state_num, IM_type, in_progress_feedback, open_env,
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
