from generalFramework2 import *
import multiprocessing
import pickle

N = 12
land_num = 1000
period = 50
state_num = 2
IM_type = "random"
crowd_information_ratio=0.25
crowd_imitation_prob=0.5

crowd_num = 100
aspiration_ratio = 0.5
time_interval_index = 3
population_feedback = False

# to_file
file_path = "./output_generalFramework_{crowd_num}_{aspiration_ratio}_{time_interval_index}_{population_feedback}".format(
    crowd_num=crowd_num,
    aspiration_ratio=aspiration_ratio,
    time_interval_index=time_interval_index,
    population_feedback=population_feedback,
)

time_interval_list = [
    [0, 50],  # crowd-sourcing
    [50],  # internal R & D
    [0, 30, 50],  # idea community
    [20, 30, 50],  # idea polishment
]


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(return_dic, idx, N, k, land_num, firm_knowledge_num, crowd_num, period, crowd_knowledge_num,
               aspiration_ratio, time_interval, feedback_frequency, population_feedback, openness, state_num=2,
               crowd_information_ratio=0.25, crowd_imitation_prob=0.5, IM_type="random",
               ):
        """
    )

    jobs = []

    index = 0

    for k in [0, 60, 121][1:2]:
        for firm_knowledge in [8, 10, 12][1:2]:
            for crowd_knowledge in [4, 6, 8][1:2]:
                for feedback_frequency in [0, 0.5, 1][1:2]:
                    for openness in [0, 0.5, 1][1:2]:

                        p = multiprocessing.Process(
                            target=simulation, args=(
                                return_dic, index, N, k, land_num, firm_knowledge, crowd_num, period, crowd_knowledge,
                                aspiration_ratio, time_interval_list[time_interval_index], feedback_frequency,
                                population_feedback, openness, state_num, crowd_information_ratio, crowd_imitation_prob,
                                IM_type,
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
