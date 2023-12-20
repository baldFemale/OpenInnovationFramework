# -*- coding: utf-8 -*-
# @Time     : 8/21/2023 19:35
# @Author   : Junyi
# @FileName: landscape_run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, state_num=None, alpha=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin", alpha=alpha)
    first_cache = landscape.first_cache.values()
    second_cache = landscape.second_cache.values()
    return_dict[loop] = [first_cache, second_cache]
    sema.release()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    t0 = time.time()
    landscape_iteration = 100
    N = 9
    state_num = 4
    alpha_list = [0.35, 0.40, 0.45, 0.50]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 100
    # Nature three colors
    nature_orange = "#F16C23"
    nature_blue = "#2B6A99"
    nature_green = "#1B7C3D"
    p_value_across_K_alpha = []
    mean_diff_across_K_alpha = []
    for alpha in alpha_list:
        # DVs
        # first_cache_across_K = []
        # second_cache_across_K = []
        p_value_across_K = []
        mean_diff_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, alpha, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            returns = return_dict.values()  # Don't need dict index, since it is repetition.

            temp_first_cache, temp_second_cache = [], []
            for result in returns:  # 50 landscape repetitions
                temp_first_cache.extend(result[0])
                temp_second_cache.extend(result[1])
            # first_cache_across_K.append(temp_first_cache)
            # second_cache_across_K.append(temp_second_cache)

            fig, ax = plt.subplots()
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            plt.hist(temp_first_cache, bins=40, facecolor=nature_blue, edgecolor="black", alpha=0.7)
            # Calculate mean and standard deviation using NumPy
            mean_fitness = np.mean(temp_first_cache)
            std_dev_fitness = np.std(temp_first_cache)
            # Annotate the plot with mean and standard deviation information
            plt.text(0.95, 0.95, f"Mean: {mean_fitness:.2f}\nStd Dev: {std_dev_fitness:.2f}",
                     horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
            plt.xlabel("Fitness Value")
            plt.ylabel("Count")
            plt.title("Coarse Landscape $N={0}$, $K={1}$, $\\alpha={2}$".format(N, K, alpha))
            plt.savefig("First_N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.clf()
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            plt.hist(temp_second_cache, bins=40, facecolor=nature_orange, edgecolor="black", alpha=0.7)
            # Calculate mean and standard deviation using NumPy
            mean_fitness = np.mean(temp_second_cache)
            std_dev_fitness = np.std(temp_second_cache)
            # Annotate the plot with mean and standard deviation information
            plt.text(0.95, 0.95, f"Mean: {mean_fitness:.2f}\nStd Dev: {std_dev_fitness:.2f}",
                     horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
            plt.xlabel("Fitness Value")
            plt.ylabel("Count")
            plt.title("Fine Landscape $N={0}$, $K={1}$, $\\alpha={2}$".format(N, K, alpha))
            plt.savefig("Second_N{0}_K{1}_alpha_{2}.png".format(N, K, alpha))
            plt.clf()
            plt.close(fig)

            # T-test
            t_statistic, p_value = stats.ttest_ind(temp_first_cache, temp_second_cache)
            mean_diff = np.mean(temp_first_cache) - np.mean(temp_second_cache)
            p_value_across_K.append(p_value)
            mean_diff_across_K.append(mean_diff)

        p_value_across_K_alpha.append(p_value_across_K)
        mean_diff_across_K_alpha.append(mean_diff_across_K)

        # with open("first_cache_alpha_{0}".format(alpha), 'wb') as out_file:
        #     pickle.dump(first_cache_across_K, out_file)
        # with open("second_cache_alpha_{0}".format(alpha), 'wb') as out_file:
        #     pickle.dump(second_cache_across_K, out_file)

    with open("p_value_across_K_alpha_3", 'wb') as out_file:
        pickle.dump(p_value_across_K_alpha, out_file)
    with open("mean_diff_across_K_alpha_3", 'wb') as out_file:
        pickle.dump(mean_diff_across_K_alpha, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


