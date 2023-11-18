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
    t0 = time.time()
    landscape_iteration = 100
    N = 9
    state_num = 4
    alpha_list = [0.20, 0.25, 0.30]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 100
    for alpha in alpha_list:
        # DVs
        first_cache_across_K = []
        second_cache_across_K = []
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
            first_cache_across_K.append(temp_first_cache)
            second_cache_across_K.append(temp_second_cache)
        with open("first_cache_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(first_cache_across_K, out_file)
        with open("second_cache_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(second_cache_across_K, out_file)

        import matplotlib.pyplot as plt
        from scipy import stats
        import numpy as np
        for K, first_cache, second_cache in zip(K_list, first_cache_across_K, second_cache_across_K):
            fig, ax = plt.subplots()
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            plt.hist(first_cache, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            # Calculate mean and standard deviation using NumPy
            mean_fitness = np.mean(first_cache)
            std_dev_fitness = np.std(first_cache)
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
            plt.hist(second_cache, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            # Calculate mean and standard deviation using NumPy
            mean_fitness = np.mean(first_cache)
            std_dev_fitness = np.std(first_cache)
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

            # Perform t-test
            t_statistic, p_value = stats.ttest_ind(first_cache, second_cache)

            # Check the p-value to determine significance
            alpha = 0.05  # Set your desired significance level (commonly 0.05)
            print("===========K={0}, alpha={1}===============".format(K, alpha))
            if p_value < alpha:
                print("The distributions are significantly different (reject the null hypothesis)")
            else:
                print("The distributions are not significantly different (fail to reject the null hypothesis)")

            # Optionally, print t-statistic and p-value
            print(f"t-statistic: {t_statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            print("=====================")

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


