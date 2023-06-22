# -*- coding: utf-8 -*-
# @Time     : 5/19/2023 16:58
# @Author   : Junyi
# @FileName: drawer.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Landscape import Landscape
from CogLandscape import CogLandscape
from BinaryLandscape import BinaryLandscape
import matplotlib.pyplot as plt
# plt.interactive(False)
from scipy.stats import gaussian_kde
from scipy.special import kl_div
import numpy as np
import pickle
legend_properties = {'weight':'bold'}
nus_blue = "#003D7C"
nus_orange = "#EF7C00"
# Nature three colors
nature_orange = "#F16C23"
nature_blue = "#2B6A99"
nature_green = "#1B7C3D"
# Morandi six colors
morandi_blue = "#046586"
morandi_green =  "#28A9A1"
morandi_yellow = "#C9A77C"
morandi_orange = "#F4A016"
morandi_pink = "#F6BBC6"
morandi_red = "#E71F19"
morandi_purple = "#B08BEB"
# Others
shallow_grey = "#D3D4D3"
deep_grey = "#A6ABB6"

# Shallow-deep pair
shallow_purple = "#EAD7EA"
deep_purple = "#BA9DB9"
shallow_cyan = "#A9D5E0"
deep_cyan = "#48C0BF"
shallow_blue = "#B6DAEC"
deep_blue = "#98CFE4"
shallow_pink = "#F5E0E5"
deep_pink = "#E5A7B6"
shallow_green = "#C2DED0"
deep_green = "#A5C6B1"

K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
expertise_domain_list = [18]
for K in K_list:
    index = 0
    fig, ax = plt.subplots()
    for expertise_domain in expertise_domain_list:
        index += 1
        bin_data_file = r"bin_cache_K_{0}_E_{1}".format(K, expertise_domain)
        data_file = r"cache_K_{0}_E_{1}".format(K, expertise_domain)
        cog_data_file = r"cog_cache_K_{0}_E_{1}".format(K, expertise_domain)
        with open(bin_data_file, "rb") as infile:
            bin_data = pickle.load(infile)
        with open(data_file, "rb") as infile:
            data = pickle.load(infile)
        with open(cog_data_file, "rb") as infile:
            cog_data = pickle.load(infile)
        plt.hist(bin_data, bins=40, color=nature_blue, alpha=0.3, density=True, label='Bin, K{0}'.format(K))
        kde_1 = gaussian_kde(bin_data)
        x_values_1 = np.linspace(min(bin_data), max(bin_data), 40)
        pdf_1 = kde_1(x_values_1)
        plt.plot(x_values_1, pdf_1, '-', color=nature_blue)

        plt.hist(data, bins=40, color=nature_orange, alpha=0.3, density=True, label='Multi, K{0}'.format(K))
        kde_2 = gaussian_kde(data)
        x_values_2 = np.linspace(min(data), max(data), 40)
        pdf_2 = kde_2(x_values_2)
        plt.plot(x_values_2, pdf_2, '-', color=nature_orange)

        plt.hist(cog_data, bins=40, color=nature_green, alpha=0.3, density=True, label='Cog, K{0}'.format(K))
        kde_3 = gaussian_kde(cog_data)
        x_values_3 = np.linspace(min(cog_data), max(cog_data), 40)
        pdf_3 = kde_3(x_values_3)
        plt.plot(x_values_3, pdf_3, '-', color=nature_green)
    # Add labels and legend
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Landscape Distribution')
    plt.legend(frameon=False, prop=legend_properties)
    plt.savefig(r"landscape_distribution_K_{0}.png".format(K), transparent=True, dpi=300)
    # plt.show()
    # plt.clf()