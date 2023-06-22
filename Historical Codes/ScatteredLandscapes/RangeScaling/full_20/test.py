from itertools import product
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
from Landscape import Landscape
from CogLandscape import CogLandscape
from BinaryLandscape import BinaryLandscape
import matplotlib.pyplot as plt
# plt.interactive(False)
from scipy.stats import gaussian_kde
from scipy.special import kl_div
import numpy as np
import pickle
# knowledge_representation = ["A", "B"]
# knowledge_representation = [0, 1, 2, 3]
# knowledge_representation = ["A", "B", "0", "1", "2", "3", "*"]
# expertise_domain = 2
# test = list(product(knowledge_representation, repeat=expertise_domain))
# print(test)
# print(len(test))

# import numpy as np
#
# list_A = np.array(range(0, 5))
# list_B = np.array(range(10, 15))
# positions = [2, 4, 6]  # Example positions where elements from list_B will be inserted
#
# combined_list = np.zeros(len(list_A) + len(list_B), dtype=list_A.dtype)
# insertion_indices = np.array(positions)
#
# combined_list[insertion_indices] = list_B
#
# remaining_indices = np.delete(np.arange(len(combined_list)), insertion_indices)
# remaining_elements = np.concatenate((list_A, list_B[len(positions):]))
#
# combined_list[remaining_indices] = remaining_elements
#
# print(list(combined_list))

# test_dict = {"1": 0.9, "2": 0.7, "3": 0.2}
# for value in test_dict.values():
#     value /= max(test_dict.values())
#
# print(test_dict)
#
# max_ = max(test_dict.values())
# for key in test_dict.keys():
#     test_dict[key] /= max_
# print(test_dict)

# test = ["1", "2", "A", "*"]
# print("".join(test))

# all_representation = ["A", "B", "0", "1", "2", "3", "*"]
# # Cartesian products outside knowledge scope (not combinations or permutations)
# unknown_products = list(product(all_representation, repeat=4))
# print(unknown_products)


import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Sample data
# X = np.arange(0.1, 1.1, 0.1)  # list X
# Y = np.arange(0.1, 1.1, 0.1)  # list Y
# Z = np.random.rand(len(X) * len(Y))  # list Z (random values for demonstration)
#
# # Reshape data
# X, Y = np.meshgrid(X, Y)  # Create a grid of X and Y values
# Z = Z.reshape(X.shape)  # Reshape Z to match the grid shape
#
# # Plot 3D surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
#
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Surface Plot')
#
# # Show the plot
# plt.show()

#
# test = np.random.choice(range(9), 12 // 2, replace=False).tolist()
# test = [str(i) for i in test]
# print(test)
# print(sorted(test))
# print(test)
# test_ = "".join(test)
# print(test_)


# import matplotlib.pyplot as plt
# uniform_list = np.random.uniform(0, 1, 100000)
# normal_list = [sum(uniform_list[index*10: (index+1)*10]) / 10 for index in range(len(uniform_list) // 10)]
# normal_list_20 = [sum(uniform_list[index*20: (index+1)*20]) / 20 for index in range(len(uniform_list) // 20)]
# normal_normal_list = [sum(normal_list[index*10: (index+1)*10]) / 10 for index in range(len(normal_list) // 10)]
#
#
# bins = 40
# # Hist 1: Uniform Distribution
# plt.hist(uniform_list, bins=bins, color=nature_blue, alpha=0.3, density=True)
# kde1 = gaussian_kde(uniform_list)
# x_values1 = np.linspace(min(uniform_list), max(uniform_list), bins)
# pdf1 = kde1(x_values1)
# plt.plot(x_values1, pdf1, '-', color=nature_blue,
#          label='Uniform, $\mu=${0}, $\sigma=${1}'.format(round(sum(uniform_list) / len(uniform_list), 2), round(float(np.std(uniform_list)), 2)))
#
# # Hist 2: Take an Average
# plt.hist(normal_list, bins=bins, color=nature_green, alpha=0.3, density=True)
# kde2 = gaussian_kde(normal_list)
# x_values2 = np.linspace(min(normal_list), max(normal_list), bins)
# pdf2 = kde2(x_values2)
# plt.plot(x_values2, pdf2, '-', color=nature_green,
#          label='Ave10, $\mu=${0}, $\sigma=${1}'.format(round(sum(normal_list) / len(normal_list), 2), round(float(np.std(normal_list)), 2)))
#
# # Hist 2-2: Take an Average across 20
# plt.hist(normal_list_20, bins=bins, color=morandi_pink, alpha=0.3, density=True)
# kde22 = gaussian_kde(normal_list_20)
# x_values22 = np.linspace(min(normal_list_20), max(normal_list_20), bins)
# pdf22 = kde22(x_values22)
# plt.plot(x_values22, pdf22, '-', color=morandi_pink,
#          label='Ave20, $\mu=${0}, $\sigma=${1}'.format(round(sum(normal_list_20) / len(normal_list_20), 2), round(float(np.std(normal_list_20)), 2)))
#
# # Hist 3: Take a Second Average
# plt.hist(normal_normal_list, bins=bins, color=nature_orange, alpha=0.3, density=True)
# kde3 = gaussian_kde(normal_normal_list)
# x_values3 = np.linspace(min(normal_normal_list), max(normal_normal_list), bins)
# pdf3 = kde3(x_values3)
# plt.plot(x_values3, pdf3, '-', color=nature_orange,
#          label='AveAve, $\mu=${0}, $\sigma=${1}'.format(round(sum(normal_normal_list) / len(normal_normal_list), 2), round(float(np.std(normal_normal_list)), 2)))
#
# plt.xlabel("Range")
# plt.ylabel("Density")
# plt.title("Average Pooling and Induced Distribution")
# plt.legend()
# plt.savefig("Distributions.png")
# plt.show()


bit_counts = {'0': 10, '1': 110, '2': 0, '3': 0}
dominant_bit = max(bit_counts, key=bit_counts.get)
print(dominant_bit, type(dominant_bit))
