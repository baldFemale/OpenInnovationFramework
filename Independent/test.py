from itertools import product

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
X = np.arange(0.1, 1.1, 0.1)  # list X
Y = np.arange(0.1, 1.1, 0.1)  # list Y
Z = np.random.rand(len(X) * len(Y))  # list Z (random values for demonstration)

# Reshape data
X, Y = np.meshgrid(X, Y)  # Create a grid of X and Y values
Z = Z.reshape(X.shape)  # Reshape Z to match the grid shape

# Plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')

# Show the plot
plt.show()
