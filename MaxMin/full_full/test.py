# -*- coding: utf-8 -*-
# @Time     : 6/3/2023 22:03
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from sklearn.cluster import KMeans
from Landscape import Landscape
import numpy as np
N = 6
K = 5
state_num = 4
np.random.seed(1000)
landscape = Landscape(N=N, K=K, state_num=state_num, norm="MaxMin")
N = 5  # Number of dimensions
# landscape = ["00000", "00001", "00002", ..., "33333"]  # Your list of characters representing the landscape
landscape_list = list(landscape.cache.keys())


# Convert the landscape to a feature matrix
features = [[int(char) for char in entry] for entry in landscape_list]

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(features)

# Get the labels assigned by the clustering algorithm
labels = kmeans.labels_

# Divide the landscape into sub-lists based on the clustering labels
subspaces = [[] for _ in range(4)]
for i, entry in enumerate(landscape_list):
    subspaces[labels[i]].append(entry)

# Print the subspaces
for subspace in subspaces:
    print(subspace)
