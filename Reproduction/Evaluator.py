# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("Generalist_N10_K0_k43_E12",'rb') as infile:
    data = pickle.load(infile)
print(data)

print(len(data), len(data[0]), len(data[0][0]))
print(np.mean(np.mean(np.array(data), axis=0), axis=0))
# for landscape_loop in range(len(data)):

plt.plot(np.mean(np.mean(np.array(data), axis=0), axis=0))
plt.legend()
plt.show()
#
