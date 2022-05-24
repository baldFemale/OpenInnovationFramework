# -*- coding: utf-8 -*-
# @Time     : 5/15/2022 23:30
# @Author   : Junyi
# @FileName: test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pickle
import numpy as np

file = r'C:\Python_Workplace\hpc-0522\Pilot_Test\1Average_N6_K0_E8_Self-interested_SS0_GG0_F0_Prop0.5_Q0_O0_'
with open(file, 'rb') as in_file:
    IM = pickle.load(in_file)
    for each in np.array(IM):
        print(each)
print(len(IM))
