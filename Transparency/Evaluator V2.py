# -*- coding: utf-8 -*-
# @Time     : 4/6/2022 18:25
# @Author   : Junyi
# @FileName: Evaluator V2.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re
import scipy


class Evaluator:
    def __init__(self, title=None, data_path='', output_path=''):
        if not title:
            self.title = ''
        else:
            self.title = title
        self.data_path = data_path
        self.output_path = output_path
        self.data = None  # The data temp
        self.files_list = self.load_files_under_folder(folders=data_path)
        # outcome variables
        self.potential_files = []
        self.unique_files = []
        self.convergence_files = []
        # independent variables

        # Parameters
        self.N = None
        self.knowledge_num = None
        self.state_num = None
        self.K_list = None
        self.frequency_list = None
        self.openness_list = None
        self.G_exposed_to_G_list = None
        self.gs_proportion_list = None
        self.exposure_type = None

    def load_iv_configuration(self, exposure_type, N, knowledge_num, state_num, K_list, frequency_list, openness_list, quality_list,
                              G_exposed_to_G_list, S_exposed_to_S_list, gs_proportion_list):
        self.exposure_type = exposure_type
        self.N = N
        self.knowledge_num = knowledge_num
        self.state_num = state_num
        self.K_list = K_list
        self.frequency_list = frequency_list
        self.openness_list = openness_list
        self.quality_list = quality_list
        self.G_exposed_to_G_list = G_exposed_to_G_list
        self.S_exposed_to_S_list = S_exposed_to_S_list
        self.gs_proportion_list = gs_proportion_list

    def load_simulation_configuration(self, landscape_iteration, agent_num,search_iteration):
        self.landscape_iteration = landscape_iteration
        self.agent_num = agent_num
        self.search_iteration = search_iteration

    def load_files_under_folder(self, folders=None):
        """
        For single-layers file loading
        :return:the data file list
        """
        data_files = []
        for root, dirs, files in os.walk(folders):
            for ech_file in files:
                if (".py" in ech_file) or (".pbs" in ech_file):
                    pass
                data_files.append(root + '\\' + ech_file)
        return data_files

    def load_folders_under_parent_folder(self, parent_folder=None):
        """
        For multi-layers file loading
        :return:the folder list containing the data files
        """
        folder_list = []
        for root, dirs, files in os.walk(parent_folder):
            for each_dir in dirs:
                folder_list.append(root + '\\' + each_dir)
        return folder_list

    def load_data_from_files(self, files_list=None):
        """
        load one data from one certain file path
        :param file_path: data file path
        :return:the loaded data (i.e., fitness list)
        """
        data = []
        for each_file in files_list:
            if (".png" not in each_file) and (".py" not in each_file):
                with open(each_file, 'rb') as infile:
                    # print(each_file)
                    temp = pickle.load(infile)
                    data.append(temp)
        return data

    def get_frequency_figure(self, data=None, title="Frequency",y_label=None,show_variance=False ):
        frequency_dimention = len(self.frequency_list)
        frequency_files_list = [[[]]*len(self.K_list)] * frequency_dimention
        K_files_list = [[]] * len(self.K_list)
        # print(len(frequency_files_list))
        selected_files_list = []
        for each_file in self.files_list:
            if ("_O0" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file):
                selected_files_list.append(each_file)

        for each_file in selected_files_list:
            for index, k in enumerate(self.K_list):
                if str("_K" + str(k) + "_") in each_file:
                    temp = K_files_list[index].copy()
                    temp.append(each_file)
                    K_files_list[index] = temp

        # K_files_list = np.array(K_files_list, dtype=object)
        # for each in K_files_list:
        #     print(len(each))
        # print("K files list shape: ", K_files_list.shape)

        for each_file in selected_files_list:
            if "Unique" in y_label:
                if "Unique" in each_file:
                    for index, each_freq in enumerate(self.frequency_list):
                        if "_F" + str(each_freq) + '_' in each_file:
                            temp = frequency_files_list[index].copy()
                            temp.append(each_file)
                            frequency_files_list[index] = temp
            elif "Average" in y_label:
                if "Convergence" in each_file:
                    for index, each_freq in enumerate(self.frequency_list):
                        if "_F" + str(each_freq) + '_' in each_file:
                            temp = frequency_files_list[index].copy()
                            temp.append(each_file)
                            frequency_files_list[index] = temp
            elif "Potential" in y_label:
                if "Potential" in each_file:
                    for index, each_freq in enumerate(self.frequency_list):
                        if "_F" + str(each_freq) + '_' in each_file:
                            temp = frequency_files_list[index].copy()
                            temp.append(each_file)
                            frequency_files_list[index] = temp
        frequency_files_list = np.array(frequency_files_list, dtype=object)
        frequency_files_list = frequency_files_list.reshape((len(self.frequency_list), len(self.K_list), -1))
        print("Frequency Files list shape: ", np.array(frequency_files_list).shape)
        print("Frequency Files list: ", np.array(frequency_files_list)[1][0])
        # all_curves_data = []
        # for each_curve_files in frequency_files_list:
        #     data_curve = self.load_data_from_files(files_list=each_curve_files)
        #     all_curves_data.append(data_curve)
        # data_list = np.array(all_curves_data, dtype=object)
        # print("Curves Data Shape: ", data_list.shape)
        # self.data = data_list
        # label_list = self.frequency_list

        # if len(all_curves_data) != len(self.frequency_list):
        #     raise ValueError("Frequency dimension mismatch")
        # if len(all_curves_data[0][0]) != self.landscape_iteration:
        #     raise ValueError("Landscape dimension mismatch")
        # if len(all_curves_data[0][0][0]) != self.agent_num:
        #     raise ValueError("Agent number dimension mismatch")
        # # curves_data_shape: (5, 100, 100, 400)
        # figure = plt.figure()
        # ax = figure.add_subplot()
        # for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
        #     print("Curve Shape: ", np.array(each_curve).shape)
        #     average_value, variation_value = None, None
        #     if "Unique" in y_label:
        #     # unique fitness lack one dimension
        #         average_value = np.mean(np.mean(np.array(each_curve), axis=3), axis=2) # 10 * (500 * 500)
        #         variation_value = np.var(np.array(each_curve).reshape((100, -1)), axis=1)
        #     elif ("Average" in y_label) or ("Potential" in y_label):
        #         average_value = np.mean(np.mean(np.mean(np.array(each_curve), axis=3), axis=2), axis=1)  # 10 * (500 * 500)
        #         variation_value = np.var(np.array(each_curve).reshape((100, -1)), axis=1)
        #     ax.plot(average_value, label="{}".format(lable))
        #     if show_variance:
        #         # draw the variance
        #         lower = [x - y for x, y in zip(average_value, variation_value)]
        #         upper = [x + y for x, y in zip(average_value, variation_value)]
        #         xaxis = list(range(len(lower)))
        #         ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)
        #
        # ax.set_xlabel('K')  # Add an x-label to the axes.
        # ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        # ax.set_title(self.title)  # Add a title to the whole figure
        # plt.legend()
        # output = self.output_path + "\\" + self.title + ".png"
        # plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        # plt.show()

    def get_outcome_variable_files(self):
        for each in self.files_list:
            if "png" in each:
                continue
            if "Potential" in each:
                self.potential_files.append(each)
            elif "Convergence" in each:
                self.convergence_files.append(each)
            elif "Unique" in each:
                self.unique_files.append(each)
            else:
                pass



if __name__ == '__main__':
    # data_foler = r'C:\Python_Workplace\hpc-0403\N6_self'
    data_foler = r'C:\Python_Workplace\hpc-0403\N6_overall'

    output_folder = r'C:\Python_Workplace\hpc-0403'
    # Search loop
    landscape_iteration = 100
    agent_num = 400
    search_iteration = 50
    # Parameter
    N = 6
    state_num = 4
    knowledge_num = 8
    K_list = [0, 2, 4, 6]
    frequency_list = [1, 5, 10, 20, 40]
    openness_list = [0, 0.25, 0.5, 0.75, 1.0]
    quality_list = [0, 0.25, 0.5, 0.75, 1.0]
    G_exposed_to_G_list = [0, 0.25, 0.5, 0.75, 1.0]
    S_exposed_to_S_list = [0, 0.25, 0.5, 0.75, 1.0]
    gs_proportion_list = [0, 0.25, 0.5, 0.75, 1.0]
    files_number = len(K_list) * len(frequency_list) * len(openness_list) * len(quality_list) *\
                   len(G_exposed_to_G_list) * len(S_exposed_to_S_list) * len(gs_proportion_list)*3
    print("Expected Files Number: ", files_number)
    exposure_type = "Self-interested"

    evaluator = Evaluator(data_path=data_foler, output_path=output_folder, )
    evaluator.load_iv_configuration(exposure_type=exposure_type, N=N, state_num=state_num, K_list=K_list, frequency_list=frequency_list,
                                    openness_list=openness_list, quality_list=quality_list, G_exposed_to_G_list=G_exposed_to_G_list,
                                    S_exposed_to_S_list=S_exposed_to_S_list, gs_proportion_list=gs_proportion_list, knowledge_num=knowledge_num)
    evaluator.load_simulation_configuration(landscape_iteration=landscape_iteration, agent_num=agent_num, search_iteration=search_iteration)
    evaluator.get_outcome_variable_files()
    evaluator.get_frequency_figure(title="Frequency", y_label="Average Fitness")
    print("END")