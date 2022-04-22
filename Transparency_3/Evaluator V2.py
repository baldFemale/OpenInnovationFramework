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
            self.title = 'Not-name'
        else:
            self.title = title
        self.data_path = data_path
        self.output_path = output_path
        self.data = None  # The data temp
        self.files_list = self.load_files_under_folder(folders=data_path)

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

    def load_data_from_folders(self, folder_list=None):
        data = []
        for each_folder in folder_list:
            temp_1 = []
            for each_file in each_folder:
                if (".png" not in each_file) and (".py" not in each_file):
                    with open(each_file, 'rb') as infile:
                        # print(each_file)
                        temp_2 = pickle.load(infile)
                        temp_1.append(temp_2)
            data.append(temp_1)
        return data

    def generate_one_dimension_figure(self, dimension=None, title=None,y_label=None,show_variance=False):
        if title:
            self.title = title
        frequency_dimention = len(self.frequency_list)
        openness_dimension = len(openness_list)
        quality_list_dimension = len(openness_list)
        gs_proportion_dimension = len(gs_proportion_list)
        if dimension == "Frequency":
            dimension_files_list = [[]] * frequency_dimention
        elif dimension == "Quality":
            dimension_files_list = [[]] * quality_list_dimension
        elif dimension == "Openness":
            dimension_files_list = [[]] * openness_dimension
        elif dimension == "Proportion":
            dimension_files_list = [[]] * gs_proportion_dimension
        else:
            raise ValueError("Unsupported")

        K_files_list = [[]] * len(self.K_list)
        # print(len(frequency_files_list))
        selected_files_list = []
        for each_file in self.files_list:
            if ("_O0" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file) and ("_Prop0.5_" in each_file):
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
                            temp = dimension_files_list[index].copy()
                            temp.append(each_file)
                            dimension_files_list[index] = temp
            elif "Average" in y_label:
                if "Convergence" in each_file:
                    for index, each_freq in enumerate(self.frequency_list):
                        if "_F" + str(each_freq) + '_' in each_file:
                            temp = dimension_files_list[index].copy()
                            temp.append(each_file)
                            dimension_files_list[index] = temp
            elif "Potential" in y_label:
                if "Potential" in each_file:
                    for index, each_freq in enumerate(self.frequency_list):
                        if "_F" + str(each_freq) + '_' in each_file:
                            temp = dimension_files_list[index].copy()
                            temp.append(each_file)
                            dimension_files_list[index] = temp
        frequency_files_list = np.array(dimension_files_list, dtype=object)
        # frequency_files_list = frequency_files_list.reshape((len(self.frequency_list), len(self.K_list), -1))
        print("Frequency Files list shape: ", frequency_files_list.shape)
        # create the K dimension
        # frequency_files_list = frequency_files_list.reshape((len(self.frequency_list), len(self.K_list), -1))
        # print("Frequency Files list shape: ", frequency_files_list.shape)
        # print("Frequency Files list 1: ", frequency_files_list[0][0])
        # print("Frequency Files list 2: ", frequency_files_list[1])
        all_curves_data = []
        for each_curve_files in frequency_files_list:
            print(each_curve_files)
            data_curve = self.load_data_from_files(files_list=each_curve_files)
            all_curves_data.append(data_curve)
        data_list = np.array(all_curves_data, dtype=object)
        print("Curves Data Shape: ", data_list.shape)
        self.data = data_list
        label_list = self.frequency_list
        # curves_data_shape: (5, 100, 100, 400); (5, 4, 100)
        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
            # print("Curve Shape: ", np.array(each_curve).shape)
            average_value, variation_value = None, None
            if "Unique" in y_label:
            # unique fitness lack one dimension
                average_value = np.mean(np.array(each_curve), axis=1)
                variation_value = np.var(np.array(each_curve).reshape((len(self.K_list), -1)), axis=1)
            elif ("Average" in y_label) or ("Potential" in y_label):
                average_value = np.mean(np.mean(np.array(each_curve), axis=2), axis=1) # 10 * (500 * 500)
                variation_value = np.var(np.array(each_curve).reshape((len(self.K_list), -1)), axis=1)
            ax.plot(self.K_list, average_value, label="{}".format(lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        my_x_ticks = np.arange(0, max(self.K_list)+1, self.K_list[1]-self.K_list[0])
        plt.xticks(my_x_ticks)
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_one_dimension_figure_v2(self, dimension=None, title=None,y_label=None,show_variance=False):
        if title:
            self.title = title
        frequency_dimention = len(self.frequency_list)
        openness_dimension = len(self.openness_list)
        quality_list_dimension = len(self.openness_list)
        gs_proportion_dimension = len(self.gs_proportion_list)
        if dimension == "Frequency":
            dimension_files_list = [[]] * frequency_dimention
            file_name_reference = self.frequency_list
        elif dimension == "Quality":
            dimension_files_list = [[]] * quality_list_dimension
            file_name_reference = self.quality_list
        elif dimension == "Openness":
            dimension_files_list = [[]] * openness_dimension
            file_name_reference = self.openness_list
        elif dimension == "Proportion":
            dimension_files_list = [[]] * gs_proportion_dimension
            file_name_reference = self.gs_proportion_list
        else:
            raise ValueError("Unsupported")

        # reduce the files number
        selected_files_list = []
        if dimension == "Quality":
            for each_file in self.files_list:
                if ("_O1.0" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file) and ("_Prop1.0_" in each_file):
                    selected_files_list.append(each_file)
        elif dimension == "Openness":
            for each_file in self.files_list:
                if ("_Q1.0_" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file) and ("_Prop1.0_" in each_file):
                    selected_files_list.append(each_file)
        elif dimension == "Frequency":
            for each_file in self.files_list:
                if ("_O1.0" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file):
                    selected_files_list.append(each_file)
        elif dimension == "Proportion":
            for each_file in self.files_list:
                if ("_O1.0" not in each_file) and ("_SS0.5_" in each_file) and ("_GG0.5_" in each_file):
                    selected_files_list.append(each_file)
        # selected_files_list = self.files_list

        for each_file in selected_files_list:
            if "Unique" in y_label:
                if "Unique" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Average" in y_label:
                if "Convergence" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Potential" in y_label:
                if "Potential" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")
        dimension_files_list = np.array(dimension_files_list, dtype=object)
        # print(dimension_files_list)
        print("Frequency Files list shape: ", dimension_files_list.shape)
        dimension_files_list = dimension_files_list.reshape((len(file_name_reference), len(self.K_list), -1))  # reference is also the label list
        print("Frequency Files list shape: ", dimension_files_list.shape)
        # create the K dimension
        # frequency_files_list = frequency_files_list.reshape((len(self.frequency_list), len(self.K_list), -1))
        # print("Frequency Files list shape: ", frequency_files_list.shape)
        # print("Frequency Files list 1: ", frequency_files_list[0][0])
        # print("Frequency Files list 2: ", frequency_files_list[1])
        all_curves_data = []
        for each_curve_files in dimension_files_list:
            # print(each_curve_files)
            data_curve = self.load_data_from_folders(folder_list=each_curve_files)
            all_curves_data.append(data_curve)
        all_curves_data = np.array(all_curves_data, dtype=object)
        print("Curves Data Shape (before): ", all_curves_data.shape)
        all_curves_data = all_curves_data.reshape((all_curves_data.shape[0], all_curves_data.shape[1], -1))
        print("Curves Data Shape (after): ", all_curves_data.shape)
        self.data = all_curves_data
        label_list = file_name_reference
        # curves_data_shape: (5, 4, 5, 100); (5, 4, 100)
        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
            # print("Curve Shape: ", np.array(each_curve).shape)
            average_value, variation_value = None, None
            average_value = np.mean(np.array(each_curve), axis=1)  #  (4, 500) -> (K, repeat)
            print("average_value: ", average_value)
            variation_value = np.var(np.array(each_curve), axis=1)
            print("variation_value: ", variation_value)
            ax.plot(self.K_list, average_value, label="{0}:{1}".format(dimension, lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        my_x_ticks = np.arange(0, max(self.K_list)+1, self.K_list[1]-self.K_list[0])
        plt.xticks(my_x_ticks)
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + "-" + dimension + "-" + y_label
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + dimension + "-" + y_label + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_solid_figure(self, dimension=None, title=None,z_label=None):
        if title:
            self.title = title
        # select the dimension for horizonal axis
        frequency_dimention = len(self.frequency_list)
        openness_dimension = len(self.openness_list)
        quality_list_dimension = len(self.openness_list)
        gs_proportion_dimension = len(self.gs_proportion_list)
        if dimension == "Frequency":
            dimension_files_list = [[]] * frequency_dimention
            file_name_reference = self.frequency_list
        elif dimension == "Quality":
            dimension_files_list = [[]] * quality_list_dimension
            file_name_reference = self.quality_list
        elif dimension == "Openness":
            dimension_files_list = [[]] * openness_dimension
            file_name_reference = self.openness_list
        elif dimension == "Proportion":
            dimension_files_list = [[]] * gs_proportion_dimension
            file_name_reference = self.gs_proportion_list
        else:
            raise ValueError("Unsupported")

        # fix some feature to reduce the file number
        selected_files_list = []
        for each_file in self.files_list:
            for k in self.K_list:
                # if "_K" + str(k) + "_" in each_file:
                #     if dimension == "Proportion":
                #         if "_Q1.0_" in each_file:
                #             selected_files_list.append(each_file)
                #     else:
                #         if "_Prop0.5_" in each_file:
                #             selected_files_list.append(each_file)

                if "_K" + str(k) + "_" in each_file:
                    selected_files_list.append(each_file)

        print("whole_files_list: ", len(self.files_list))
        print("selected_files_list: ", len(selected_files_list))
        for each_file in selected_files_list:
            if "Unique" in z_label:
                if "Unique" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Average" in z_label:
                if "Convergence" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Potential" in z_label:
                if "Potential" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Frequency":
                            if "_F" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        else:
                            raise ValueError("Unsupported")

        dimension_files_list = np.array(dimension_files_list, dtype=object)
        print("Dimension Files list shape: ", dimension_files_list.shape)  # (dimension, K*Z)
        dimension_files_list = dimension_files_list.reshape((len(file_name_reference), len(self.K_list), -1)) #  (dimension, K, Z), K is for vertical axis
        print("Dimension Files list shape: ", dimension_files_list.shape)
        print("Dimension files example 003: \n", dimension_files_list[0][0][0:3])
        print("Dimension files example 113: \n", dimension_files_list[1][1][0:3])
        x = []
        y = []
        for i in self.G_exposed_to_G_list:
            for j in self.S_exposed_to_S_list:
                x.append(i)  # horizonal
                y.append(j)  # vertical
        X = np.array(x).reshape((len(self.G_exposed_to_G_list), len(S_exposed_to_S_list)))
        Y = np.array(y).reshape((len(self.G_exposed_to_G_list), len(S_exposed_to_S_list)))
        # print(X)
        # print(Y)
        i = 0
        Z = np.zeros((len(file_name_reference), len(self.K_list), len(self.G_exposed_to_G_list), len(S_exposed_to_S_list)))
        for x_index, x_value in enumerate(self.G_exposed_to_G_list):
            for y_index, y_value in enumerate(self.S_exposed_to_S_list):
                for d, each_dimension_files in enumerate(dimension_files_list):
                    # print("each_dimension_files example 03:\n", each_dimension_files[0][0:3])
                    # print("each_dimension_files example 13:\n", each_dimension_files[1][0:3])
                    for k, each_k_curve_files in enumerate(each_dimension_files):
                        # print("each_k_curve_files: 3\n ", each_k_curve_files[0:3])
                        if x_value + y_value == 0:  # Non-socialization
                            targeted_files = [each for each in each_k_curve_files if "_F0_" in each]
                            if len(targeted_files) == 0:
                                if "Average" in z_label:
                                # !!! currently, there is no data point for non-socialization
                                    Z[d][k][x_index][y_index] = 0.50
                                else:
                                    Z[d][k][x_index][y_index] = None
                            else:
                                data = self.load_data_from_files(files_list=targeted_files)
                                data = np.array(data, dtype=object)
                                data = np.mean(data, axis=1)
                                Z[d][k][x_index][y_index] = data
                        else:
                            # same dimension, same K, but still different GG, SS; fit the GG/SS combination
                            targeted_files = [each for each in each_k_curve_files if "_SS" + str(y_value) + "_GG" + str(x_value) + "_" in each]
                            # print(file_name_reference[d], k, x_value, y_value, targeted_files)
                            # print("_SS"+str(y_value)+"_GG"+str(x_value)+"_F5_Prop0.5_Q"+str(file_name_reference[d])+"_")
                            # print("targeted_files:\n", targeted_files)
                            data = self.load_data_from_files(files_list=targeted_files)
                            data = np.array(data, dtype=object)
                            # print("data shape: ", data.shape)  # (16, 100, 400); (files number/Z, landscape, agent)
                            data = data.reshape((1, -1))
                            data = np.mean(data, axis=1)
                            Z[d][k][x_index][y_index] = data
        # print(Z)
        # To avoid the curve go out side of the right place
        # zlim = {
        #     0: [0.8, 1],
        #     1: [4, 20],
        #     2: [10, 35],
        # }
        # zticks = {
        #     0: [0.8, 0.85, 0.9, 0.95, 1.0],
        #     1: [4, 8, 12, 16, 20],
        #     2: [10, 15, 20, 25, 30, 35],
        # }
        # Z = Z.reshape((len(self.K_list), len(file_name_reference), len(self.G_exposed_to_G_list), len(S_exposed_to_S_list)))
        f = plt.figure(figsize=(30, 30))
        for d in range(len(file_name_reference)):  # Row
            for k in range(len(self.K_list)):  # Column
                ax = f.add_subplot(len(file_name_reference),len(self.K_list), d*len(self.K_list)+k+1, projection="3d")   # nrow, ncol, index
                # print("Z shape: ", Z[d][k].shape)
                ax.plot_surface(X, Y, Z[d][k], color="grey")

                ax.view_init(elev=10., azim=120)

                if "Average" in z_label:
                    ax.set_zlim(0.3, 0.7)
                elif "Unique" in z_label:
                    ax.set_zlim(20, 80)
                elif "Potential" in z_label:
                    ax.set_zlim(0, 30)
                ax.set_xlabel("G -> S Probability", fontsize="18", labelpad=10)
                plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                ax.set_ylabel("S -> G probability", fontsize="18", labelpad=10)
                plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                # place the surface in the right place
                # ax.set_zlim(zlim[k][0], zlim[k][1])
                ax.set_zlabel(z_label, fontsize="12")
                # ax.set_zticks(zticks[k])
                # ax.set_zticklabels(zticks[k], fontsize="12")

                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.02)
                # if d == 0:  # d is column, so it occurs across rows
                #     ax.annotate("K = %d" % (self.K_list[k]), xy=(0.5, 0.9), xytext=(0, 5),
                #                 xycoords='axes fraction',
                #                 textcoords='offset points', ha='center', va='baseline', fontsize="20",
                #                 fontweight="bold")
                # if k == 0:  # k is row, so it occurs across columns
                #     ax.annotate("{0}-{1}".format(dimension, file_name_reference[d]), xy=(0, 0.5), xytext=(-ax.zaxis.labelpad-450, 0),
                #                 xycoords=ax.zaxis.label, textcoords='offset points',
                #                 size='large', ha='right', va='center', rotation=90, fontsize="18", fontweight='bold')

                if d == 0:  # d is row, so it occurs across columns
                    ax.annotate("K = %d" % (self.K_list[k]), xy=(0.5, 0.9), xytext=(0, 5),
                                xycoords='axes fraction',
                                textcoords='offset points', ha='center', va='baseline', fontsize="20",
                                fontweight="bold")
                if k == 0:  # k is column, so it occurs across columns
                    ax.annotate("{0}-{1}".format(dimension, file_name_reference[d]), xy=(0, 0.5), xytext=(-ax.zaxis.labelpad-450, 0),
                                xycoords=ax.zaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', rotation=90, fontsize="18", fontweight='bold')
        output = self.output_path + "\\" + self.title + "-" + dimension + "-" + z_label
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + dimension + "-" + z_label + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()



if __name__ == '__main__':
    data_foler = r'C:\Python_Workplace\hpc-0403\N6_self'
    # data_foler = r'C:\Python_Workplace\hpc-0407\N6_random'

    output_folder = r'C:\Python_Workplace\hpc-0403'
    # Search loop
    landscape_iteration = 100
    agent_num = 400
    search_iteration = 50
    # Parameter
    N = 6
    state_num = 4
    knowledge_num = 8
    # K_list = [1, 3, 5]
    K_list = [2,4,6]
    # frequency_list = [1, 5, 10, 20, 40]
    frequency_list = [5]
    openness_list = [0.5, 0.75, 1.0]
    quality_list = [0.5, 0.75, 1.0]
    G_exposed_to_G_list = [0, 0.25, 0.5, 0.75, 1.0]
    S_exposed_to_S_list = [0, 0.25, 0.5, 0.75, 1.0]
    # gs_proportion_list = [0, 0.25, 0.5, 0.75, 1.0]
    gs_proportion_list = [0, 0.5, 1.0]
    files_number = len(K_list) * len(openness_list) * len(quality_list) *\
                   len(G_exposed_to_G_list) * len(S_exposed_to_S_list) * len(gs_proportion_list)*3
    print("Expected Files Number: ", files_number)
    exposure_type = "Self-interested"

    evaluator = Evaluator(data_path=data_foler, output_path=output_folder, )
    evaluator.load_iv_configuration(exposure_type=exposure_type, N=N, state_num=state_num, K_list=K_list, frequency_list=frequency_list,
                                    openness_list=openness_list, quality_list=quality_list, G_exposed_to_G_list=G_exposed_to_G_list,
                                    S_exposed_to_S_list=S_exposed_to_S_list, gs_proportion_list=gs_proportion_list, knowledge_num=knowledge_num)
    evaluator.load_simulation_configuration(landscape_iteration=landscape_iteration, agent_num=agent_num, search_iteration=search_iteration)
    # evaluator.generate_one_dimension_figure(title="N6-Overall-Frequency", dimension="Quality",y_label="Unique Fitness",
    #                                         show_variance=True)
    # evaluator.generate_one_dimension_figure_v2(title="N6-Random", dimension="Quality",
    #                                            y_label="Average Fitness", show_variance=False)
    evaluator.generate_solid_figure(title="3D-N6-Self", dimension="Quality",
                                               z_label="Average Fitness")
    print("END")