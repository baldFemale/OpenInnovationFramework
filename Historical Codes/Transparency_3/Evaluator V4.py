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

    def generate_one_dimension_figure(self, dimension=None, title=None,y_label=None,
                                      show_variance=False, percentage=None, top_coverage=None):
        if title:
            self.title = title
        openness_dimension = len(self.openness_list)
        quality_list_dimension = len(self.quality_list)
        gs_proportion_dimension = len(self.gs_proportion_list)
        if dimension == "Quality":
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

        selected_files_list = self.files_list
        print("file_name_reference: ", file_name_reference)
        print("selected_files_list: ", len(selected_files_list))
        for each_file in selected_files_list:
            if "Coverage" in y_label:
                if "2AverageRank" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + '_' in each_file:
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
                if "1Average" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + '_' in each_file:
                                # print("_O" + str(each_reference))
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
            elif ("Potential Fitness" in y_label) or (("Potential" in y_label) and "Rank" not in y_label):
                if "3Potential" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + '_' in each_file:
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
            elif "Maximum" in y_label:
                if "1Average" in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
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
            elif "Divergence" in y_label:
                if ("3" + y_label) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
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
            elif "Quality" in y_label:
                if ("4" + y_label) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
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
            elif "Utilization" in y_label:
                if ("5" + y_label) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_files_list[index].copy()
                                temp.append(each_file)
                                dimension_files_list[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
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
            else:
                raise ValueError("Unsupported")
        print(dimension_files_list)
        dimension_files_list = np.array(dimension_files_list, dtype=object)
        print("Dimension Files list shape: ", dimension_files_list.shape)
        dimension_files_list = dimension_files_list.reshape((len(file_name_reference), len(self.K_list), -1))  # reference is also the label list
        print("Dimension Files list shape: ", dimension_files_list.shape)
        print("Dimension Files Example: \n", dimension_files_list[0])

        all_curves_data = []
        for each_curve_files in dimension_files_list:
            # print(each_curve_files)
            data_curve = self.load_data_from_folders(folder_list=each_curve_files)
            all_curves_data.append(data_curve)
        all_curves_data = np.array(all_curves_data, dtype=object)
        print("Curves Data Shape (before): ", all_curves_data.shape)

        # Coverage, Maximum, and Divergence need further calculations
        if "Coverage" in y_label:
            # copy_all_curve_data = np.zeros((all_curves_data.shape[0], all_curves_data.shape[1], all_curves_data.shape[2], all_curves_data.shape[3]))
            for d in range(all_curves_data.shape[0]):
                for k in range(all_curves_data.shape[1]):
                    for f in range(all_curves_data.shape[2]):
                        for l in range(all_curves_data.shape[3]):
                            temp_coverage = 0
                            coverage_cache = []
                            for a in range(all_curves_data.shape[4]):
                                if percentage:
                                    if all_curves_data[d][k][f][l][a] <= percentage * (self.state_num ** self.N) / 100:
                                        if all_curves_data[d][k][f][l][a] not in coverage_cache:
                                            temp_coverage += 1
                                            coverage_cache.append(all_curves_data[d][k][f][l][a])
                                elif top_coverage:
                                    if all_curves_data[d][k][f][l][a] <= top_coverage:
                                        if all_curves_data[d][k][f][l][a] not in coverage_cache:
                                            temp_coverage += 1
                                            coverage_cache.append(all_curves_data[d][k][f][l][a])
                                if top_coverage and percentage:
                                    raise ValueError("Cannot provide both")
                            # print(temp_coverage)
                            all_curves_data[d][k][f][l] = temp_coverage
        if "Maximum" in y_label:
            for d in range(all_curves_data.shape[0]):
                for k in range(all_curves_data.shape[1]):
                    for f in range(all_curves_data.shape[2]):
                        for l in range(all_curves_data.shape[3]):
                            max_temp = max(all_curves_data[d][k][f][l])
                            all_curves_data[d][k][f][l] = max_temp
        if "Divergence" in y_label:
            for d in range(all_curves_data.shape[0]):
                for k in range(all_curves_data.shape[1]):
                    for f in range(all_curves_data.shape[2]):
                        for l in range(all_curves_data.shape[3]):
                            pool_temp = list(all_curves_data[d][k][f][l])
                            divergence_temp = 0
                            for each_solution_pool in pool_temp:
                                # print(len(each_solution_pool))
                                temp = ["".join(each) for each in each_solution_pool]
                                temp = set(temp)
                                divergence_temp += len(temp)
                            all_curves_data[d][k][f][l] = divergence_temp/100/200  # 100 values


        print("Curves Data Shape (before2): ", all_curves_data.shape)
        all_curves_data = all_curves_data.reshape((all_curves_data.shape[0], all_curves_data.shape[1], -1))
        print("Curves Data Shape (after): ", all_curves_data.shape)
        self.data = all_curves_data
        label_list = file_name_reference
        # curves_data_shape: (5, 4, 5, 100); (5, 4, 100)


        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
            # print("Curve Shape: ", np.array(each_curve).shape)
            average_value = np.mean(np.array(each_curve), axis=1)  #  (4, 500) -> (K, repeat)
            # print("average_value: ", average_value)
            variation_value = np.var(np.array(each_curve), axis=1)
            # print("variation_value: ", variation_value)
            if dimension == "Proportion":
                ax.plot(self.K_list, average_value, label="{0} of G :{1}".format(dimension, lable))
            else:
                ax.plot(self.K_list, average_value, label="{0}:{1}".format(dimension, lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        my_x_ticks = np.arange(min(self.K_list), max(self.K_list)+1, self.K_list[1]-self.K_list[0])
        plt.xticks(my_x_ticks)
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + "-" + dimension + "-" + str(y_label)
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + dimension + "-" + y_label + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_solid_figure(self, title=None, percentage=None, top_coverage=None):
        if title:
            self.title = title
        x = []
        y = []
        for i in self.G_exposed_to_G_list:
            for j in self.S_exposed_to_S_list:
                x.append(i)  # horizonal
                y.append(j)  # vertical
        X = np.array(x).reshape((len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))
        Y = np.array(y).reshape((len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))

        Z = np.zeros((3, len(self.K_list), len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))
        for x_index, x_value in enumerate(["_GG" + str(each) + "_" for each in self.G_exposed_to_G_list]):
            for y_index, y_value in enumerate(["_SS" + str(each) + "_" for each in self.S_exposed_to_S_list]):
                for d, column_name in enumerate(["1Average", "2AverageRank", "3Potential"]):
                    for k, k_name in enumerate(["_K" + str(each_k) + "_" for each_k in self.K_list]):
                        temp = []
                        for file in self.files_list:
                            if (x_value in file) and (y_value in file) and (column_name in file) and (k_name) in file:
                                temp.append(file)
                        data = self.load_data_from_files(files_list=temp)
                        data = np.array(data, dtype=object)
                        data = data.reshape(1, -1)
                        # print(np.array(data).shape)
                        if (column_name == "1Average") or (column_name == "3Potential"):
                            Z[d][k][x_index][y_index] = np.mean(data, axis=1)
                        elif column_name == "2AverageRank":
                            data = np.unique(data)  # only count the unique coverage
                            if percentage:
                                Z[d][k][x_index][y_index] = np.count_nonzero(data <= percentage / 100 * (self.state_num ** self.N) / self.landscape_iteration)
                            elif top_coverage:
                                Z[d][k][x_index][y_index] = np.count_nonzero(data <= top_coverage)
        # print(Z)
        # # To avoid the curve go out side of the right place
        # # zlim = {
        # #     0: [0.8, 1],
        # #     1: [4, 20],
        # #     2: [10, 35],
        # # }
        # # zticks = {
        # #     0: [0.8, 0.85, 0.9, 0.95, 1.0],
        # #     1: [4, 8, 12, 16, 20],
        # #     2: [10, 15, 20, 25, 30, 35],
        # # }
        # # Z = Z.reshape((len(self.K_list), len(file_name_reference), len(self.G_exposed_to_G_list), len(S_exposed_to_S_list)))
        f = plt.figure(figsize=(30, 30))
        # limitation_list = [[(0.570, 0.580), (0.525, 0.540), (0.51, 0.5225), (0.505, 0.525)],
        #                    [(200, 350), (40,140), (40,120), (40,120)],
        #                    [(0.905, 0.915), (0.89, 0.90), (0.886, 0.888), (0.884, 0.887)]]
        # limitation_list = [[(0.570, 0.580), (0.525, 0.540), (0.51, 0.5225), (0.505, 0.525)],
        #                    [(60, 160), (20, 80), (20, 80), (20, 80)],
        #                    [(0.905, 0.915), (0.89, 0.90), (0.886, 0.888), (0.884, 0.887)]]
        z_label_list = ["Average Fitness", "Coverage", "Potential Fitness"]
        for row, K_label in enumerate(self.K_list):
            for column, z_label in enumerate(z_label_list):
                ax = f.add_subplot(len(self.K_list), 3, row*3+column+1, projection="3d")   # nrow, ncol, index
                ax.plot_surface(X, Y, Z[column][row], color="grey")
                # ax.set_zlim(limitation_list[column][row])
                # if (column == 0) and (row == 0):
                #     ax.set_zlim(0.575, 0.595)
                # elif (column == 0) and (row != 0):
                #     ax.set_zlim(0.525, 0.55)
                # elif (column == 1) and (row == 0):
                #     ax.set_zlim(120, 220)
                # elif (column ==1) and (row != 0):
                #     ax.set_zlim(20, 120)
                # elif (column == 2) and (row == 0):
                #     ax.set_zlim(0.94, 0.95)
                # elif (column == 2) and (row == 1):
                #     ax.set_zlim(0.92, 0.93)
                # elif (column == 2) and (row == 2):
                #     ax.set_zlim(0.91, 0.92)
                ax.set_xlabel("G -> G Probability", fontsize="18", labelpad=10)
                plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                ax.set_ylabel("S -> S probability", fontsize="18", labelpad=10)
                plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                # place the surface in the right place
                # ax.set_zlim(zlim[k][0], zlim[k][1])
                ax.set_zlabel(z_label, fontsize="12")
                # ax.set_zticks(zticks[k])
                # ax.set_zticklabels(zticks[k], fontsize="12")

                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.02)

                if row == 0:  # d is row, so it occurs across columns
                    ax.annotate("{0}".format(z_label_list[column]), xy=(0.5, 0.9), xytext=(0, 5),
                                xycoords='axes fraction',
                                textcoords='offset points', ha='center', va='baseline', fontsize="20",
                                fontweight="bold")
                if column == 0:  # k is column, so it occurs across columns
                    ax.annotate("K={0}".format(self.K_list[row]), xy=(0, 0.5), xytext=(-ax.zaxis.labelpad-450, 0),
                                xycoords=ax.zaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', rotation=90, fontsize="18", fontweight='bold')
        output = self.output_path + "\\" + self.title
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_interaction_figure_1(self, title=None, percentage=None, top_coverage=None, x_label=None, y_label=None):
        """
        Not direction, because direction itself contain two dimensition;
        Except for direction, x_label and y_label will make up the x and y axis
        """
        if title:
            self.title = title + "_" + x_label + "_" + y_label
        x_dimension, x_flag, y_dimension, y_flag = None, None, None, None
        if (x_label == "Composition") or (x_label == "Proportion"):
            x_dimension = self.gs_proportion_list
            x_flag = "_Prop"
        elif x_label == "Openness":
            x_dimension = self.openness_list
            x_flag = "_O"
        elif x_label == "Frequency":
            x_dimension = self.frequency_list
            x_flag = "_F"
        elif x_label == "Quality":
            x_dimension = self.quality_list
            x_flag = "_Q"

        if (y_label == "Composition") or (y_label == "Proportion"):
            y_dimension = self.gs_proportion_list
            y_flag = "_Prop"
        elif y_label == "Openness":
            y_dimension = self.openness_list
            y_flag = "_O"
        elif y_label == "Frequency":
            y_dimension = self.frequency_list
            y_flag = "_F"
        elif y_label == "Quality":
            y_dimension = self.quality_list
            y_flag = "_Q"
        if (x_label == "Direction") or (y_label == "Direction"):
            raise ValueError("Do not support direction dimension")
        x = []
        y = []
        for i in x_dimension:
            for j in y_dimension:
                x.append(i)  # horizonal
                y.append(j)  # vertical
        X = np.array(x).reshape((len(x_dimension), len(x_dimension)))
        Y = np.array(y).reshape((len(y_dimension), len(y_dimension)))
        Z = np.zeros((3, len(self.K_list), len(x_dimension), len(y_dimension)))

        selected_files_list = self.files_list
        print("selected_files_list: ", len(selected_files_list))
        for x_index, x_value in enumerate([x_flag + str(each) + "_" for each in x_dimension]):
            for y_index, y_value in enumerate([y_flag + str(each) + "_" for each in y_dimension]):
                for d, column_name in enumerate(["1Average", "2AverageRank", "3Potential"]):
                    for k, k_name in enumerate(["_K" + str(each_k) + "_" for each_k in self.K_list]):
                        temp = []
                        for file in selected_files_list:
                            if (x_value in file) and (y_value in file) and (column_name in file) and (k_name) in file:
                                temp.append(file)
                        if not temp:
                            print(x_value, y_value, column_name, k_name)
                            print("None file")
                            print("_"*10)
                            continue
                        else:
                            print(x_value, y_value, column_name, k_name)
                            print(len(temp))
                            print("_"*10)
                        data = self.load_data_from_files(files_list=temp)
                        data = np.array(data, dtype=object)
                        data = data.reshape(1, -1)
                        # print(np.array(data).shape, temp)
                        if (column_name == "1Average") or (column_name == "3Potential"):
                            Z[d][k][x_index][y_index] = np.mean(data, axis=1)
                        elif column_name == "2AverageRank":
                            data = np.unique(data)  # fix bug and only count the unique rank/fitness
                            if percentage:
                                Z[d][k][x_index][y_index] = np.count_nonzero(data <= percentage / 100 * (self.state_num ** self.N) / self.landscape_iteration)
                            elif top_coverage:
                                Z[d][k][x_index][y_index] = np.count_nonzero(data <= top_coverage)
        f = plt.figure(figsize=(30, 30))
        # print("Z: ", Z)
        z_label_list = ["Average Fitness", "Coverage", "Potential Fitness"]
        for row, K_label in enumerate(self.K_list):
            for column, z_label in enumerate(z_label_list):
                ax = f.add_subplot(len(self.K_list), 3, row*3+column+1, projection="3d")   # nrow, ncol, index
                ax.plot_surface(X, Y, Z[column][row], color="grey")
                # print(X, Y, Z[column][row])
                # print(row, column, Z[column][row])
                ax.set_xlabel(x_label, fontsize="18", labelpad=10)
                ax.set_ylabel(y_label, fontsize="18", labelpad=10)
                # if x_dimension == "Frequency":
                #     plt.xticks([1, 5, 10, 20, 40], [1, 5, 10, 20, 40], fontsize="12")
                #     plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                # if y_dimension == "Frequency":
                #     plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                #     plt.yticks([1, 5, 10, 20, 40], [1, 5, 10, 20, 40], fontsize="8")
                # else:
                #     plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                #     plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                ax.set_zlabel(z_label, fontsize="12")
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.02)
                if row == 0:  # d is row, so it occurs across columns
                    ax.annotate("{0}".format(z_label_list[column]), xy=(0.5, 0.9), xytext=(0, 5),
                                xycoords='axes fraction',
                                textcoords='offset points', ha='center', va='baseline', fontsize="20",
                                fontweight="bold")
                if column == 0:  # k is column, so it occurs across columns
                    ax.annotate("K={0}".format(self.K_list[row]), xy=(0, 0.5), xytext=(-ax.zaxis.labelpad-450, 0),
                                xycoords=ax.zaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', rotation=90, fontsize="18", fontweight='bold')
        output = self.output_path + "\\" + self.title
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_interaction_figure_2(self, title=None, percentage=None, top_coverage=None, given_k=None, row_label=None):
        if title:
            self.title = title + "_K" + str(given_k) + "_" + "Direction" + "_" + row_label
        row_dimension, row_flag = None, None
        if (row_label == "Composition") or (row_label == "Proportion"):
            row_dimension = self.gs_proportion_list
            row_flag = "_Prop"
        elif row_label == "Openness":
            row_dimension = self.openness_list
        elif row_label == "Quality":
            row_dimension = self.openness_list
        elif row_label == "Frequency":
            row_dimension = self.frequency_list

        x = []
        y = []
        for i in self.G_exposed_to_G_list:
            for j in self.S_exposed_to_S_list:
                x.append(i)  # horizonal
                y.append(j)  # vertical
        X = np.array(x).reshape((len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))
        Y = np.array(y).reshape((len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))
        Z = np.zeros((3, len(row_dimension), len(self.G_exposed_to_G_list), len(self.S_exposed_to_S_list)))
        selected_files_list = []
        for file in self.files_list:
            if "_K" + str(given_k) + "_" in file:
                selected_files_list.append(file)
        print("selected_files_list: ", len(selected_files_list))
        print("row dimension: ", row_dimension)
        for x_index, x_value in enumerate(["_GG" + str(each) + "_" for each in self.G_exposed_to_G_list]):
            for y_index, y_value in enumerate(["_SS" + str(each) + "_" for each in self.S_exposed_to_S_list]):
                for c, column_name in enumerate(["1Average", "2AverageRank", "3Potential"]):
                    for r, row_name in enumerate([row_flag + str(row_value) + "_" for row_value in row_dimension]):
                        temp = []
                        for file in selected_files_list:
                            if (x_value in file) and (y_value in file) and (column_name in file) and (row_name) in file:
                                temp.append(file)
                        if not temp:
                            print(x_value, y_value, column_name, row_name)
                            continue
                        else:
                            print(x_value, y_value, column_name, row_name)
                            print(len(temp))
                            print("_"*10)
                        data = self.load_data_from_files(files_list=temp)
                        data = np.array(data, dtype=object)
                        data = data.reshape(1, -1)
                        # print(np.array(data).shape)
                        if (column_name == "1Average") or (column_name == "3Potential"):
                            Z[c][r][x_index][y_index] = np.mean(data, axis=1)
                        elif column_name == "2AverageRank":
                            data = np.unique(data)
                            if percentage:
                                Z[c][r][x_index][y_index] = np.count_nonzero(
                                    data <= percentage / 100 * (self.state_num ** self.N) / self.landscape_iteration)
                            elif top_coverage:
                                Z[c][r][x_index][y_index] = np.count_nonzero(data <= top_coverage)
        print("Test: ", Z.shape)
        f = plt.figure(figsize=(30, 30))
        z_label_list = ["Average Fitness", "Coverage", "Potential Fitness"]
        for row, row_value in enumerate(row_dimension):
            for column, z_label in enumerate(z_label_list):
                ax = f.add_subplot(len(row_dimension), 3, row * 3 + column + 1, projection="3d")  # nrow, ncol, index
                ax.plot_surface(X, Y, Z[column][row], color="grey")
                print(row, column, Z[column][row])
                ax.set_xlabel("G -> G Probability", fontsize="18", labelpad=10)
                plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                ax.set_ylabel("S -> S probability", fontsize="18", labelpad=10)
                plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize="12")
                ax.set_zlabel(z_label, fontsize="12")
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.02)
                if row == 0:  # d is row, so it occurs across columns
                    ax.annotate("{0}".format(z_label_list[column]), xy=(0.5, 0.9), xytext=(0, 5),
                                xycoords='axes fraction',
                                textcoords='offset points', ha='center', va='baseline', fontsize="20",
                                fontweight="bold")
                if column == 0:  # k is column, so it occurs across columns
                    ax.annotate("{0}={1}".format(row_flag, row_value), xy=(0, 0.5), xytext=(-ax.zaxis.labelpad - 450, 0),
                                xycoords=ax.zaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', rotation=90, fontsize="18", fontweight='bold')
        output = self.output_path + "\\" + self.title
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()


    def count_divergence(self, state_1=None, state_2=None):
        divergence = 0
        for i in range(len(state_1)):
            if state_1[i] != state_2[i]:
                divergence += 1
        return divergence

    # def generate_surface_main_effect_surface(self, title=fore_title, dimension=None, y_label=None):
    #     if title:
    #         self.title = title
    #     openness_dimension = len(self.openness_list)
    #     quality_list_dimension = len(self.quality_list)
    #     gs_proportion_dimension = len(self.gs_proportion_list)
    #     if dimension == "Quality":
    #         dimension_files_list = [[]] * quality_list_dimension
    #         file_name_reference = self.quality_list
    #     elif dimension == "Openness":
    #         dimension_files_list = [[]] * openness_dimension
    #         file_name_reference = self.openness_list
    #     elif dimension == "Proportion":
    #         dimension_files_list = [[]] * gs_proportion_dimension
    #         file_name_reference = self.gs_proportion_list
    #     else:
    #         raise ValueError("Unsupported")
    #
    #     selected_files_list = self.files_list
    #     print("file_name_reference: ", file_name_reference)
    #     print("selected_files_list: ", len(selected_files_list))
    #     for each_file in selected_files_list:
    #         if "Divergence" in y_label:
    #             if ("3" + y_label) in each_file:
    #                 for index, each_reference in enumerate(file_name_reference):
    #                     if dimension == "Quality":
    #                         if "_Q" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Openness":
    #                         if "_O" + str(each_reference) + "_" in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Proportion":
    #                         if "_Prop" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     else:
    #                         raise ValueError("Unsupported")
    #         elif "Quality" in y_label:
    #             if ("4" + y_label) in each_file:
    #                 for index, each_reference in enumerate(file_name_reference):
    #                     if dimension == "Quality":
    #                         if "_Q" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Openness":
    #                         if "_O" + str(each_reference) + "_" in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Proportion":
    #                         if "_Prop" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     else:
    #                         raise ValueError("Unsupported")
    #         elif "Utilization" in y_label:
    #             if ("5" + y_label) in each_file:
    #                 for index, each_reference in enumerate(file_name_reference):
    #                     if dimension == "Quality":
    #                         if "_Q" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Openness":
    #                         if "_O" + str(each_reference) + "_" in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     elif dimension == "Proportion":
    #                         if "_Prop" + str(each_reference) + '_' in each_file:
    #                             temp = dimension_files_list[index].copy()
    #                             temp.append(each_file)
    #                             dimension_files_list[index] = temp
    #                     else:
    #                         raise ValueError("Unsupported")
    #         else:
    #             raise ValueError("Unsupported")
    #     print(dimension_files_list)
    #     dimension_files_list = np.array(dimension_files_list, dtype=object)
    #     print("Dimension Files list shape: ", dimension_files_list.shape)
    #     dimension_files_list = dimension_files_list.reshape((len(file_name_reference), len(self.K_list), -1))  # reference is also the label list
    #     print("Dimension Files list shape: ", dimension_files_list.shape)
    #     print("Dimension Files Example: \n", dimension_files_list[0])
    #
    #     all_curves_data = []
    #     for each_curve_files in dimension_files_list:
    #         # print(each_curve_files)
    #         data_curve = self.load_data_from_folders(folder_list=each_curve_files)
    #         all_curves_data.append(data_curve)
    #     all_curves_data = np.array(all_curves_data, dtype=object)
    #     print("Curves Data Shape (before): ", all_curves_data.shape)
    #
    #     # Coverage, Maximum, and Divergence need further calculations
    #     if "Divergence" in y_label:
    #         for d in range(all_curves_data.shape[0]):
    #             for k in range(all_curves_data.shape[1]):
    #                 for f in range(all_curves_data.shape[2]):
    #                     for l in range(all_curves_data.shape[3]):
    #                         pool_temp = list(all_curves_data[d][k][f][l])
    #                         divergence_temp = []
    #                         for each_solution_pool in pool_temp:
    #                             # print(len(each_solution_pool))
    #                             temp = ["".join(each) for each in each_solution_pool]
    #                             temp = set(temp)
    #                             divergence_temp.append(len(temp)/self.agent_num)
    #                         all_curves_data[d][k][f][l] = divergence_temp  # 100 values
    #     if "Quality" in y_label:
    #         for d in range(all_curves_data.shape[0]):
    #             for k in range(all_curves_data.shape[1]):
    #                 for f in range(all_curves_data.shape[2]):
    #                     for l in range(all_curves_data.shape[3]):
    #                         pool_temp = list(all_curves_data[d][k][f][l])
    #                         divergence_temp = 0
    #                         for each_solution_pool in pool_temp:
    #                             # print(len(each_solution_pool))
    #                             temp = ["".join(each) for each in each_solution_pool]
    #                             temp = set(temp)
    #                             divergence_temp += len(temp)
    #                         all_curves_data[d][k][f][l] = divergence_temp/100/200  # 100 values
    #
    #
    #     print("Curves Data Shape (before2): ", all_curves_data.shape)
    #     all_curves_data = all_curves_data.reshape((all_curves_data.shape[0], all_curves_data.shape[1], self.search_iteration -1)) # dim, K, search
    #     print("Curves Data Shape (after): ", all_curves_data.shape)
    #     self.data = all_curves_data
    #     label_list = file_name_reference
    #     # curves_data_shape: (5, 4, 5, 100); (5, 4, 100)
    #
    #
    #     figure = plt.figure()
    #     for row, K_label in enumerate(self.K_list):
    #         for column, z_label in enumerate(z_label_list):
    #             ax = figure.add_subplot()
    #             for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
    #                 # print("Curve Shape: ", np.array(each_curve).shape)
    #                 average_value = np.mean(np.array(each_curve), axis=1)  #  (4, 500) -> (K, repeat)
    #                 # print("average_value: ", average_value)
    #                 if dimension == "Proportion":
    #                     ax.plot(self.K_list, average_value, label="{0} of G :{1}".format(dimension, lable))
    #                 else:
    #                     ax.plot(self.K_list, average_value, label="{0}:{1}".format(dimension, lable))
    #                 if show_variance:
    #                     # draw the variance
    #                     lower = [x - y for x, y in zip(average_value, variation_value)]
    #                     upper = [x + y for x, y in zip(average_value, variation_value)]
    #                     xaxis = list(range(len(lower)))
    #                     ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)
    #
    #             ax.set_xlabel('K')  # Add an x-label to the axes.
    #             ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
    #             my_x_ticks = np.arange(min(self.K_list), max(self.K_list)+1, self.K_list[1]-self.K_list[0])
    #             plt.xticks(my_x_ticks)
    #             ax.set_title(self.title)  # Add a title to the whole figure
    #             plt.legend()
    #
    #     output = self.output_path + "\\" + self.title + "-" + dimension + "-" + str(y_label)
    #     i = 1
    #     while os.path.exists(output + ".png"):
    #         i += 1
    #         print("File Exists")
    #         output = self.output_path + "\\" + self.title + "-" + dimension + "-" + y_label + "-" + str(i)
    #     plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
    #     plt.show()

if __name__ == '__main__':
    data_foler = r'C:\Python_Workplace\hpc-0422\Experiments_V3\Composition'
    output_folder = r'C:\Python_Workplace\hpc-0422\Experiments_V3'
    ###############################################################
    landscape_iteration = 1000
    agent_num = 200
    search_iteration = 100
    # Parameter
    N = 9
    state_num = 4
    knowledge_num = 16
    K_list = [2, 4, 6, 8]
    frequency_list = [1]
    openness_list = [1.0]
    quality_list = [1.0]
    G_exposed_to_G_list = [0.5]
    S_exposed_to_S_list = [0.5]
    gs_proportion_list = [0, 0.25, 0.5, 0.75, 1.0]
    ###############################################################
    exposure_type = "Self-interested"

    fore_title = "N" + str(N) + "-" + exposure_type
    evaluator = Evaluator(data_path=data_foler, output_path=output_folder, )
    evaluator.load_iv_configuration(exposure_type=exposure_type, N=N, state_num=state_num, K_list=K_list, frequency_list=frequency_list,
                                    openness_list=openness_list, quality_list=quality_list, G_exposed_to_G_list=G_exposed_to_G_list,
                                    S_exposed_to_S_list=S_exposed_to_S_list, gs_proportion_list=gs_proportion_list, knowledge_num=knowledge_num)
    evaluator.load_simulation_configuration(landscape_iteration=landscape_iteration, agent_num=agent_num, search_iteration=search_iteration)

    # Main effect for one dimension except for Direction
    evaluator.generate_one_dimension_figure(title=fore_title, dimension="Proportion", y_label="Utilization", show_variance=False, percentage=0.1, top_coverage=None)

    # Surface evolution with search iterations
    # evaluator.generate_surface_main_effect_surface(title=fore_title, dimension="Proportion", y_label="QualityG")

    # Main effect for Direction
    # evaluator.generate_solid_figure(title=fore_title, percentage=10)

    # Interaction effect for two dimension, which do not include Direction.
    # evaluator.generate_interaction_figure_1(title=fore_title, x_label="Composition", y_label="Openness", percentage=10)

    # Interaction effect for Direction (GG/SS) plus one more dimension
    # evaluator.generate_interaction_figure_2(title=fore_title, given_k=8, row_label="Composition", top_coverage=100)
    print("END")