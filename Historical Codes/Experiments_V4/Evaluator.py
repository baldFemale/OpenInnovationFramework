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

        self.dimension_divergence_files = None
        self.dimension_quality_files = None
        self.dimension_utilization_files = None
        self.divergence_curves_data = None
        self.divergence_curves_data_mode = None
        self.divergence_curves_data_haoming = None
        self.quality_curves_data = None
        self.utilization_curves_data = None
        self.Z = None
        self.data_temp =None

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
                for d, column_name in enumerate(["1Average"]):
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

    def generate_interaction_figure_1(self, title=None, x_label=None, y_label=None):
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
        Z = np.zeros((2, len(self.K_list), len(x_dimension), len(y_dimension)))

        selected_files_list = self.files_list
        print("selected_files_list: ", len(selected_files_list))
        for x_index, x_value in enumerate([x_flag + str(each) + "_" for each in x_dimension]):
            for y_index, y_value in enumerate([y_flag + str(each) + "_" for each in y_dimension]):
                for d, column_name in enumerate(["Average", "Maximum"]):
                    for k, k_name in enumerate(["_K" + str(each_k) + "_" for each_k in self.K_list]):
                        temp = []
                        for file in selected_files_list:
                            if (x_value in file) and (y_value in file) and ("1Average" in file) and (k_name) in file:
                                temp.append(file)
                        if not temp:
                            print(x_value, y_value, k_name)
                            continue
                        # else:
                        #     print(x_value, y_value, column_name, k_name)
                        #     print(len(temp))
                        data = self.load_data_from_files(files_list=temp)
                        data = np.array(data, dtype=object)
                        # print("data shape: ", np.array(data).shape, len(temp))
                        if column_name == "Average":
                            data = data.reshape(1, -1)
                            Z[d][k][x_index][y_index] = np.mean(data, axis=1)
                        elif column_name == "Maximum":
                            data = data.reshape(1000, -1)
                            max_ = np.max(data, axis=1)
                            # print("max_: ",  np.mean(max_, axis=0))
                            Z[d][k][x_index][y_index] = np.mean(max_, axis=0)
        f = plt.figure(figsize=(30, 30))
        # print("Z: ", Z)
        self.Z = Z
        z_label_list = ["Average", "Maximum"]
        for row, K_label in enumerate(self.K_list):
            for column, z_label in enumerate(z_label_list):
                ax = f.add_subplot(len(self.K_list), 2, row*2+column+1, projection="3d")   # nrow, ncol, index
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

    def get_mode_solution(self, pool):
        mode_solution = []
        for i in range(self.N):
            count_0 = sum([1 if state[i] == "0" else 0 for state in pool])
            count_1 = sum([1 if state[i] == "1" else 0 for state in pool])
            count_2 = sum([1 if state[i] == "2" else 0 for state in pool])
            count_3 = sum([1 if state[i] == "3" else 0 for state in pool])
            max_count = max(count_0, count_1, count_2, count_3)
            if count_0 == max_count:
                mode_solution.append("0")
            elif count_1 == max_count:
                mode_solution.append("1")
            elif count_2 == max_count:
                mode_solution.append("2")
            else:
                mode_solution.append("3")
        return mode_solution

    def get_solution_distance(self, solution_a=None, solution_b=None):
        counts = 0
        for a, b in zip(solution_a, solution_b):
            if a != b:
                counts += 1
        return counts

    def generate_surface_evolution(self, title=None, dimension=None, GS_flag='', y_label=None):
        if title:
            self.title = title
        y_label_list = ["Divergence", "Quality", "Utilization"]
        openness_dimension = len(self.openness_list)
        quality_list_dimension = len(self.quality_list)
        gs_proportion_dimension = len(self.gs_proportion_list)
        if dimension == "Quality":
            dimension_divergence_files = [[]] * quality_list_dimension
            dimension_quality_files = [[]] * quality_list_dimension
            dimension_utilization_files = [[]] * quality_list_dimension
            file_name_reference = self.quality_list
        elif dimension == "Openness":
            dimension_divergence_files = [[]] * openness_dimension
            dimension_quality_files = [[]] * openness_dimension
            dimension_utilization_files = [[]] * openness_dimension
            file_name_reference = self.openness_list
        elif dimension == "Proportion":
            dimension_divergence_files = [[]] * gs_proportion_dimension
            dimension_quality_files = [[]] * gs_proportion_dimension
            dimension_utilization_files = [[]] * gs_proportion_dimension
            file_name_reference = self.gs_proportion_list
        else:
            raise ValueError("Unsupported")

        selected_files_list = self.files_list
        print("file_name_reference: ", file_name_reference)
        print("selected_files_list: ", len(selected_files_list))
        for each_file in selected_files_list:
            if "Divergence" in y_label:
                if ("3Divergence" + GS_flag) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_divergence_files[index].copy()
                                temp.append(each_file)
                                dimension_divergence_files[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
                                temp = dimension_divergence_files[index].copy()
                                temp.append(each_file)
                                dimension_divergence_files[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_divergence_files[index].copy()
                                temp.append(each_file)
                                dimension_divergence_files[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Quality" in y_label:
                if ("4Quality" + GS_flag) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_quality_files[index].copy()
                                temp.append(each_file)
                                dimension_quality_files[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
                                temp = dimension_quality_files[index].copy()
                                temp.append(each_file)
                                dimension_quality_files[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_quality_files[index].copy()
                                temp.append(each_file)
                                dimension_quality_files[index] = temp
                        else:
                            raise ValueError("Unsupported")
            elif "Utilization" in y_label:
                if ("Utilization" + GS_flag) in each_file:
                    for index, each_reference in enumerate(file_name_reference):
                        if dimension == "Quality":
                            if "_Q" + str(each_reference) + '_' in each_file:
                                temp = dimension_utilization_files[index].copy()
                                temp.append(each_file)
                                dimension_utilization_files[index] = temp
                        elif dimension == "Openness":
                            if "_O" + str(each_reference) + "_" in each_file:
                                temp = dimension_utilization_files[index].copy()
                                temp.append(each_file)
                                dimension_utilization_files[index] = temp
                        elif dimension == "Proportion":
                            if "_Prop" + str(each_reference) + '_' in each_file:
                                temp = dimension_utilization_files[index].copy()
                                temp.append(each_file)
                                dimension_utilization_files[index] = temp
                        else:
                            raise ValueError("Unsupported")
                else:pass

        if "Divergence" in y_label:
            dimension_divergence_files = np.array(dimension_divergence_files, dtype=object)
            dimension_divergence_files = dimension_divergence_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))
        elif "Quality" in y_label:
            dimension_quality_files = np.array(dimension_quality_files, dtype=object)
            dimension_quality_files = dimension_quality_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))
        elif "Utilization" in y_label:
            dimension_utilization_files = np.array(dimension_utilization_files, dtype=object)
            dimension_utilization_files = dimension_utilization_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))

        if "Divergence" in y_label:
            divergence_curves_data = []
            for each_curve_files in dimension_divergence_files:
                # print(each_curve_files)
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                divergence_curves_data.append(data_curve)
            divergence_curves_data = np.array(divergence_curves_data, dtype=object)
            print("Divergence Shape (before): ", divergence_curves_data.shape)
        elif "Quality" in y_label:
            quality_curves_data = []
            for each_curve_files in dimension_quality_files:
                # print(each_curve_files)
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                quality_curves_data.append(data_curve)
            quality_curves_data = np.array(quality_curves_data, dtype=object)
            print("Quality Shape (before): ", quality_curves_data.shape)
        elif "Utilization" in y_label:
            utilization_curves_data = []
            for each_curve_files in dimension_utilization_files:
                # print(each_curve_files)
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                utilization_curves_data.append(data_curve)
            utilization_curves_data = np.array(utilization_curves_data, dtype=object)
            print("Utilization Shape (before): ", utilization_curves_data.shape)

        if "Divergence" in y_label:
            for d in range(divergence_curves_data.shape[0]):
                for k in range(divergence_curves_data.shape[1]):
                    for f in range(divergence_curves_data.shape[2]):
                        for l in range(divergence_curves_data.shape[3]):
                            pools_temp = list(divergence_curves_data[d][k][f][l])
                            divergence_temp = []
                            for solution_pool in pools_temp:
                                mode_solution = self.get_mode_solution(pool=solution_pool)
                                divegence_pool = sum(self.get_solution_distance(mode_solution, solution) for solution in solution_pool)
                                divergence_temp.append(divegence_pool/self.agent_num)
                            divergence_curves_data[d][k][f][l] = sum(divergence_temp)/len(divergence_temp)  # 100 values
        elif "Quality" in y_label:
            for d in range(quality_curves_data.shape[0]):
                for k in range(quality_curves_data.shape[1]):
                    for f in range(quality_curves_data.shape[2]):
                        for l in range(quality_curves_data.shape[3]):
                            pools_temp = list(quality_curves_data[d][k][f][l])
                            quality_temp = []
                            for quality_pool in pools_temp:
                                overall_quality = sum(quality_pool)/len(quality_pool)
                                quality_temp.append(overall_quality)
                            quality_curves_data[d][k][f][l] = sum(quality_temp)/len(quality_temp)  # 100 values
        elif "Utilization" in y_label:
            for d in range(utilization_curves_data.shape[0]):
                for k in range(utilization_curves_data.shape[1]):
                    for f in range(utilization_curves_data.shape[2]):
                        for l in range(utilization_curves_data.shape[3]):
                            pools_temp = list(utilization_curves_data[d][k][f][l])
                            utilization_temp = []
                            for utilization_pool in pools_temp:
                                overall_utilization = sum(utilization_pool)/len(utilization_pool)
                                utilization_temp.append(overall_utilization)
                            utilization_curves_data[d][k][f][l] = sum(utilization_temp)/len(utilization_temp)  # 100 values
        if "Divergence" in y_label:
            divergence_curves_data = divergence_curves_data.reshape((len(file_name_reference), len(self.K_list), -1))# K, dim, search,
            all_curves_data = divergence_curves_data
            print("Divergence Shape (after): ", divergence_curves_data.shape)
        elif "Quality" in y_label:
            quality_curves_data = quality_curves_data.reshape((len(file_name_reference), len(self.K_list), -1))# dim, K, search
            all_curves_data = quality_curves_data
            print("Quality Shape (after): ", quality_curves_data.shape)
        elif "Utilization" in y_label:
            utilization_curves_data = utilization_curves_data.reshape((len(file_name_reference), len(self.K_list), -1)) # dim, K, search
            all_curves_data = utilization_curves_data
            print("Utilization Shape (after): ", utilization_curves_data.shape)
        else:
            all_curves_data = None
        # with open("cache" + y_label, 'wb') as out_file:
        #     pickle.dump(all_curves_data, out_file)
        label_list = file_name_reference
        f = plt.figure()
        all_curves_data = np.squeeze(all_curves_data)
        ax = f.add_subplot()
        for lable, each_curve in zip(label_list, all_curves_data):  # dimention
            average_value = np.mean(np.array(each_curve), axis=-1)  # dim, search,
            if dimension == "Proportion":
                ax.plot(self.K_list, average_value, label="{0} of G :{1}".format(dimension, lable))
            else:
                ax.plot(self.K_list, average_value, label="{0}:{1}".format(dimension, lable))
        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        my_x_ticks = np.arange(min(self.K_list), max(self.K_list)+1, self.K_list[1]-self.K_list[0])
        plt.xticks(my_x_ticks)
        # ax.set_title(self.title)  # Add a title to the whole figure

        plt.legend()

        output = self.output_path + "\\" + self.title + "-" + dimension
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + dimension + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def generate_surface_direction(self, title=None, y_label=None, GS_flag=None):
        if title:
            self.title = title
        y_label_list = ["Divergence", "Quality", "Utilization"]
        file_name_reference = ["_SS1.0_GG1.0_", "_SS0_GG0_", "_SS0.5_GG0.5_"]  # 正向，反向，baseline
        temp_level_1 = []
        self.divergence_curves_data_mode = np.zeros((len(self.K_list), len(file_name_reference)))
        self.divergence_curves_data_haoming = np.zeros((len(self.K_list), len(file_name_reference)))
        self.quality_curves_data = np.zeros((len(self.K_list), len(file_name_reference)))
        self.utilization_curves_data = np.zeros((len(self.K_list), len(file_name_reference)))
        for gs, dimension in enumerate(file_name_reference):
            for k, k_name in enumerate(["_K" + str(each_k) + "_" for each_k in self.K_list]):
                temp = []
                for file in self.files_list:
                    if (dimension in file) and ((y_label + GS_flag) in file) and (k_name) in file:
                        temp.append(file)
                # print(temp)
                if len(temp) == 0:
                    # print(x_value, y_value, column_name, k_name)
                    continue
                data = self.load_data_from_files(files_list=temp)
                data = np.array(data, dtype=object)
                # data = data.reshape(1, -1)
                data = np.squeeze(data)
                self.data_temp = data
                print(np.array(data).shape)  # divergence:  (1000, 100, 100, 9): land, agent, search, N

                if "Divergence" in y_label:
                    for l in range(self.landscape_iteration):
                        pools_temp = list(data[l])
                        divergence_temp = []
                        for solution_pool in pools_temp:
                            mode_solution = self.get_mode_solution(pool=solution_pool)
                            divegence_pool = sum(
                                self.get_solution_distance(mode_solution, solution) for solution in solution_pool)
                            divergence_temp.append(divegence_pool / self.agent_num)
                        self.divergence_curves_data_mode[k][gs] = sum(divergence_temp)/len(divergence_temp)  # 100 values

                    # divergence_landscape = []
                    # for l in range(self.landscape_iteration):
                    #     pools_temp = list(data[l])
                    #     divergence_pool = []
                    #     for solution_pool in pools_temp:
                    #         divergence_each_solution = []
                    #         for each_solution in solution_pool:
                    #             divergence_each_solution += [self.get_solution_distance(each_solution, solution) for solution in solution_pool]
                    #         divergence_pool.append(sum(divergence_each_solution)/len(divergence_each_solution))  # [d1, d2, ... dm], m个solution
                    #     divergence_landscape.append(sum(divergence_pool)/len(divergence_pool)) # [D1, D2, ..., D100]
                    # self.divergence_curves_data_haoming[k][gs] = sum(divergence_landscape)/len(divergence_landscape)  # 100 values

                elif "Quality" in y_label:
                    quality_landscape = []
                    for l in range(self.landscape_iteration):
                        pools_temp = list(data[l])  # (1000, 100, 100): land, agent, search
                        overall_quality = 0
                        for quality_pool in pools_temp:
                            overall_quality = sum(quality_pool) / len(quality_pool)
                        quality_landscape.append(overall_quality)
                    self.quality_curves_data[k][gs] = sum(quality_landscape)/self.landscape_iteration
                elif "Utilization" in y_label:
                    utilization_landscape = []
                    for l in range(self.landscape_iteration):
                        pools_temp = list(data[l])
                        overall_utilization = 0
                        for utilization_pool in pools_temp:
                            overall_utilization = sum(utilization_pool) / len(utilization_pool)
                        utilization_landscape.append(overall_utilization)
                    self.utilization_curves_data[k][gs] = sum(utilization_landscape)/len(utilization_landscape)  # 100 values
        # print(self.quality_curves_data)
        if "Divergence" in y_label:
            self.divergence_curves_data_mode = self.divergence_curves_data_mode.reshape(len(file_name_reference),(len(self.K_list)))
            # self.divergence_curves_data_haoming = self.divergence_curves_data_haoming.reshape(len(file_name_reference),(len(self.K_list)))
            print("Divergence Shape (after): ", self.divergence_curves_data_mode.shape)
        elif "Quality" in y_label:
            self.quality_curves_data = self.quality_curves_data.reshape(len(file_name_reference),(len(self.K_list)))
            print("Quality Shape (after): ", self.quality_curves_data.shape)
        elif "Utilization" in y_label:
            self.utilization_curves_data = self.utilization_curves_data.reshape(len(file_name_reference),(len(self.K_list)))
            print("Utilization Shape (after): ", self.utilization_curves_data.shape)
        else:
            all_curves_data = None
        label_list = ["Homo", "Hetero", "Base"]
        figure = plt.figure()
        ax = figure.add_subplot()

        if "Divergence" in y_label:
            average_value = self.divergence_curves_data_mode
        elif "Quality" in y_label:
            average_value = self.quality_curves_data
        elif "Utilization" in y_label:
            average_value = self.utilization_curves_data
        else:
            average_value = None
        average_value = np.squeeze(average_value)
        # print(average_value.shape)
        for label, curve in zip(label_list, average_value):
            ax.plot(self.K_list, curve, label=label)
        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        my_x_ticks = np.arange(min(self.K_list), max(self.K_list)+1, self.K_list[1]-self.K_list[0])
        plt.xticks(my_x_ticks)
        plt.legend()

        output = self.output_path + "\\" + "Direction_Mechanism" + "-" + y_label
        i = 1
        while os.path.exists(output + ".png"):
            i += 1
            print("File Exists")
            output = self.output_path + "\\" + self.title + "-" + "Direction_Mechanism" + "-" + str(i)
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()


if __name__ == '__main__':
    data_folder = r'C:\Python_Workplace\hpc-0422\Experiments_V3\Direction'
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
    G_exposed_to_G_list = [0, 0.25, 0.5, 0.75, 1.0]
    S_exposed_to_S_list = [0, 0.25, 0.5, 0.75, 1.0]
    gs_proportion_list = [0.5]
    ###############################################################
    exposure_type = "Self-interested"

    fore_title = "N" + str(N) + "-" + exposure_type
    evaluator = Evaluator(data_path=data_folder, output_path=output_folder, )
    evaluator.load_iv_configuration(exposure_type=exposure_type, N=N, state_num=state_num, K_list=K_list, frequency_list=frequency_list,
                                    openness_list=openness_list, quality_list=quality_list, G_exposed_to_G_list=G_exposed_to_G_list,
                                    S_exposed_to_S_list=S_exposed_to_S_list, gs_proportion_list=gs_proportion_list, knowledge_num=knowledge_num)
    evaluator.load_simulation_configuration(landscape_iteration=landscape_iteration, agent_num=agent_num, search_iteration=search_iteration)

    # Main effect for one dimension except for Direction
    # evaluator.generate_one_dimension_figure(title=fore_title, dimension="Openness", y_label="Average", show_variance=False, percentage=0.1, top_coverage=None)

    # Surface evolution with search iterations
    # evaluator.generate_surface_evolution(title=fore_title, dimension="Proportion", GS_flag="G", y_label="Utilization")

    # Main effect for Direction
    # evaluator.generate_solid_figure(title=fore_title, percentage=10)

    # Interaction effect for two dimension, which do not include Direction.
    # evaluator.generate_interaction_figure_1(title=fore_title, x_label="Composition", y_label="Quality")

    # Interaction effect for Direction (GG/SS) plus one more dimension
    # evaluator.generate_interaction_figure_2(title=fore_title, given_k=8, row_label="Composition", top_coverage=100)

    # Surface evolution with search iterations using Direction
    evaluator.generate_surface_direction(title=fore_title, y_label="Divergence", GS_flag="G")
    print("END")

