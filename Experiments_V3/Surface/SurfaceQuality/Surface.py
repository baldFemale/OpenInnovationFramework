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


class Surface:
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

    def generate_surface_outcome(self, title=None, dimension=None, GS_flag='', y_label=None):
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

        self.dimension_divergence_files = dimension_divergence_files
        self.dimension_quality_files = dimension_quality_files
        self.dimension_utilization_files = dimension_utilization_files

        if "Divergence" in y_label:
            self.dimension_divergence_files = np.array(self.dimension_divergence_files, dtype=object)
            self.dimension_divergence_files = self.dimension_divergence_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))
        elif "Quality" in y_label:
            self.dimension_quality_files = np.array(self.dimension_quality_files, dtype=object)
            self.dimension_quality_files = self.dimension_quality_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))
        elif "Utilization" in y_label:
            self.dimension_utilization_files = np.array(self.dimension_utilization_files, dtype=object)
            self.dimension_utilization_files = self.dimension_utilization_files.reshape(
                (len(file_name_reference), len(self.K_list), -1))

        if "Divergence" in y_label:
            self.divergence_curves_data = []
            for each_curve_files in self.dimension_divergence_files:
                # print(each_curve_files)
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                self.divergence_curves_data.append(data_curve)
            self.divergence_curves_data = np.array(self.divergence_curves_data, dtype=object)
            print("Divergence Shape (before): ", self.divergence_curves_data.shape)
        elif "Quality" in y_label:
            self.quality_curves_data = []
            for each_curve_files in self.dimension_quality_files:
                # print(each_curve_files)
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                self.quality_curves_data.append(data_curve)
            self.quality_curves_data = np.array(self.quality_curves_data, dtype=object)
            print("Quality Shape (before): ", self.quality_curves_data.shape)
        elif "Utilization" in y_label:
            self.utilization_curves_data = []
            for each_curve_files in self.dimension_utilization_files:
                # print(each_curve_files) # dim, K, file
                data_curve = self.load_data_from_folders(folder_list=each_curve_files)
                # print(len(data_curve))
                self.utilization_curves_data.append(data_curve)
            self.utilization_curves_data = np.array(self.utilization_curves_data, dtype=object)
            print("Utilization Shape (before): ", self.utilization_curves_data.shape)

        if "Divergence" in y_label:
            shape = self.divergence_curves_data.shape
            self.divergence_curves_data_mode = np.ones((len(file_name_reference), len(self.K_list), 1, 1000, 100))
            for d in range(self.divergence_curves_data.shape[0]):
                for k in range(self.divergence_curves_data.shape[1]):
                    for f in range(self.divergence_curves_data.shape[2]):
                        for l in range(self.divergence_curves_data.shape[3]):
                            pools_temp = list(self.divergence_curves_data[d][k][f][l])
                            divergence_temp = []
                            for solution_pool in pools_temp:
                                mode_solution = self.get_mode_solution(pool=solution_pool)
                                divegence_pool = sum(self.get_solution_distance(mode_solution, solution) for solution in solution_pool)
                                divergence_temp.append(divegence_pool/self.agent_num)
                            self.divergence_curves_data_mode[d][k][f][l] = divergence_temp  # 100 values
            self.divergence_curves_data_haoming = np.ones((len(file_name_reference), len(self.K_list), 1, 1000, 100))
            for d in range(self.divergence_curves_data.shape[0]):
                for k in range(self.divergence_curves_data.shape[1]):
                    for f in range(self.divergence_curves_data.shape[2]):
                        for l in range(self.divergence_curves_data.shape[3]):
                            pools_temp = list(self.divergence_curves_data[d][k][f][l])
                            divergence_temp = []
                            for solution_pool in pools_temp:
                                distance_pool = []
                                for each_solution in solution_pool:
                                    distance_temp = sum([self.get_solution_distance(each_solution, solution) for solution in solution_pool])
                                    distance_pool.append(distance_temp)  # [d1, d2, ... dm]
                                divergence_temp.append(sum(distance_pool)/len(distance_pool)) # [D1, D2, ..., D100]
                            self.divergence_curves_data_haoming[d][k][f][l] = divergence_temp  # 100 values
        elif "Quality" in y_label:
            for d in range(self.quality_curves_data.shape[0]):
                for k in range(self.quality_curves_data.shape[1]):
                    for f in range(self.quality_curves_data.shape[2]):
                        for l in range(self.quality_curves_data.shape[3]):
                            pools_temp = list(self.quality_curves_data[d][k][f][l])
                            quality_temp = []
                            for quality_pool in pools_temp:
                                overall_quality = sum(quality_pool)/len(quality_pool)
                                quality_temp.append(overall_quality)
                            self.quality_curves_data[d][k][f][l] = quality_temp  # 100 values
        elif "Utilization" in y_label:
            for d in range(self.utilization_curves_data.shape[0]):
                for k in range(self.utilization_curves_data.shape[1]):
                    for f in range(self.utilization_curves_data.shape[2]):
                        for l in range(self.utilization_curves_data.shape[3]):
                            pools_temp = list(self.utilization_curves_data[d][k][f][l])
                            utilization_temp = []
                            for utilization_pool in pools_temp:
                                overall_utilization = sum(utilization_pool)/len(utilization_pool)
                                utilization_temp.append(overall_utilization)
                            self.utilization_curves_data[d][k][f][l] = utilization_temp  # 100 values
        if "Divergence" in y_label:
            self.divergence_curves_data = self.divergence_curves_data.reshape((len(self.K_list),
                                                                    len(file_name_reference), self.search_iteration, -1)) # K, dim, search,
            print("Divergence Shape (after): ", self.divergence_curves_data.shape)
        elif "Quality" in y_label:
            self.quality_curves_data = self.quality_curves_data.reshape((len(self.K_list),
                                                                    len(file_name_reference), self.search_iteration, -1)) # dim, K, search
            print("Quality Shape (after): ", self.quality_curves_data.shape)
        elif "Utilization" in y_label:
            self.utilization_curves_data = self.utilization_curves_data.reshape((len(self.K_list),
                                                                    len(file_name_reference), self.search_iteration, -1)) # dim, K, search
            print("Utilization Shape (after): ", self.utilization_curves_data.shape)
        else:
            all_curves_data = None
        if "Divergence" in y_label:
            with open(self.output_path + r"\mode_" + dimension + "_" + GS_flag + "_" + y_label, 'wb') as out_file:
                pickle.dump(self.divergence_curves_data_mode, out_file)
            with open(self.output_path + r"\haoming_" + dimension + "_" + GS_flag + "_" + y_label, 'wb') as out_file:
                pickle.dump(self.divergence_curves_data_haoming, out_file)
        elif "Quality" in y_label:
            with open(self.output_path + r"\cache_" + dimension + "_" + GS_flag + "_" + y_label, 'wb') as out_file:
                pickle.dump(self.quality_curves_data, out_file)
        elif "Utilization" in y_label:
            with open(self.output_path + r"\cache_" + dimension + "_" + GS_flag + "_" + y_label, 'wb') as out_file:
                pickle.dump(self.utilization_curves_data, out_file)


if __name__ == '__main__':
    data_foler = r'e0546117@atlas8.nus.edu.sg:/hpctmp/e0546117/Experiments_V3/Quality'
    output_folder = r'e0546117@atlas8.nus.edu.sg:/hpctmp/e0546117/Experiments_V3\result'
    # ###############################################################
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
    quality_list = [0, 0.25, 0.5, 0.75, 1.0]
    G_exposed_to_G_list = [0.5]
    S_exposed_to_S_list = [0.5]
    gs_proportion_list = [0.5]
    ###############################################################
    exposure_type = "Self-interested"

    fore_title = "N" + str(N) + "-" + exposure_type
    surface = Surface(data_path=data_foler, output_path=output_folder, )
    surface.load_iv_configuration(exposure_type=exposure_type, N=N, state_num=state_num, K_list=K_list, frequency_list=frequency_list,
                                    openness_list=openness_list, quality_list=quality_list, G_exposed_to_G_list=G_exposed_to_G_list,
                                    S_exposed_to_S_list=S_exposed_to_S_list, gs_proportion_list=gs_proportion_list, knowledge_num=knowledge_num)
    surface.load_simulation_configuration(landscape_iteration=landscape_iteration, agent_num=agent_num, search_iteration=search_iteration)
#################################################################
    # Openness, divergence
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="G", y_label="Divergence")
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="S", y_label="Divergence")

    # Composition, Quality
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="G", y_label="Quality")
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="S", y_label="Quality")

    # Openness, utilization
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="G", y_label="Utilization")
    surface.generate_surface_outcome(title=fore_title, dimension="Quality", GS_flag="S", y_label="Utilization")
    ##################################################################
    print("END")
