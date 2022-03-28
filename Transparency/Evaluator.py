# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Evaluator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re
import scipy


class Evaluator:
    def __init__(self, title=None, data_path=None, output_path=None):
        if not title:
            self.title = ''
        else:
            self.title = title
        if not data_path:
            data_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\output"
        if not output_path:
            output_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\output"
        self.data_path = data_path
        self.output_path = output_path

    def load_files_under_folder(self, folders=None):
        """
        For single-layers file loading
        :return:the data file list
        """
        data_files = []
        for root, dirs, files in os.walk(folders):
            for ech_file in files:
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
            if ".png" not in each_file:
                with open(each_file, 'rb') as infile:
                    # print(each_file)
                    temp = pickle.load(infile)
                    data.append(temp)
        return data

    def f_test(self, baseline=None, showline=None):
        """
        Compare everay two dataset to see whether there is a significant difference and how much
        Such a F-test is more PRECISE than the figure, especiallt when there are too many groups to compare
        We need a baseline; and then other groups are described as [+/- diff with sig.]
        :return: a F-test table with difference and significance level. e.g., 0.2 (***)
        """
        x = np.array(baseline)
        y = np.array(showline)
        f = np.var(x, ddof=1) / np.var(y, ddof=1)  # calculate F test statistic
        dfn = x.size - 1  # define degrees of freedom numerator
        dfd = y.size - 1  # define degrees of freedom denominator
        p = 1 - scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
        if p < 0.01:
            star = '***'
        elif p < 0.05:
            star = '**'
        elif p < 0.1:
            star = '*'
        else:
            star = '.'
        return f, p, star

    def simple_evaluation(self, label_list=None, show_variance=False, select_list=None):
        """
        1-D Analysis: either across K or acorss Agent type or IM type
        :param label_list: the labels for the curves
        :param show_variance: show the upper and lower bound
        :param select_list: only select part of the curve, via the list index
        :return: the figure in the same folder
        """
        data_files = self.load_files_under_folder(folders=self.data_path)
        data_folder = self.load_data_from_files(data_files)
        print(np.array(data_folder).shape)  # (4, 10, 10, 200): (folders/agents, K, landscape loop, agent loop)
        for each in np.array(data_folder):
            print(np.mean(np.mean(each, axis=1), axis=0))
        # only evaluate part of these data
        if (len(label_list) != len(data_folder)) or select_list:
            data_folder = [data_folder[i] for i in select_list]
            label_list = [label_list[i] for i in select_list]

        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, data_folder):  # release the Agent type level
            average_value = np.mean(np.array(each_curve), axis=1)  # 10 * (500 * 500)
            ax.plot(average_value, label="{}".format(lable))

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel('Average fitness')  # Add a y-label to the axes.
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def convergence_evaluation(self, label_list=None, show_variance=False, select_list=None):
        """
        2-D Analysis: across K and across Agent type
        :param label_list: the labels for the curves
        :param show_variance: show the upper and lower bound
        :param select_list: only select part of the curve, via the list index
        :return: the figure in the same folder
        """
        folders_list = self.load_folders_under_parent_folder(parent_folder=self.data_path)
        data_folders = []
        for folder in folders_list:
            data_files = self.load_files_under_folder(folder)
            print(data_files)
            data_list = self.load_data_from_files(data_files)
            data_folders.append(data_list)
        print(np.array(data_folders).shape)  # (4, 10, 10, 200): (folders/agents, K, landscape loop, agent loop)
        print("data:", data_folders[0][0][0])

        # only evaluate part of these data
        if (len(label_list) != len(data_folders)) or select_list:
            data_folders = [data_folders[i] for i in select_list]
            label_list = [label_list[i] for i in select_list]

        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, data_folders):  # release the Agent type level
            average_value = np.mean(np.mean(np.array(each_curve), axis=2), axis=1)  # 10 * (500 * 500)
            variation_value = np.var(np.array(each_curve).reshape((10, -1)), axis=1)
            ax.plot(average_value, label="{}".format(lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K') # Add an x-label to the axes.
        ax.set_ylabel('Average fitness')  # Add a y-label to the axes.
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def match_evaluation(self, label_list=None, show_variance=False, select_list=None):
        folders_list = self.load_folders_under_parent_folder(parent_folder=self.data_path)

        data_folders = []
        for folder in folders_list:
            data_files = self.load_files_under_folder(folder)
            data_list = self.load_data_from_files(data_files)
            data_folders.append(data_list)
        print(np.array(data_folders).shape)  # (4, 10, 10, 200): (folders/agents, K, landscape loop, agent loop)
        print("data:", data_folders[0][4][0])

        # only evaluate part of these data
        if (len(label_list) != len(data_folders)) or select_list:
            data_folders = [data_folders[i] for i in select_list]
            label_list = [label_list[i] for i in select_list]

        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, data_folders):  # release the Agent type level
            average_value = np.mean(np.mean(np.array(each_curve), axis=2), axis=1)  # 10 * (500 * 500)  K * Landscape * Agent
            variation_value = np.var(np.array(each_curve).reshape((10, -1)), axis=1)
            print("average_value", average_value)
            print("variation_value",variation_value)
            ax.plot(average_value, label="{}".format(lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel('Match')  # Add a y-label to the axes.
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def transparency_evaluation(self, label_list=None, show_variance=False, y_label=None):
        # sort the files into folders
        files_list = self.load_files_under_folder(folders=self.data_path)
        # print(files_list)
        print(len(files_list))
        potential_files = []
        convergence_files = []
        unique_files = []
        for each in files_list:
            if "png" in each:
                continue
            if "Potential" in each:
                potential_files.append(each)
            elif "Convergence" in each:
                convergence_files.append(each)
            elif "Unique" in each:
                unique_files.append(each)
            else:
                pass
                # the files left are the .py or .pyc files
                # print(each)
        transparency_A_file_list = []
        transparency_G_file_list = []
        transparency_S_file_list = []
        transparency_Inverse_file_list = []
        for each in files_list:
            if "_A_" in each:
                transparency_A_file_list.append(each)
            elif "_G_" in each:
                transparency_G_file_list.append(each)
            elif "_S_" in each:
                transparency_S_file_list.append(each)
            elif "_Inverse_" in each:
                transparency_Inverse_file_list.append(each)
            else:pass
                # print(each)
        frequency_1_file_list = []
        frequency_5_file_list = []
        frequency_10_file_list = []
        frequency_20_file_list = []
        frequency_50_file_list = []
        for each in files_list:
            if "_F1_" in each:
                frequency_1_file_list.append(each)
            elif "_F5_" in each:
                frequency_5_file_list.append(each)
            elif "_F10_" in each:
                frequency_10_file_list.append(each)
            elif "_F20_" in each:
                frequency_20_file_list.append(each)
            elif "_F50_" in each:
                frequency_50_file_list.append(each)
            else:pass
                # print(each)
        # Potentional-K, with social frequency as the interaction
        curve_1_data_files = []
        curve_2_data_files = []
        curve_3_data_files = []
        curve_4_data_files = []
        curve_5_data_files = []
        all_curves_data = []
        # This is for 5 types of frequency setting as the label list
        selected_y_files = []
        if "Potential" in y_label:
            selected_y_files = potential_files
        elif "Unique" in y_label:
            selected_y_files = unique_files
        elif "Average" in y_label:
            selected_y_files = convergence_files

        # need to clarify the naming of transparency direction parameter
        # selected_direction_files = []
        # if "S" in self.title:
        #     selected_direction_files = transparency_S_file_list
        # elif "G" in y_label:
        #     selected_direction_files = unique_files
        # elif "Average" in y_label:
        #     selected_direction_files = convergence_files

        # The first kind of figure: Fitness-K across different frequency
        if "1" in self.title:
            for each in selected_y_files:
                if each in transparency_S_file_list:
                    if each in frequency_1_file_list:
                        curve_1_data_files.append(each)
                    elif each in frequency_5_file_list:
                        curve_2_data_files.append(each)
                    elif each in frequency_10_file_list:
                        curve_3_data_files.append(each)
                    elif each in frequency_20_file_list:
                        curve_4_data_files.append(each)
                    elif each in frequency_50_file_list:
                        curve_5_data_files.append(each)
            all_curves_data_files = [curve_1_data_files, curve_2_data_files, curve_3_data_files, curve_4_data_files, curve_5_data_files]
            for each in all_curves_data_files:
                curve_data = self.load_data_from_files(each)
                all_curves_data.append(curve_data)
            print(np.array(all_curves_data).shape)

        # compare different exposure directions and their diff.
        # This is for the three types of exposure directions as the label list
        # Second kind of figure: Fitness-K across directions
        elif "2" in self.title:
            for each in selected_y_files:
                if each in frequency_50_file_list:
                    if each in transparency_S_file_list:
                        curve_1_data_files.append(each)
                    elif each in transparency_G_file_list:
                        curve_2_data_files.append(each)
                    elif each in transparency_A_file_list:
                        curve_3_data_files.append(each)
                    elif each in transparency_Inverse_file_list:
                        curve_4_data_files.append(each)
            all_curves_data_files = [curve_1_data_files, curve_2_data_files, curve_3_data_files, curve_4_data_files]
            for each in all_curves_data_files:
                curve_data = self.load_data_from_files(each)
                all_curves_data.append(curve_data)
            print(np.array(all_curves_data).shape)

        # Third kind of figure: Fitness-K across information quality
        #################
        # waiting for codes #
        #################

        if len(label_list) != len(all_curves_data):
            raise ValueError("Mismatch between curve number {0} and label number {1}".format(len(label_list), len(all_curves_data)))

        figure = plt.figure()
        ax = figure.add_subplot()
        for lable, each_curve in zip(label_list, all_curves_data):  # release the Agent type level
            average_value, variation_value = None, None
            if "Unique" in y_label:
            # unique fitness lack one dimension
                average_value = np.mean(np.array(each_curve), axis=1)  # 10 * (500 * 500)
                variation_value = np.var(np.array(each_curve).reshape((100, -1)), axis=1)
            elif ("Average" in y_label) or ("Potential" in y_label):
                average_value = np.mean(np.mean(np.array(each_curve), axis=2), axis=1)  # 10 * (500 * 500)
                variation_value = np.var(np.array(each_curve).reshape((100, -1)), axis=1)
            ax.plot(average_value, label="{}".format(lable))
            if show_variance:
                # draw the variance
                lower = [x - y for x, y in zip(average_value, variation_value)]
                upper = [x + y for x, y in zip(average_value, variation_value)]
                xaxis = list(range(len(lower)))
                ax.fill_between(x=xaxis, y1=lower, y2=upper, alpha=0.15)

        ax.set_xlabel('K')  # Add an x-label to the axes.
        ax.set_ylabel(str(y_label))  # Add a y-label to the axes.
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

        # data_potential_social_frequency_files_1 = self.load_data_from_files()
        # for each in :
        #     print(each)
        # print(np.array(data_potential_social_frequency_files_1).shape)


if __name__ == '__main__':
    # Test Example
    # Test different types of socialization frequency-> The frequency doesn't matter
    # The reason is that the current version, agent only select the same state to copy
    # title = '1-Unique-S'
    # data_path = r'C:\Python_Workplace\hpc-0328\transparency_random'
    # label_list = ["1", "5", "10", "20", "50"]
    # y_label = "Unique Fitness"
    # evaluator = Evaluator(title=title, data_path=data_path, output_path=data_path)
    # evaluator.transparency_evaluation(label_list=label_list, y_label=y_label)

    # Test different kind ->
    title = '2-Different Transparency Directions measured by Average-50'
    data_path = r'C:\Python_Workplace\hpc-0328\transparency_random'
    label_list = ["S", "G", "A", "Inverse"]
    y_label = "Average Fitness"
    evaluator = Evaluator(title=title, data_path=data_path, output_path=data_path)
    evaluator.transparency_evaluation(label_list=label_list, y_label=y_label)

