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

    def load_data_from_files(self, files_path=None):
        """
        load one data from one certain file path
        :param file_path: data file path
        :return:the loaded data (i.e., fitness list)
        """
        data = []
        for each_file in files_path:
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


if __name__ == '__main__':
    # Test Example
    # Convergence evaluation
    # parent_folder = r"C:\Python_Workplace\hpc-0126\nk\Factor\convergence"
    # output_path = parent_folder
    # evaluator = Evaluator(title="GS Convergence Factor in IM N10", data_path=parent_folder, output_path=output_path)
    # team_name = ["G", "S", "T43", "T62"]
    # evaluator.convergence_evaluation(label_list=team_name)
    # evaluator.convergence_evaluation(label_list=team_name, select_list=[0,1], show_variance=True)


    # match evaluation
    # parent_folder = r"C:\Python_Workplace\hpc-0126\nk\Factor\column_match"
    # output_path = parent_folder
    # evaluator = Evaluator(title="Individual Match in Factor IM N10", data_path=parent_folder, output_path=output_path)
    # team_name = ["G", "S", "T43", "T62"]
    # # team_name = ["G", "S"]
    # # evaluator.match_evaluation(label_list=team_name)
    # evaluator.match_evaluation(label_list=team_name, select_list=[3], show_variance=True)

    # expert test
    #
    parent_folder = r"C:\Python_Workplace\OpenInnovationFramework\Socialized_innovation_crowdsourcing\expert_results\expert_factor\convergency"
    output_path = parent_folder
    evaluator = Evaluator(title="Expert Convergence Influential N10", data_path=parent_folder, output_path=output_path)
    team_name = ["T27", "T46"]
    evaluator.convergence_evaluation(label_list=team_name)

