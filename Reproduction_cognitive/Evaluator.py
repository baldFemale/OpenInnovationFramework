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
        return  data_files

    def load_folders_under_parent_folder(self, parent_folder=None):
        """
        For multi-layers file loading
        :return:the folder list containing the data files
        """
        folder_list = []
        for root, dirs, files in os.walk(parent_folder):
            for ech_dir in dirs:
                folder_list.append(root + '\\' + ech_dir)
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
                with open(each_file,'rb') as infile:
                    print(each_file)
                    temp = pickle.load(infile)
                    data.append(temp)
        return data

    def individual_evaluate(self, label=None):
        """
        Only for the individual search
        :param label: the curve label, mainly for the agent name (e.g., T shape 22, T shape 41)
        :return: the figure
        """
        data_files = self.load_files_under_folder()
        data = self.load_data_from_files(files_path=data_files)
        # individual level comparison
        figure, axis = plt.subplots()
        for name, each_data in zip(label, data):
            # print(len(each_data), len(each_data[0]), len(each_data[0][0]))
            # landscape_iteraton=5;     agent_iteration = 200; search_iteration = 200
            axis.plot(np.mean(np.mean(np.array(each_data), axis=0), axis=0), label="{}".format(name))

        axis.set_xlabel('Search iteration')  # Add an x-label to the axes.
        axis.set_ylabel('Average fitness')  # Add a y-label to the axes.

        flag = 0
        for each in list(data_files[0]):
            if (each == "_") & (flag < 2):
                flag += 1
            if flag == 2:
                self.title += each

        axis.set_title(self.title)  # Add a title to the axes.
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def team_evaluate(self, label=None):
        """
        For team search, but also for one dimension figure
        either fixed team type + dynamic K or fixed K + dynamic team types
        :param label: the dimension that changes (e.g., either K values or team types)-> added into legend
        :return: the figure in the same file folder as data
        """
        data_files = []
        for root, dirs, files in os.walk(self.data_path):
            for ech_file in files:
                data_files.append(root + '\\' + ech_file)
        data = []
        for each_file in data_files:
            if ".png" not in each_file:
                with open(each_file,'rb') as infile:
                    print(each_file)
                    temp = pickle.load(infile)
                    data.append(temp)
        print("file number, landscape loop, agent loop. searh loop: ", np.array(data).shape)

        # individual level comparison
        figure, axis = plt.subplots()
        for name, each_data in zip(label, data):
            # print(len(each_data), len(each_data[0]), len(each_data[0][0]))
            # landscape_iteraton=5;     agent_iteration = 200; search_iteration = 200
            axis.plot(np.mean(np.mean(np.array(each_data), axis=0), axis=0), label="{}".format(name))

        axis.set_xlabel('Search iteration')  # Add an x-label to the axes.
        axis.set_ylabel('Average fitness')  # Add a y-label to the axes.
        axis.set_title(self.title)  # Add a title to the axes.
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

    def team_evaluate_2by2_matrix(self, label=None, figure_x=None, figure_y=None, re_x_rule_list=None):
        """
        For team search, but also for one dimension figure
        either fixed team type + dynamic K or fixed K + dynamic team types
        :param label: the dimension that changes (e.g., either K values or team types)
        :return: the figure in the same file folder as data
        """
        # here data_path could be a list-> because we have two dimensions (team type + K)
        # or three dimensions (team type + K + N)
        # if figure_x:
        #     if figure_x != len(self.data_path[0]):
        #         raise ValueError("{0} folders doesn't fit figure_x = {1}".format(len(self.data_path), figure_x))
        # if figure_y:
        #     if figure_y != len(self.data_path):
        #         raise ValueError("{0} folders doesn't fit figure_y = {1}".format(len(self.data_path[0]), figure_y))

        y_parent_folder = []
        subplot_title = []
        for each_x_rule in re_x_rule_list:
            x_parent_folder = []
            for root, dirs, files in os.walk(self.data_path):
                for each_dir in dirs:
                    if re.search(pattern=each_x_rule, string=each_dir):
                        x_parent_folder.append(root +"\\"+ each_dir)
                        subplot_title.append(re.search(pattern=each_x_rule, string=each_dir).group(1))
            y_parent_folder.append(x_parent_folder)

        # load the data
        data = []
        for y_index in range(figure_y):
            y_folder_list = y_parent_folder[y_index]
            data_x = []
            for x_index in range(figure_x):
                x_folder_list = y_folder_list[x_index]
                print("**************")
                for root, dirs, files in os.walk(x_folder_list):
                    for each_file in files:
                        if ".png" not in each_file:
                            print("Check the order of team type (the label !!)", each_file)
                            with open(root+"\\" + each_file, 'rb') as infile:
                                temp = pickle.load(infile)
                                data_x.append(temp)
            data.append(data_x)



        print("len(label)", len(label))
        # individual level comparison
        figure = plt.figure()
        for y, each_row_data in zip(range(figure_y), data): # within-figure: different team types
            for x, each_column_data in zip(range(figure_x), each_row_data):
                ax = figure.add_subplot(y+1, figure_x, x+1)
                ax.title.set_text("{0}".format(subplot_title[x]))
                for z, each_curve in zip(range(len(label)), each_column_data):
                    ax.plot(np.mean(np.mean(np.array(each_column_data), axis=0), axis=0), label="{}".format(label[z]))
        ax.set_xlabel('Search iteration')  # Add an x-label to the axes.
        ax.set_ylabel('Average fitness')  # Add a y-label to the axes.
        ax.set_title(self.title)  # Add a title to the whole figure
        plt.legend()
        output = self.output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

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

    def final_fitness_order(self):
        """
        The final converge is the performance of each agent-task combination
        :return: the converge figure
        """
        pass

    def performance_polyline_across_type_k(self):
        """
        Line graph across different agent or team type and landscape ruggedness
        :return: polyline figure
        """
        agent_level_data = []
        ruggedness_level_data = []
        agent_folders = self.load_folders_under_parent_folder(parent_folder=self.data_path)
        for agent_level_folder in agent_folders:
            ruggedness_level_folder = self.load_folders_under_parent_folder(parent_folder=agent_level_folder)
            for each_folder in ruggedness_level_folder:
                data_files = self.load_files_under_folder()
                data =


if __name__ == '__main__':
    # Test Example

    # Individual search
    # data_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\output10"
    # output_path = data_path
    # evaluator = Evaluator(title=None, data_path=data_path, output_path=output_path)
    # agent_name = ["Generalist", "Specialist", "T shape 22", "T shape 41"]     # for the curve name
    # evaluator.individual_evaluate(label=agent_name)

    #-----------------------------------------------------------------------------------------------------#

    # team search
    # fixed team type: GS
    # dynamic K value from 2 to 10
    # input folder: only *one* folder which contains several bin data
    # team_search_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\N10GS12"
    # output_path = team_search_path
    # evaluator2 = Evaluator(title="GS team (N=10)", data_path=team_search_path, output_path=output_path)
    # team_name = ["K2", "K4", "K6", "K8", "K10"]
    # evaluator2.team_evaluate(label=team_name)

    team_search_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\N_10_K_10_HeteroTeamSerial"
    output_path = team_search_path
    evaluator2 =Evaluator(title="HeteroTeamSerial_N10K10", data_path=team_search_path, output_path=output_path)
    team_name = ['GS', "GT22", "GT41", "SG", "ST22", "ST41", "T22G", "T22S", "T41G", "T41S"]
    evaluator2.team_evaluate(label=team_name)

    # team_search_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\N_10_K_08_HeteroTeamSerial"
    # output_path = team_search_path
    # evaluator2 =Evaluator(title="HeteroTeamSerial_N10K08", data_path=team_search_path, output_path=output_path)
    # team_name = ['GS', "GT22", "GT41", "SG", "ST22", "ST41", "T22G", "T22S", "T41G", "T41S"]
    # evaluator2.team_evaluate(label=team_name)


    # fixed N, dynamic K, team type as the label;
    # figure: 1 by K sub-figures
    # input folder: the folder list/parent folder which contain (N_type * K_type) folders; each folde contain several team types data
    # parent_folder = r'C:\Python_Workplace\OpenInnovationFramework\Reproduction'
    # figure_x = 6
    # figure_y = 1
    # re_x_rule_list = [r'N_10_(.*?)_HeteroTeamSerial']
    # evaluator2 =Evaluator(title="Hetero Team Serial", data_path=parent_folder, output_path=parent_folder)
    # team_name = ['GS', "GT22", "GT41", "SG", "ST22", "ST41", "T22G", "T22S", "T41G", "T41S"]
    # evaluator2.team_evaluate_2by2_matrix(label=team_name, figure_y=figure_y,
    #                                      figure_x=figure_x, re_x_rule_list=re_x_rule_list)

    #------------------------------------------------------------------------------------------------#
    # line graph across different agents and landscape ruggedness

