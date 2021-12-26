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

    def individual_evaluate(self, label=None):
        """
        Only for the individual search
        :param label: the curve label, mainly for the agent name (e.g., T shape 22, T shape 41)
        :return: the figure
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
        print(data)

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

    def team_evaluate_2by2_matrix(self, label=None, figure_x=None, figure_y=None):
        """
        For team search, but also for one dimension figure
        either fixed team type + dynamic K or fixed K + dynamic team types
        :param label: the dimension that changes (e.g., either K values or team types)
        :return: the figure in the same file folder as data
        """
        # here data_path could be a list-> because we have two dimensions (team type + K)
        # or three dimensions (team type + K + N)
        if not isinstance(self.data_path, list):
            raise ValueError("For a matrix figure, data path need to be a list. Otherwise, team_evaluate() is enough.")
        if figure_x:
            if figure_x != len(self.data_path):
                raise ValueError("{0} folders doesn't fit figure_x = {1}".format(len(self.data_path), figure_x))
        if figure_y:
            if figure_y != len(self.data_path[0]):
                raise ValueError("{0} folders doesn't fit figure_y = {1}".format(len(self.data_path[0]), figure_y))

        # x by 1 figure
        if figure_x and (not figure_y):
            data_matrix = []
            for each_data_folder in self.data_path:
                data_files_path = []
                for root, dirs, files in os.walk(each_data_folder):
                    for each_file in files:
                        data_files_path.append(root + '\\' + each_file)
                file2data = []
                for each_file in data_files_path:
                    if ".png" not in each_file:
                        with open(each_file,'rb') as infile:
                            print(each_file)
                            temp = pickle.load(infile)
                            file2data.append(temp)
                data_matrix.append(file2data)

        if figure_x and figure_y:  # Y first (i.e., N); X second (i.e., K) for a figure
            data_matrix = []
            for y_data_folder in self.data_path:
                data_y = []
                for x_data_folder in y_data_folder:
                    data_files_path = []
                    for root, dirs, files in os.walk(x_data_folder):
                        for each_file in files:
                            data_files_path.append(root + '\\' + each_file)
                    file2data = []
                    for each_file in data_files_path:
                        if ".png" not in each_file:
                            with open(each_file, 'rb') as infile:
                                print(each_file)
                                temp = pickle.load(infile)
                                file2data.append(temp)
                    data_matrix.append(file2data)


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
        output = output_path + "\\" + self.title + ".png"
        plt.savefig(output)  # save the figure before plt.show(). Otherwise, there is no information.
        plt.show()

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
    team_search_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\N10GS12"
    output_path = team_search_path
    evaluator2 =Evaluator(title="GS team", data_path=team_search_path, output_path=output_path)
    team_name = ["K2", "K4", "K6", "K8", "K10"]
    evaluator2.team_evaluate(label=team_name)



    team_search_path = r"C:\Python_Workplace\OpenInnovationFramework\Reproduction\N10GS12"
    output_path = team_search_path
    evaluator2 =Evaluator(title="GS team", data_path=team_search_path, output_path=output_path)
    team_name = ["K2", "K4", "K6", "K8", "K10"]
    evaluator2.team_evaluate(label=team_name)


