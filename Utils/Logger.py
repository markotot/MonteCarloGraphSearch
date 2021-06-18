import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from datetime import datetime
from math import ceil, floor, sqrt


class Logger:

    file = "test_data.txt"
    file_graph = "test_graph.txt"
    file_reroute = "test_reroute.txt"
    file_novel = "test_novel.txt"

    logs_folder = "Results/Logs/"
    metrics_folder = "Results/Experiments/"
    directory_path = None

    @staticmethod
    def setup(env_info, seed, path):

        if not os.path.isdir(Logger.logs_folder.split('/')[0]):
            os.mkdir(Logger.logs_folder.split('/')[0])

        if not os.path.isdir(Logger.logs_folder):
            os.mkdir(Logger.logs_folder)

        if not os.path.isdir(Logger.logs_folder + env_info):
            os.mkdir(Logger.logs_folder + env_info)

        current_time = datetime.now().strftime("%d_%h_%y_%H_%M_%S")
        Logger.directory_path = Logger.logs_folder + env_info + "/" + current_time + "/"
        if not os.path.isdir(Logger.directory_path):
            os.mkdir(Logger.directory_path)

        Logger.file = open(Logger.directory_path + path + "_data.txt", mode='w', buffering=1)
        Logger.file_graph = open(Logger.directory_path + path + "_graph.txt", mode='w', buffering=1)
        Logger.file_reroute = open(Logger.directory_path + path + "_reroute.txt", mode='w', buffering=1)
        Logger.file_novel = open(Logger.directory_path + path + "_novel.txt", mode='w', buffering=1)


        dt_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        Logger.file.write(f"Date: {dt_now} \n\n")
        Logger.file_graph.write(f"Date: {dt_now} \n\n")
        Logger.file_reroute.write(f"Date: {dt_now} \n\n")
        Logger.file_novel.write(f"Date: {dt_now} \n\n")

    @staticmethod
    def time_now():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def log_data(message, time=True):
        if time:
            Logger.file.write(f"{Logger.time_now()} - {message} \n")
        else:
            Logger.file.write(f"{message} \n")

    @staticmethod
    def log_graph_data(message, time=True):
        if time:
            Logger.file_graph.write(f"{Logger.time_now()} - {message} \n")
        else:
            Logger.file_graph.write(f"{message} \n")


    @staticmethod
    def log_reroute_data(message, time=True):
        if time:
            Logger.file_reroute.write(f"{Logger.time_now()} - {message} \n")
        else:
            Logger.file_reroute.write(f"{message} \n")

    @staticmethod
    def log_novel_data(message, time=True):
        if time:
            Logger.file_novel.write(f"{Logger.time_now()} - {message} \n")
        else:
            Logger.file_novel.write(f"{message} \n")

    @staticmethod
    def close():
        Logger.file.close()
        Logger.file_graph.close()
        Logger.file_reroute.close()
        Logger.file_novel.close()

    @staticmethod
    def save_experiment_metrics(env_name, experiments, experiment_metrics):

        current_time = datetime.now().strftime("%d_%h_%y_%H_%M_%S")
        directory_path = Logger.metrics_folder + env_name + "/" + current_time

        #creates Results/Experiments
        if not os.path.isdir(Logger.metrics_folder):
            os.mkdir(Logger.metrics_folder)

        #creates Results/Experiments/env_name
        if not os.path.isdir(Logger.metrics_folder + env_name):
            os.mkdir(Logger.metrics_folder + env_name)

        #creates full path
        if not os.path.isdir(directory_path):
            os.mkdir(f"{directory_path}")
            os.mkdir(f"{directory_path}/AgentConfigs")

        experiment_metrics.to_csv(f"{directory_path}/experiment_metrics.csv")

        for experiment in experiments:
            config_name = experiment.split('/')[-1]
            shutil.copy(experiment, f"{directory_path}/AgentConfigs/{config_name}")


def plot_images(number, images, reward, verbose):
    image_len = len(images)
    empty = np.array(images[0].copy())
    empty.fill(0)

    cols = sqrt(image_len)
    if floor(cols) < cols:
        cols = floor(cols) + 1
    else:
        cols = floor(cols)  # for some reason this is needed

    rows = ceil(len(images) / cols)

    images.extend(((cols * rows) - image_len) * [empty])

    image_rows = []
    for i in range(rows):
        image_row = np.concatenate(images[i * cols: (i + 1) * cols], 1)
        image_rows.append(image_row)

    plt.axis('off')
    plt.title(f"Test:{number} Steps: {image_len - 1}   Reward: {round(reward, 2)}")

    plt.imshow(np.concatenate(image_rows, 0))
    #plt.savefig(f"{Logger.directory_path + str(number)}.png", dpi=384)
    if verbose:
        plt.show()
    else:
        plt.close()



