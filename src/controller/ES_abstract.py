import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
from settings import *
import matplotlib.pyplot as plt


class ES_abstract():

    def __init__(self, loss_function, pop_size, args):
        """
        Abstract Evolution Strategy trainer
        :param loss_function (function) rollout of the Neurosmash environment, function to be optimized
        :param pop_size   (float) population size
        :param args
        """
        self.args = args
        self.pop_size = pop_size
        self.w_dim = z_dim + h_dim
        self.dim = int(w_dim * move_dim) + move_dim
        self.loss_func = loss_function

    def train(self, n_iter, parallel=False):
        """
        abstract function
        """
        raise NotImplementedError

    def plot_results(self, reward, NES=False):
        plt.figure(figsize=(8, 5))
        if NES:
            plt.title('Cumulative reward of Natural Evoltuion Strategies (NES)')
        else:
            plt.title(f'Cumulative reward of Evolution Strategies (ES)')
        plt.xlabel('Generation')
        plt.ylabel('Cumulative reward')

        # mean population
        plt.plot(np.arange(1, reward.shape[0] + 1), reward[:, 0], label="Population solution")
        # best performer
        plt.plot(np.arange(1, reward.shape[0] + 1), reward[:, 1], label="Best performer")
        # Worst performer
        plt.plot(np.arange(1, reward.shape[0] + 1), reward[:, 2], label="Worst performer")
        # sampled population average
        plt.plot(np.arange(1, reward.shape[0] + 1), reward[:, 3], label="Sampled population reward")

        plt.legend()
        if NES:
            plt.savefig('data/images/ControllerResults/controller_training_NES.png')
        else:
            plt.savefig('data/images/ControllerResults/controller_training_ES.png')

        plt.show()


    def get_reward_stats(self, w, fitness):
        reward = np.zeros(4)
        mean_controller = Controller(self.args)
        mean_controller.set_weight_array(w)
        reward[0] = self.loss_func(mean_controller, r_rounds=2)  # First entry is of mean weights, after that of pop. members
        reward[1] = np.max(fitness)  # best performer
        reward[2] = np.min(fitness)  # worst performer
        reward[3] = np.mean(fitness)  # sampled population mean
        return reward


