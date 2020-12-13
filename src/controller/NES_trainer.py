import numpy as np
np.random.seed(2020)
from multiprocessing import Pool
from tqdm import tqdm
import time
from controller.Controller import *
from settings import *
from controller.ES_trainer import ES_abstract


class NES_trainer(ES_abstract):

    def __init__(self, loss_function, pop_size, elite_size, args):
        """
        Natural Evolution Strategy algorithm
        :param loss_function (function) rollout of the Neurosmash environment, function to be optimized
        :param pop_size   (float) population size
        :param elite_size (float) elite group size
        :param args
        """
        super().__init__(loss_function, pop_size, elite_size, args)
        self.weights = np.random.normal(0, 1, self.dim)
        self.sigma = 1

    def train(self, n_iter, parallel=False):
        """
        :param n_iter    (int) number of iterations

        :return trained controller (Controller) trained controller object with trained weights
        :return fitness  (matrix) fitness score for each member for each iteration
        """
        raise NotImplementedError

